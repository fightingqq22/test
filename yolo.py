import numpy as np
import tensorflow as tf
from tensorflow.image import combined_non_max_suppression

from detection_common import tf_postproc_nms

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class YoloPostProc(object):
    def __init__(
        self,
        img_dims=(640, 640),  # 根据您的输入分辨率
        nms_iou_thresh=0.45,
        score_threshold=0.01,
        anchors=None,
        output_scheme=None,
        classes=2,  # 修改为2个类别
        labels_offset=0,
        meta_arch="yolo_v5",
        should_clip=True,
        **kwargs,
    ):
        self._network_arch = meta_arch
        self._image_dims = img_dims
        self._nms_iou_thresh = nms_iou_thresh
        self.score_threshold = score_threshold
        if anchors is None or anchors["strides"] is None:
            raise ValueError("Missing detection anchors/strides metadata")
        self._anchors_list = anchors["sizes"]
        self._strides = anchors["strides"]
        self._num_classes = classes  # 现在是2
        self._labels_offset = labels_offset
        self._yolo_decoding = {
            "yolo_v5": YoloPostProc._yolo5_decode,
        }
        self._nms_on_device = False
        self._should_clip = should_clip

    @staticmethod
    def _yolo3_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride):
        box_centers = (sigmoid(raw_box_centers) + offsets) * stride
        box_scales = np.exp(raw_box_scales) * anchors_for_stride
        confidence = sigmoid(objness)
        class_pred = sigmoid(class_pred)
        return box_centers, box_scales, confidence, class_pred

    @staticmethod
    def _yolo4_decode(
        raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride, scale_x_y=1.05
    ):
        box_scales = np.exp(raw_box_scales) * anchors_for_stride
        box_centers = (
            sigmoid(raw_box_centers) * scale_x_y - 0.5 * (scale_x_y - 1) + offsets
        ) * stride
        confidence = sigmoid(objness)
        class_pred = sigmoid(class_pred)
        return box_centers, box_scales, confidence, class_pred

    @staticmethod
    def _yolo5_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride):
        box_centers = (raw_box_centers * 2.0 - 0.5 + offsets) * stride  # [BS, H*W, 1, 2]
        box_scales = (raw_box_scales * 2) ** 2 * anchors_for_stride     # [BS, H*W, 1, 2]
        confidence = objness.reshape(objness.shape[0], objness.shape[1], 1)  # [BS, H*W, 1]
        return box_centers, box_scales, confidence, class_pred

    @staticmethod
    def _yolox_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride):
        box_centers = (raw_box_centers + offsets) * stride
        box_scales = np.exp(raw_box_scales) * stride
        return box_centers, box_scales, objness, class_pred

    @staticmethod
    def _yolo6_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride):
        x1y1 = offsets + 0.5 - raw_box_centers
        x2y2 = offsets + 0.5 + raw_box_scales
        box_centers = ((x1y1 + x2y2) / 2) * stride
        box_scales = (x2y2 - x1y1) * stride
        return box_centers, box_scales, objness, class_pred

    def iou_nms(self, detection_boxes, detection_scores):
        (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = combined_non_max_suppression(
            boxes=detection_boxes,
            scores=detection_scores,
            score_threshold=self.score_threshold,
            iou_threshold=self._nms_iou_thresh,
            max_output_size_per_class=100,
            max_total_size=100,
        )

        nmsed_classes = tf.cast(tf.add(nmsed_classes, self._labels_offset), tf.int16)
        return {
            "detection_boxes": nmsed_boxes,
            "detection_scores": nmsed_scores,
            "detection_classes": nmsed_classes,
            "num_detections": num_detections,
        }

    def yolo_postprocessing(self, endnodes, **kwargs):
        """
        处理三个特征图:
        - yolov5m/conv65: (1, 80, 80, 21) -> stride 8
        - yolov5m/conv74: (1, 40, 40, 21) -> stride 16
        - yolov5m/conv82: (1, 20, 20, 21) -> stride 32
        
        自定义数据集:
        - class 0: Mickey
        - class 1: Minnie
        """
        H_input = self._image_dims[0]
        W_input = self._image_dims[1]
        anchors_list = self._anchors_list
        strides = self._strides
        num_classes = self._num_classes
        
        # 按stride从大到小排序endnodes
        sorted_indices = np.argsort([node.shape[1] for node in endnodes])[::-1]
        endnodes = [endnodes[i] for i in sorted_indices]
        
        all_detection_boxes = []
        all_detection_scores = []
        
        for output_ind, output_branch in enumerate(endnodes):
            stride = strides[::-1][output_ind]
            anchors_for_stride = np.array(anchors_list[::-1][output_ind])
            anchors_for_stride = np.reshape(anchors_for_stride, (1, 1, -1, 2))
            
            H = output_branch.shape[1]
            W = output_branch.shape[2]
            num_detections = H * W
            
            output_branch_and_data = [output_branch, anchors_for_stride, stride]
            detection_boxes, detection_scores = tf.numpy_function(
                self.yolo_postprocess_numpy,
                output_branch_and_data,
                ["float32", "float32"],
                name=f"{self._network_arch}_postprocessing",
            )
            
            # 设置正确的shape
            BS = output_branch.shape[0]
            detection_boxes.set_shape((BS, num_detections, 1, 4))
            detection_scores.set_shape((BS, num_detections, self._num_classes))
            
            all_detection_boxes.append(detection_boxes)
            all_detection_scores.append(detection_scores)
        
        # 连接所有特征图的结果
        detection_boxes_full = tf.concat(all_detection_boxes, axis=1)
        detection_scores_full = tf.concat(all_detection_scores, axis=1)
        
        # 应用NMS
        (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = tf.image.combined_non_max_suppression(
        boxes=detection_boxes_full,
        scores=detection_scores_full,
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=self._nms_iou_thresh,
        score_threshold=self.score_threshold,
        clip_boxes=self._should_clip
    )
    
        # 转换类别为整数
        nmsed_classes = tf.cast(nmsed_classes, tf.int32)
        
        return {
            'boxes': nmsed_boxes,           # 改用简单的键名
            'scores': nmsed_scores,
            'classes': nmsed_classes,
            'num_detections': num_detections
        }

    def yolo_postprocess_numpy(self, net_out, anchors_for_stride, stride):
        BS = net_out.shape[0]  # 1
        H = net_out.shape[1]   # 80/40/20
        W = net_out.shape[2]   # 80/40/20
        C = net_out.shape[3]   # 21
        
        # 创建网格
        grid_x = np.arange(W)
        grid_y = np.arange(H)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        offsets = np.stack((grid_x, grid_y), axis=-1)  # [H,W,2]
        offsets = offsets.reshape(1, H*W, 1, 2)  # [1,H*W,1,2]
        
        # 重组预测输出
        pred = net_out.reshape(BS, H*W, C)
        
        # 分离预测组件
        raw_box_centers = pred[:, :, 0:2].reshape(BS, H*W, 1, 2)  # x,y
        raw_box_scales = pred[:, :, 2:4].reshape(BS, H*W, 1, 2)   # w,h
        objness = pred[:, :, 4:5].reshape(BS, H*W, 1, 1)
        class_pred = pred[:, :, 5:7].reshape(BS, H*W, 2)  # 只取2个类别
        
        # 应用解码获取归一化坐标
        box_centers, box_scales, confidence, class_pred = self._yolo_decoding[self._network_arch](
            raw_box_centers=raw_box_centers,
            raw_box_scales=raw_box_scales,
            objness=objness,
            class_pred=class_pred,
            anchors_for_stride=anchors_for_stride,
            offsets=offsets,
            stride=stride
        )
        
        # 计算最终的检测框和分数
        confidence = confidence.reshape(BS, H*W, 1)
        class_score = class_pred * confidence
        
        # 转换为x1,y1,x2,y2格式
        wh = box_scales / 2.0
        bbox = np.concatenate((box_centers - wh, box_centers + wh), axis=-1)  # x1,y1,x2,y2
        
        # 确保输出形状正确
        detection_boxes = bbox.reshape(BS, H*W, 1, 4)
        detection_scores = class_score
        
        # 修改坐标转换，保持[x1,y1,x2,y2]顺序
        detection_boxes_tmp = np.zeros(detection_boxes.shape)
        detection_boxes_tmp[:, :, :, 0] = detection_boxes[:, :, :, 0] / self._image_dims[1]  # x1
        detection_boxes_tmp[:, :, :, 1] = detection_boxes[:, :, :, 1] / self._image_dims[0]  # y1
        detection_boxes_tmp[:, :, :, 2] = detection_boxes[:, :, :, 2] / self._image_dims[1]  # x2
        detection_boxes_tmp[:, :, :, 3] = detection_boxes[:, :, :, 3] / self._image_dims[0]  # y2
        
        # 裁剪到0-1范围
        detection_boxes_tmp = np.clip(detection_boxes_tmp, 0, 1)
        
        return detection_boxes_tmp.astype(np.float32), detection_scores.astype(np.float32)

    def reorganize_split_output(self, endnodes):
        reorganized_endnodes_list = []
        for index in range(len(self._anchors_list)):
            branch_index = int(4 * index)
            if "yolox" in self._network_arch:
                branch_index = int(3 * index)
                centers = endnodes[branch_index][:, :, :, :2]
                scales = endnodes[branch_index][:, :, :, 2:]
                obj = endnodes[branch_index + 1]
                probs = endnodes[branch_index + 2]
            elif "yolo_v6" in self._network_arch:
                branch_index = int(2 * index)
                centers = endnodes[branch_index][:, :, :, :2]
                scales = endnodes[branch_index][:, :, :, 2:]
                probs = endnodes[branch_index + 1]
                obj = tf.ones((1, 1, 1, 2))
            else:
                centers = endnodes[branch_index]
                scales = endnodes[branch_index + 1]
                obj = endnodes[branch_index + 2]
                probs = endnodes[branch_index + 3]
            branch_endnodes = tf.numpy_function(
                self.reorganize_split_output_numpy,
                [centers, scales, obj, probs],
                ["float32"],
                name="yolov3_match_remodeled_output",
            )

            reorganized_endnodes_list.append(branch_endnodes[0])
        return reorganized_endnodes_list

    def reorganize_split_output_numpy(self, centers, scales, obj, probs):
        num_anchors = len(self._anchors_list[0]) // 2
        if obj.shape == (1, 1, 1, 2):
            obj = np.ones((list(probs.shape[:3]) + [1]), dtype=np.float32)
        for anchor in range(num_anchors):
            concat_arrays_for_anchor = [
                centers[:, :, :, 2 * anchor : 2 * anchor + 2],
                scales[:, :, :, 2 * anchor : 2 * anchor + 2],
                obj[:, :, :, anchor : anchor + 1],
                probs[:, :, :, anchor * self._num_classes : (anchor + 1) * self._num_classes],
            ]

            partial_concat = np.concatenate(concat_arrays_for_anchor, 3)

            if anchor == 0:
                full_concat_array = partial_concat
            else:
                full_concat_array = np.concatenate([full_concat_array, partial_concat], 3)
        return full_concat_array

    def hpp_detection_postprocess(self, endnodes):
        endnodes = tf.transpose(endnodes, [0, 1, 3, 2])
        detection_boxes = endnodes[:, 0, :, :4]
        detection_scores = endnodes[:, 0, :, 4]
        detection_classes = endnodes[:, 0, :, 5]

        num_detections = tf.reduce_sum(tf.cast(detection_scores > 0, dtype=tf.int32), axis=1)
        nmsed_classes = tf.cast(tf.add(detection_classes, self._labels_offset), tf.int16)

        return {
            "detection_boxes": detection_boxes,
            "detection_scores": detection_scores,
            "detection_classes": nmsed_classes,
            "num_detections": num_detections,
        }

    def postprocessing(self, endnodes, **kwargs):
        if self.hpp:
            if kwargs.get("bbox_decoding_only", False):
                endnodes = tf.squeeze(endnodes, axis=1)
                detection_boxes = endnodes[:, :, None, :4]
                detection_scores = endnodes[..., 4:5] * endnodes[..., 5:]
                return self.iou_nms(detection_boxes, detection_scores)

            return tf_postproc_nms(
                endnodes, labels_offset=kwargs["labels_offset"], score_threshold=0.0, coco_2017_to_2014=False
            )
        if self._nms_on_device:
            endnodes = tf.transpose(endnodes, [0, 3, 1, 2])
            detection_boxes = endnodes[:, :, :, :4]
            detection_scores = tf.squeeze(endnodes[:, :, :, 4:], axis=3)
            return self.iou_nms(detection_boxes, detection_scores)
        else:
            return self.yolo_postprocessing(endnodes, **kwargs)
