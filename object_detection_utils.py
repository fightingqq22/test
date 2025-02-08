from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import traceback


def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114), label_font: str = "LiberationSans-Regular.ttf"):
        """Initialize the ObjectDetectionUtils class."""
        self.labels = self.get_labels(labels_path)
        print("Available labels:", self.labels)
        self.padding_color = padding_color
        self.label_font = label_font
        self.model_input_size = (640, 640)
        self.class_colors = {
            0: (255, 0, 0),    # Red for Mickey
            1: (255, 192, 203) # Pink for Minnie
        }
    
    def get_labels(self, labels_path: str) -> list:
        """
        Load labels from a file or create default Mickey/Minnie labels.
        """
        
        # Default labels for Mickey/Minnie model
        class_names = ["Mickey", "Minnie"]
        return class_names
        
      
        
    def preprocess(self, image):
        """
        Preprocess image for YOLOv5 inference - keeping same as object_detection_utils_pic.py
        """
        try:
            # Convert to RGB if needed
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get original dimensions
            orig_h, orig_w = image.shape[:2]

            # Calculate scale
            scale = min(self.model_input_size[0] / orig_h, self.model_input_size[1] / orig_w)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)

            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Calculate padding
            pad_h = (self.model_input_size[0] - new_h) // 2
            pad_w = (self.model_input_size[1] - new_w) // 2

            # Create padded image
            padded_img = np.full((self.model_input_size[0], self.model_input_size[1], 3),
                               self.padding_color, dtype=np.uint8)
            padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized

            # Store padding info for later use
            self.pad_info = {
                'scale': scale,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'orig_shape': (orig_h, orig_w),
                'new_shape': (new_h, new_w)
            }

            # Add batch dimension and ensure uint8 type
            processed_img = np.expand_dims(padded_img, axis=0)
            return processed_img

        except Exception as e:
            print(f"Error in preprocess: {str(e)}")
            traceback.print_exc()
            return None


    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, 
                 stride=32, scaleup=True):
        """
        Resize and pad image while meeting stride-multiple constraints.
        Args:
            im: Input image
            new_shape: Desired output shape
            color: Padding color
            auto: Minimum rectangle
            stride: Stride constraint
            scaleup: Allow scale up
        Returns:
            Resized and padded image, ratio, (dw, dh)
        """
        # Current shape [height, width]
        shape = im.shape[:2]  
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
            
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up
            r = min(r, 1.0)
            
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
            
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add padding
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                               cv2.BORDER_CONSTANT,
                               value=color)
                               
        return im, ratio, (dw, dh)

    def draw_detection(self, draw: ImageDraw.Draw, box: list, cls: int, score: float):
        """
        Draw box and label for one detection with fixed colors for Mickey/Minnie.
        Args:
            draw: ImageDraw object
            box: list of [x1, y1, x2, y2] normalized coordinates
            cls: class index (0 for Mickey, 1 for Minnie)
            score: detection confidence score
        """
        try:
            # Get image dimensions for scaling
            img_width, img_height = draw._image.size
            
            # Convert normalized coordinates to image coordinates
            x1 = int(box[0] * img_width)
            y1 = int(box[1] * img_height)
            x2 = int(box[2] * img_width)
            y2 = int(box[3] * img_height)
            
            # Get color for class
            color = self.class_colors[cls]
            
            # Draw the bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            
            # Prepare label text
            label = f"{self.labels[cls]}: {score*100:.1f}%"
            
            try:
                font = ImageFont.truetype(self.label_font, size=15)
            except OSError:
                font = ImageFont.load_default()
                
            # Get text size
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Add padding for label background
            padding = 2
            
            # Draw label background
            draw.rectangle(
                [
                    (x1, y1),
                    (x1 + text_w + padding * 2, y1 + text_h + padding * 2)
                ],
                fill=color
            )
            
            # Draw label text
            draw.text(
                (x1 + padding, y1 + padding),
                label,
                font=font,
                fill='white'
            )
            
            print(f"Drew detection: class={self.labels[cls]}, score={score:.3f}, box=[{x1}, {y1}, {x2}, {y2}]")
            
        except Exception as e:
            print(f"Error in draw_detection: {str(e)}")
            traceback.print_exc()

    def _non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, 
                        iou_thres: float) -> np.ndarray:
        """
        Apply Non-Maximum Suppression with safety checks
        """
        # 检查输入是否为空
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
            
        # Convert to numpy arrays if not already
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)
        
        # Sort by score
        indices = np.argsort(-scores)
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]
        
        keep = []
        while len(boxes) > 0:
            keep.append(indices[0])
            
            if len(boxes) == 1:
                break
                
            # Calculate IoU between first box and all other boxes
            ious = np.array([self._calculate_iou(boxes[0], box) for box in boxes[1:]])
            
            # Keep boxes with IoU below threshold
            mask = ious <= iou_thres
            indices = indices[1:][mask]
            boxes = boxes[1:][mask]
            scores = scores[1:][mask]
            classes = classes[1:][mask]
        
        # 确保返回的是numpy数组
        keep = np.array(keep, dtype=np.int32)
        return keep
    
    def extract_detections(self, input_data: dict, conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
        """Parse YOLOv5 model output and process detections"""
        try:
            z = []
            for layer_name, detection_output in input_data.items():
                x = detection_output.astype(np.float32)
                x = (x - 128) / 128.0  # Scale as in pic version
                
                batch_size, grid_h, grid_w, channels = x.shape
                num_classes = 2  # Mickey, Minnie
                
                # Reshape output
                x = x.reshape(batch_size, grid_h, grid_w, 3, 5 + num_classes)
                
                # Create grid
                yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
                grid = np.stack((xv, yv), 2).reshape((1, grid_h, grid_w, 1, 2)).astype(np.float32)
                
                # Process predictions
                y = np.zeros_like(x, dtype=np.float32)
                
                # Box coordinates and scale
                stride = self.model_input_size[0] / grid_w
                anchors = self.get_anchors_for_stride(stride)
                
                y[..., 0:2] = (self.sigmoid(x[..., 0:2]) * 2 - 0.5 + grid) * stride
                y[..., 2:4] = (self.sigmoid(x[..., 2:4]) * 2) ** 2 * anchors[None, None, None, :, :]
                
                # Objectness and class scores
                y[..., 4] = self.sigmoid(x[..., 4])
                y[..., 5:] = self.sigmoid(x[..., 5:])
                
                # Normalize coordinates
                y[..., 0:4] = y[..., 0:4] / float(self.model_input_size[0])
                
                z.append(y.reshape(batch_size, -1, 5 + num_classes))

            # Process detections
            z = np.concatenate(z, axis=1)
            
            # Calculate confidence scores
            obj_conf = z[..., 4]
            cls_conf = z[..., 5:]
            cls_scores = obj_conf[..., None] * cls_conf
            max_scores = np.max(cls_scores, axis=-1)
            
            # Filter by confidence
            mask = max_scores > conf_thres
            if not np.any(mask):
                print("No detections above confidence threshold")
                return {
                    'detection_boxes': np.array([]),
                    'detection_classes': np.array([]),
                    'detection_scores': np.array([]),
                    'num_detections': 0
                }
            
            # Get filtered predictions
            filtered_boxes = z[..., :4][mask]
            filtered_scores = max_scores[mask]
            filtered_classes = np.argmax(cls_scores[mask], axis=-1)
            
            # Convert center/width/height to x1/y1/x2/y2
            boxes_xy = filtered_boxes[..., :2]
            boxes_wh = filtered_boxes[..., 2:4]
            x1y1 = boxes_xy - boxes_wh / 2
            x2y2 = boxes_xy + boxes_wh / 2
            pred_boxes = np.concatenate([x1y1, x2y2], axis=1)
            
            # Apply NMS
            final_boxes = []
            final_scores = []
            final_classes = []
            
            indices = np.argsort(-filtered_scores)
            while len(indices) > 0:
                # Keep highest scoring box
                final_boxes.append(pred_boxes[indices[0]])
                final_scores.append(filtered_scores[indices[0]])
                final_classes.append(filtered_classes[indices[0]])
                
                if len(indices) == 1:
                    break
                    
                # Calculate IoU between highest scoring box and remaining boxes
                ious = np.array([self._calculate_iou(pred_boxes[indices[0]], pred_boxes[i]) 
                               for i in indices[1:]])
                
                # Keep boxes with IoU below threshold
                indices = indices[1:][ious <= iou_thres]
            
            # Convert lists to numpy arrays
            final_boxes = np.array(final_boxes)
            final_scores = np.array(final_scores)
            final_classes = np.array(final_classes)
            
            # Scale boxes to original image size if needed
            if hasattr(self, 'pad_info'):
                final_boxes = self.scale_boxes_to_original(final_boxes)
                
            # Print debug information
            print(f"\nFinal detections:")
            print(f"Number of detections: {len(final_boxes)}")
            for i, (box, cls, score) in enumerate(zip(final_boxes, final_classes, final_scores)):
                print(f"Detection {i}:")
                print(f"- Class: {self.labels[int(cls)]}")
                print(f"- Score: {score:.3f}")
                print(f"- Box: {box}")
            
            return {
                'detection_boxes': final_boxes.astype(np.float32),
                'detection_classes': final_classes.astype(np.int32),
                'detection_scores': final_scores.astype(np.float32),
                'num_detections': len(final_boxes)
            }
                
        except Exception as e:
            print(f"Error in extract_detections: {str(e)}")
            traceback.print_exc()
            return {
                'detection_boxes': np.array([]),
                'detection_classes': np.array([]),
                'detection_scores': np.array([]),
                'num_detections': 0
            }


    def scale_boxes_to_original(self, boxes):
        """
        Scale boxes from normalized coordinates back to original image size
        Args:
            boxes: array of shape (N, 4) containing normalized coordinates [x1, y1, x2, y2]
        Returns:
            Scaled boxes in original image coordinates
        """
        try:
            # Print input boxes
            print("\n=== Scaling Boxes ===")
            print(f"Input boxes (normalized):")
            print(boxes)
            print(f"Pad info: {self.pad_info}")
            
            # Make a copy to avoid modifying the original
            boxes = boxes.copy()
            
            # First convert normalized coordinates (0-1) to model input size coordinates
            boxes[..., [0, 2]] *= self.model_input_size[1]  # scale x coordinates
            boxes[..., [1, 3]] *= self.model_input_size[0]  # scale y coordinates
            
            print(f"\nAfter scaling to model input size:")
            print(boxes)
            
            # Remove padding
            boxes[..., [0, 2]] -= self.pad_info['pad_w']  # x coordinates
            boxes[..., [1, 3]] -= self.pad_info['pad_h']  # y coordinates
            
            print(f"\nAfter removing padding:")
            print(boxes)
            
            # Apply inverse scale to get back to original image size
            scale = 1.0 / self.pad_info['scale']
            boxes *= scale
            
            print(f"\nAfter applying inverse scale ({scale}):")
            print(boxes)
            
            # Clip coordinates to image bounds
            boxes[..., [0, 2]] = np.clip(
                boxes[..., [0, 2]], 
                0, 
                self.pad_info['orig_shape'][1]
            )
            boxes[..., [1, 3]] = np.clip(
                boxes[..., [1, 3]], 
                0, 
                self.pad_info['orig_shape'][0]
            )
            
            print(f"\nFinal boxes (after clipping):")
            print(boxes)
            print(f"Original image shape: {self.pad_info['orig_shape']}")
            print(f"X range: [{boxes[..., [0, 2]].min()}, {boxes[..., [0, 2]].max()}]")
            print(f"Y range: [{boxes[..., [1, 3]].min()}, {boxes[..., [1, 3]].max()}]")
            
            return boxes
            
        except Exception as e:
            print(f"Error in scale_boxes_to_original: {str(e)}")
            traceback.print_exc()
            return boxes

    def get_anchors_for_stride(self, stride: float) -> np.ndarray:
        """Get anchors for specific stride level"""
        if stride == 32:  # 8x reduction
            return np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32)
        elif stride == 16:  # 16x reduction  
            return np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32)
        else:  # stride == 8
            return np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32)

    def sigmoid(self, x):
        """
        Compute sigmoid activation with better numerical stability and scaling
        """
        x = np.clip(x, -88, 88)
        # 添加缩放因子以提高置信度
        x = x * 2.0  # 增加斜率，使得更容易产生极值
        return 1.0 / (1.0 + np.exp(-x))
        
        
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-7)
        
        
        
    def visualize_video(self, detections, image, width, height):
        """Visualize detections on video frame"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            draw = ImageDraw.Draw(image)
            
            print("\nDrawing detections:")
            for i in range(detections['num_detections']):
                box = detections['detection_boxes'][i]
                cls = detections['detection_classes'][i]
                score = detections['detection_scores'][i]
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                print(f"Detection {i}:")
                print(f"Class: {self.labels[cls]}")
                print(f"Score: {score:.3f}")
                print(f"Box: [{x1}, {y1}, {x2}, {y2}]")
                
                # Draw thick bounding box
                color = self.class_colors[int(cls)]
                for thickness in range(3):
                    draw.rectangle(
                        [(x1 + thickness, y1 + thickness),
                         (x2 - thickness, y2 - thickness)],
                        outline=color,
                        width=2
                    )
                
                # Draw label
                label = f"{self.labels[cls]}: {score*100:.1f}%"
                try:
                    font = ImageFont.truetype(self.label_font, size=15)
                except OSError:
                    font = ImageFont.load_default()
                
                # Get text size
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw label background
                draw.rectangle(
                    [(x1, y1 - text_height - 4),
                     (x1 + text_width + 4, y1)],
                    fill=color
                )
                
                # Draw text
                draw.text((x1 + 2, y1 - text_height - 2),
                         label, fill='white', font=font)
            
            return image
            
        except Exception as e:
            print(f"Error in visualize_video: {str(e)}")
            traceback.print_exc()
            return image