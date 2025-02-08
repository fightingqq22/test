from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os


def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114), label_font: str = "LiberationSans-Regular.ttf"):
        """
        Initialize the ObjectDetectionUtils class.
        """
        self.labels = self.get_labels(labels_path)
        print("Available labels:", self.labels)
        self.padding_color = padding_color
        self.label_font = label_font
        self.model_input_size = (640, 640)  # YOLOv5默认输入尺寸
        # Define fixed colors for Mickey and Minnie
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
        
      
        
    def preprocess(self, image, model_size=(640, 640)):
        """
        Preprocess image for YOLOv5 inference with RGBA handling
        Args:
            image: Input PIL Image
            model_size: Model input size (height, width)
        Returns:
            Preprocessed image as numpy array
        """
        try:
            print("\n=== Debug Info: Preprocessing Steps ===")
            
            # Print input image type and initial info
            print(f"1. Input image type: {type(image)}")
            if isinstance(image, Image.Image):
                print(f"   PIL Image size: {image.size}")
                print(f"   PIL Image mode: {image.mode}")
                # Convert RGBA/LA to RGB/L
                if image.mode in ('RGBA', 'LA'):
                    print("   Converting RGBA to RGB")
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])  # Using alpha channel as mask
                    image = background
                elif image.mode != 'RGB':
                    print(f"   Converting {image.mode} to RGB")
                    image = image.convert('RGB')
                    
                image = np.array(image)
                print(f"   After conversion to numpy: shape={image.shape}, dtype={image.dtype}")
                
            # Print numpy array info
            print(f"2. Numpy array initial shape: {image.shape}")
            print(f"   Data type: {image.dtype}")
            print(f"   Value range: [{np.min(image)}, {np.max(image)}]")
            
            # Ensure 3 channels
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
            
            print(f"   After channel check: shape={image.shape}")
            
            # Convert to BGR if in RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print("3. Converted to BGR")
                print(f"   New shape: {image.shape}")
                
            # Get original image size
            orig_h, orig_w = image.shape[:2]
            print(f"4. Original dimensions: height={orig_h}, width={orig_w}")
            
            # Calculate scale ratio
            scale = min(model_size[0] / orig_h, model_size[1] / orig_w)
            print(f"5. Calculated scale: {scale}")
            
            # Calculate new size
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            print(f"6. New dimensions: height={new_h}, width={new_w}")
            
            # Initialize padding
            pad_h = (model_size[0] - new_h) // 2
            pad_w = (model_size[1] - new_w) // 2
            print(f"7. Padding: top/bottom={pad_h}, left/right={pad_w}")
            
            # Create padded image
            padded_img = np.full((model_size[0], model_size[1], 3), 
                                self.padding_color,
                                dtype=np.uint8)
            print(f"8. Padded image shape: {padded_img.shape}")
            
            # Resize original image
            resized = cv2.resize(image, (new_w, new_h),
                                interpolation=cv2.INTER_LINEAR)
            print(f"9. Resized image shape: {resized.shape}")
            
            # Place resized image in padded image
            padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
            print(f"10. After placing resized image in padding: shape={padded_img.shape}")
            
            # Store padding info
            self.pad_info = {
                'scale': scale,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'orig_shape': (orig_h, orig_w)
            }
            
            # Convert BGR to RGB
            processed_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
            print("11. Converted back to RGB")
            
            # Keep as uint8 type to match expected input format
            processed_img = processed_img.astype(np.uint8)
            print(f"12. Final dtype: {processed_img.dtype}")
            print(f"    Value range: [{processed_img.min()}, {processed_img.max()}]")
            
            # Add batch dimension
            processed_img = np.expand_dims(processed_img, axis=0)
            print(f"13. Final output shape: {processed_img.shape}")
            print(f"    Size in bytes: {processed_img.nbytes}")
            
            # Calculate expected size
            expected_size = 1228800  # 640*640*3
            print(f"\n=== Size Comparison ===")
            print(f"Expected size: {expected_size} bytes")
            print(f"Actual size: {processed_img.nbytes} bytes")
            
            if processed_img.nbytes != expected_size:
                print(f"WARNING: Size mismatch detected!")
                print(f"Input shape: {processed_img.shape}")
                print(f"Input dtype: {processed_img.dtype}")
                print(f"Expected shape should be: (1, 640, 640, 3) with dtype uint8")
            
            return processed_img
            
        except Exception as e:
            print(f"\n=== Error in preprocess ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            import traceback
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

    def _non_max_suppression(self, boxes, scores, classes, iou_thres):
        """
        Apply Non-Maximum Suppression.
        """
        # Convert to numpy arrays
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
            keep.append(0)  # Keep the box with highest score
            
            if len(boxes) == 1:
                break
                
            ious = np.array([self._calculate_iou(boxes[0], box) for box in boxes[1:]])
            filtered_indices = np.where(ious <= iou_thres)[0] + 1
            
            boxes = boxes[filtered_indices]
            scores = scores[filtered_indices]
            classes = classes[filtered_indices]
        
        keep = indices[keep]
        return boxes[keep], scores[keep], classes[keep]
    
    def extract_detections(self, input_data: dict, conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
        """
        Parse YOLOv5 model output and process detections with improved confidence calculation
        """
        z = []
        print("\n=== Detection Processing Debug ===")
        
        try:
            for layer_name, detection_output in input_data.items():
                print(f"\nProcessing layer: {layer_name}")
                # 将输入值调整到合适的范围
                x = detection_output.astype(np.float32)
                x = (x - 128) / 128.0  # 中心化并缩放
                
                batch_size, grid_h, grid_w, channels = x.shape
                num_classes = 2  # Mickey, Minnie
                
                print(f"Raw output shape: {x.shape}")
                print(f"Raw output range: [{x.min():.3f}, {x.max():.3f}]")
                
                # Get stride and anchors
                stride = self.model_input_size[0] / grid_w
                anchors = self.get_anchors_for_stride(stride)
                
                # Reshape
                x = x.reshape(batch_size, grid_h, grid_w, 3, 5 + num_classes)
                
                # Create grid
                yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
                grid = np.stack((xv, yv), 2).reshape((1, grid_h, grid_w, 1, 2)).astype(np.float32)
                
                # Process predictions
                y = np.zeros_like(x, dtype=np.float32)
                
                # Box coordinates
                y[..., 0:2] = (self.sigmoid(x[..., 0:2]) * 2 - 0.5 + grid) * stride
                y[..., 2:4] = (self.sigmoid(x[..., 2:4]) * 2) ** 2 * anchors[None, None, None, :, :]
                
                # Objectness score with scaling
                raw_obj = x[..., 4]
                y[..., 4] = self.sigmoid(raw_obj)
                print(f"Objectness range: raw=[{raw_obj.min():.3f}, {raw_obj.max():.3f}], "
                      f"sigmoid=[{y[..., 4].min():.3f}, {y[..., 4].max():.3f}]")
                
                # Class probabilities with scaling
                raw_cls = x[..., 5:]
                y[..., 5:] = self.sigmoid(raw_cls)
                print(f"Class scores range: raw=[{raw_cls.min():.3f}, {raw_cls.max():.3f}], "
                      f"sigmoid=[{y[..., 5:].min():.3f}, {y[..., 5:].max():.3f}]")
                
                # Normalize coordinates
                y[..., 0:4] = y[..., 0:4] / float(self.model_input_size[0])
                
                z.append(y.reshape(batch_size, -1, 5 + num_classes))
        
            # Concatenate predictions from different scales
            z = np.concatenate(z, axis=1)
            print(f"\nConcatenated predictions shape: {z.shape}")
            
            # Calculate confidence scores
            obj_conf = z[..., 4]
            cls_conf = z[..., 5:]
            cls_scores = obj_conf[..., None] * cls_conf
            max_scores = np.max(cls_scores, axis=-1)
            
            print(f"\nConfidence distribution:")
            print(f"- Objectness: min={obj_conf.min():.3f}, max={obj_conf.max():.3f}")
            print(f"- Class confidence: min={cls_conf.min():.3f}, max={cls_conf.max():.3f}")
            print(f"- Final scores: min={max_scores.min():.3f}, max={max_scores.max():.3f}")
            
            # Filter by confidence
            mask = max_scores > conf_thres
            if not np.any(mask):
                print("No detections above confidence threshold")
                return {'detection_boxes': np.array([]), 
                       'detection_classes': np.array([]), 
                       'detection_scores': np.array([]), 
                       'num_detections': 0}
            
            # Get filtered predictions
            filtered_boxes = z[..., :4][mask]
            filtered_scores = max_scores[mask]
            filtered_classes = np.argmax(cls_scores[mask], axis=-1)
            
            print("\nBefore NMS:")
            print(f"Number of detections: {len(filtered_boxes)}")
            print(f"Score range: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}")
            
            # Convert to corners format (x1, y1, x2, y2)
            pred_boxes = np.zeros_like(filtered_boxes)
            pred_boxes[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x1
            pred_boxes[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y1
            pred_boxes[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x2
            pred_boxes[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y2
            
            # Clip to image bounds
            pred_boxes = np.clip(pred_boxes, 0, 1)
            
            # Sort by confidence
            idxs = np.argsort(-filtered_scores)
            pred_boxes = pred_boxes[idxs]
            filtered_scores = filtered_scores[idxs]
            filtered_classes = filtered_classes[idxs]
            
            # Apply NMS
            final_boxes = []
            final_scores = []
            final_classes = []
            
            while len(pred_boxes) > 0:
                final_boxes.append(pred_boxes[0])
                final_scores.append(filtered_scores[0])
                final_classes.append(filtered_classes[0])
                
                if len(pred_boxes) == 1:
                    break
                
                ious = np.array([self._calculate_iou(pred_boxes[0], box) for box in pred_boxes[1:]])
                print(f"IoUs with first box: {ious}")
                
                keep_mask = ious <= iou_thres
                pred_boxes = pred_boxes[1:][keep_mask]
                filtered_scores = filtered_scores[1:][keep_mask]
                filtered_classes = filtered_classes[1:][keep_mask]
                
                print(f"Boxes remaining after IoU filter: {len(pred_boxes)}")
            
            final_boxes = np.array(final_boxes)
            final_scores = np.array(final_scores)
            final_classes = np.array(final_classes)
            
            print("\nAfter NMS:")
            print(f"Number of detections: {len(final_boxes)}")
            for i in range(len(final_boxes)):
                print(f"Detection {i}:")
                print(f"- Box (normalized): {final_boxes[i]}")
                print(f"- Class: {self.labels[final_classes[i]]}")
                print(f"- Score: {final_scores[i]:.3f}")
            
            return {
                'detection_boxes': final_boxes.astype(np.float32),
                'detection_classes': final_classes.astype(np.int32),
                'detection_scores': final_scores.astype(np.float32),
                'num_detections': len(final_boxes)
            }
            
        except Exception as e:
            print(f"Error in extract_detections: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'detection_boxes': np.array([], dtype=np.float32),
                'detection_classes': np.array([], dtype=np.int32),
                'detection_scores': np.array([], dtype=np.float32),
                'num_detections': 0
            }

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        """
        Rescale boxes (xyxy) from img1_shape to img0_shape
        Args:
            img1_shape: Shape of preprocessed image (height, width)
            boxes: Boxes in xyxy format 
            img0_shape: Shape of original image (height, width)
        Returns:
            Rescaled boxes
        """
        # Make a copy to avoid modifying original boxes
        boxes = boxes.copy()
        
        # Calculate gain (how much we scaled up/down)
        gain = min(img1_shape[0] / img0_shape[0], 
                  img1_shape[1] / img0_shape[1])
        
        # Calculate padding
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,  # width padding
            (img1_shape[0] - img0_shape[0] * gain) / 2   # height padding
        )
        
        print(f"Scale boxes - gain: {gain:.3f}, padding: {pad}")
        
        # Remove padding
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        
        # Scale down
        boxes[..., :4] /= gain
        
        # Clip boxes to image bounds
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])  # y1, y2
        
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
        
        
        
    def visualize_video(
        self,
        detections: dict,
        image: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Visualize detections for video frames with proper coordinate handling.
        Similar to visualize() but optimized for video processing.
        """
        try:
            # Convert image to PIL format if needed
            if isinstance(image, np.ndarray):
                from PIL import Image
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Create drawing context
            draw = ImageDraw.Draw(image)
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Process each detection
            boxes = detections['detection_boxes']
            classes = detections['detection_classes']
            scores = detections['detection_scores']
            num_detections = detections['num_detections']
            
            for i in range(num_detections):
                # Get normalized coordinates
                box = boxes[i]
                
                # Convert to pixel coordinates
                x1 = int(box[0] * img_width)
                y1 = int(box[1] * img_height)
                x2 = int(box[2] * img_width)
                y2 = int(box[3] * img_height)
                
                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # Get color for class
                color = self.class_colors[int(classes[i])]
                
                # Draw bounding box
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                
                # Prepare and draw label
                label = f"{self.labels[int(classes[i])]}: {scores[i]*100:.1f}%"
                
                try:
                    font = ImageFont.truetype(self.label_font, size=15)
                except OSError:
                    font = ImageFont.load_default()
                
                # Get text size
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # Draw label background and text
                padding = 2
                draw.rectangle(
                    [
                        (x1, y1),
                        (x1 + text_w + padding * 2, y1 + text_h + padding * 2)
                    ],
                    fill=color
                )
                draw.text(
                    (x1 + padding, y1 + padding),
                    label,
                    font=font,
                    fill='white'
                )
            
            return image
            
        except Exception as e:
            logger.error(f"Error in visualize_video: {str(e)}")
            return image