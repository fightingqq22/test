from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SceneSegment:
    start_frame: int
    end_frame: int
    bboxes: List[Tuple[float, float, float, float]]  # List of (x, y, w, h)


class SceneDetector:
    def __init__(self, max_gap_frames=3, min_confidence=0.5):
        """Initialize scene detector.
        
        Args:
            max_gap_frames (int): Maximum frames without Mickey before splitting scenes
            min_confidence (float): Minimum confidence threshold for valid Mickey detection
        """
        self.max_gap_frames = max_gap_frames
        self.min_confidence = min_confidence
        self.current_scene = None
        self.scenes = []
        self.gap_count = 0
        print(f"Initialized SceneDetector with max_gap_frames={max_gap_frames}, min_confidence={min_confidence}")

    def process_frame(self, frame_number: int, detections: dict) -> None:
        """Process detection results and update scene information."""
        # Find Mickey detection with highest confidence
        mickey_detected = False
        mickey_info = None
        
        for i in range(detections['num_detections']):
            cls = detections['detection_classes'][i]
            score = detections['detection_scores'][i]
            box = detections['detection_boxes'][i]
            
            if cls == 0 and score >= self.min_confidence:  # Mickey class
                mickey_detected = True
                # Convert [x1,y1,x2,y2] to [center_x,center_y,width,height]
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                mickey_info = (center_x, center_y, width, height, score)
                break

        # Update scene information
        if mickey_detected and mickey_info is not None:
            x, y, w, h, _ = mickey_info
            if self.current_scene is None:
                # Start new scene
                self.current_scene = SceneSegment(frame_number, frame_number, [(x, y, w, h)])
                print(f"Starting new scene at frame {frame_number}")
                self.gap_count = 0
            else:
                # Continue current scene
                self.current_scene.end_frame = frame_number
                self.current_scene.bboxes.append((x, y, w, h))
                self.gap_count = 0
        else:
            if self.current_scene is not None:
                self.gap_count += 1
                print(f"No Mickey detected in frame {frame_number}, gap_count: {self.gap_count}")
                if self.gap_count > self.max_gap_frames:
                    # End current scene
                    print(f"Ending scene at frame {frame_number-self.max_gap_frames} due to gap")
                    self.scenes.append(self.current_scene)
                    self.current_scene = None
                    self.gap_count = 0

    def finalize(self) -> None:
        """Finalize scene detection and add last scene if exists"""
        if self.current_scene is not None:
            print(f"Finalizing: Adding last scene (frames {self.current_scene.start_frame}-{self.current_scene.end_frame})")
            self.scenes.append(self.current_scene)
            self.current_scene = None
        
        # 打印检测到的场景信息
        print(f"\nScene detection completed. Found {len(self.scenes)} scenes:")
        for i, scene in enumerate(self.scenes):
            print(f"Scene {i}: frames {scene.start_frame}-{scene.end_frame} ({len(scene.bboxes)} detections)")



class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (255, 255, 255)):  # 使用白色填充
        """Initialize the ObjectDetectionUtils class."""
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        self.model_input_size = (640, 640)
        self.class_colors = {
            0: (255, 0, 0),    # Red for Mickey
            1: (255, 192, 203)  # Pink for Minnie
        }
    
    def get_labels(self, labels_path: str) -> list:
        """Load labels from a file or create default Mickey/Minnie labels."""
        # Default labels for Mickey/Minnie model
        class_names = ["Mickey", "Minnie"]
        return class_names

    def preprocess(self, image):
        """Preprocess image for YOLOv5 inference."""
        try:
            # 保持 BGR 格式，与训练时保持一致
            orig_h, orig_w = image.shape[:2]

            # Calculate scale while maintaining aspect ratio
            scale = min(self.model_input_size[0] / orig_h, self.model_input_size[1] / orig_w)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)

            # Resize using INTER_LINEAR
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Calculate padding to center the image
            pad_h = (self.model_input_size[0] - new_h) // 2
            pad_w = (self.model_input_size[1] - new_w) // 2

            # Create padded image with white padding (BGR format)
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
            processed_img = np.expand_dims(padded_img, axis=0).astype(np.uint8)
            return processed_img

        except Exception as e:
            print(f"Error in preprocess: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_crop_coordinates(self, frame_width: int, frame_height: int, 
                                 mickey_x: int, mickey_y: int, 
                                 bbox_width: int, bbox_height: int,
                                 target_width: int = 1920, target_height: int = 1080) -> Tuple[int, int, int, int]:
        """Calculate crop coordinates based on Mickey's position.
        
        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            mickey_x: Mickey center x coordinate
            mickey_y: Mickey center y coordinate
            bbox_width: Bounding box width
            bbox_height: Bounding box height
            target_width: Target output width
            target_height: Target output height
            
        Returns:
            tuple: (crop_x1, crop_y1, crop_x2, crop_y2)
        """
        # Calculate expanded bbox dimensions (8% expansion)
        expansion_x = bbox_width * 0.08
        expansion_y = bbox_height * 0.08
        expanded_width = bbox_width + 2 * expansion_x
        expanded_height = bbox_height + 2 * expansion_y

        # Calculate target crop dimensions with 16:9 aspect ratio
        aspect_ratio = target_width / target_height
        if expanded_width / expanded_height > aspect_ratio:
            crop_width = expanded_width
            crop_height = crop_width / aspect_ratio
        else:
            crop_height = expanded_height
            crop_width = crop_height * aspect_ratio

        # Initialize crop coordinates
        crop_x1 = mickey_x - crop_width / 2
        crop_y1 = mickey_y - crop_height / 2
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        # Check available space
        x_has_space = crop_width <= frame_width
        y_has_space = crop_height <= frame_height

        # Case 1: Both dimensions have enough space
        if x_has_space and y_has_space:
            pass  # Keep current coordinates

        # Case 2: X dimension needs adjustment
        elif not x_has_space and y_has_space:
            crop_x1 = 0
            crop_x2 = frame_width
            # Keep y-axis centered
            crop_y1 = mickey_y - crop_height / 2
            crop_y2 = mickey_y + crop_height / 2

        # Case 3: Y dimension needs adjustment
        elif x_has_space and not y_has_space:
            crop_y1 = 0
            crop_y2 = frame_height
            # Keep x-axis centered
            crop_x1 = mickey_x - crop_width / 2
            crop_x2 = mickey_x + crop_width / 2

        # Case 4: Both dimensions need adjustment
        else:
            # Use quarter regions
            quarter_width = frame_width / 2
            quarter_height = frame_height / 2

            if mickey_x < frame_width / 2:
                crop_x1 = 0
                crop_x2 = quarter_width * 2
            else:
                crop_x1 = frame_width - quarter_width * 2
                crop_x2 = frame_width

            if mickey_y < frame_height / 2:
                crop_y1 = 0
                crop_y2 = quarter_height * 2
            else:
                crop_y1 = frame_height - quarter_height * 2
                crop_y2 = frame_height

        # Ensure coordinates are within frame bounds
        crop_x1 = max(0, min(int(crop_x1), frame_width - 1))
        crop_y1 = max(0, min(int(crop_y1), frame_height - 1))
        crop_x2 = max(crop_x1 + 1, min(int(crop_x2), frame_width))
        crop_y2 = max(crop_y1 + 1, min(int(crop_y2), frame_height))

        return crop_x1, crop_y1, crop_x2, crop_y2

    def draw_detection_opencv(self, frame: np.ndarray, detections: dict, frame_width: int, frame_height: int) -> None:
        """Draw detections using OpenCV.
        
        Args:
            frame: Input frame
            detections: Detection results
            frame_width: Original frame width
            frame_height: Original frame height
        """
        for i in range(detections['num_detections']):
            cls = detections['detection_classes'][i]
            score = detections['detection_scores'][i]
            box = detections['detection_boxes'][i]
            
            if cls == 0 and score >= 0.65:  # Mickey class
                # Convert normalized coordinates to absolute
                x1, y1, x2, y2 = box
                bbox_x1 = int(x1 * frame_width)
                bbox_y1 = int(y1 * frame_height)
                bbox_x2 = int(x2 * frame_width)
                bbox_y2 = int(y2 * frame_height)
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 0, 0), 2)
                
                # Prepare label
                label = f"Mickey {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.3
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (bbox_x1, bbox_y1 - text_height - 10),
                    (bbox_x1 + text_width + 10, bbox_y1),
                    (255, 0, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (bbox_x1 + 5, bbox_y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )

    def extract_detections(self, input_data: dict, conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
        """Parse YOLOv5 model output and process detections."""
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
            #print(f"\nFinal detections:")
            #print(f"Number of detections: {len(final_boxes)}")
            #for i, (box, cls, score) in enumerate(zip(final_boxes, final_classes, final_scores)):
            #    print(f"Detection {i}:")
            #    print(f"- Class: {self.labels[int(cls)]}")
            #    print(f"- Score: {score:.3f}")
            #    print(f"- Box: {box}")
            
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
        """Scale boxes from normalized coordinates back to original image size
        
        Args:
            boxes: array of shape (N, 4) containing normalized coordinates [x1, y1, x2, y2]
            
        Returns:
            Scaled boxes in original image coordinates
        """
        try:
            # Print input boxes
            #print("\n=== Scaling Boxes ===")
            #print(f"Input boxes (normalized):")
            #print(boxes)
            #print(f"Pad info: {self.pad_info}")
            
            # Make a copy to avoid modifying the original
            boxes = boxes.copy()
            
            # First convert normalized coordinates (0-1) to model input size coordinates
            boxes[..., [0, 2]] *= self.model_input_size[1]  # scale x coordinates
            boxes[..., [1, 3]] *= self.model_input_size[0]  # scale y coordinates
            
            #print(f"\nAfter scaling to model input size:")
            #print(boxes)
            
            # Remove padding
            boxes[..., [0, 2]] -= self.pad_info['pad_w']  # x coordinates
            boxes[..., [1, 3]] -= self.pad_info['pad_h']  # y coordinates
            
            #print(f"\nAfter removing padding:")
            #print(boxes)
            
            # Apply inverse scale to get back to original image size
            scale = 1.0 / self.pad_info['scale']
            boxes *= scale
            
            #print(f"\nAfter applying inverse scale ({scale}):")
            #print(boxes)
            
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
            
            #print(f"\nFinal boxes (after clipping):")
            #print(boxes)
            #print(f"Original image shape: {self.pad_info['orig_shape']}")
            #print(f"X range: [{boxes[..., [0, 2]].min()}, {boxes[..., [0, 2]].max()}]")
            #print(f"Y range: [{boxes[..., [1, 3]].min()}, {boxes[..., [1, 3]].max()}]")
            
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
        """Compute sigmoid activation with better numerical stability"""
        x = np.clip(x, -88, 88)
        # Add scaling factor for better confidence
        x = x * 2.0  # Increase slope for more extreme values
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
        
    def calculate_scene_bbox(self, scene: SceneSegment, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Calculate maximum bbox range for current scene.
        
        Args:
            scene: Scene segment containing bbox information
            frame_width: Original video width
            frame_height: Original video height
            
        Returns:
            tuple: (center_x, center_y, width, height) in absolute coordinates
        """
        # Convert relative coordinates to absolute and calculate corners
        abs_corners = []
        for x, y, w, h in scene.bboxes:
            abs_x = x * frame_width
            abs_y = y * frame_height
            abs_w = w * frame_width
            abs_h = h * frame_height

            x1 = abs_x - abs_w / 2
            y1 = abs_y - abs_h / 2
            x2 = abs_x + abs_w / 2
            y2 = abs_y + abs_h / 2
            abs_corners.append((x1, y1, x2, y2))

        # Calculate bbox union
        left = min(corner[0] for corner in abs_corners)
        top = min(corner[1] for corner in abs_corners)
        right = max(corner[2] for corner in abs_corners)
        bottom = max(corner[3] for corner in abs_corners)

        # Calculate width and height
        width = right - left
        height = bottom - top

        # Add 8% expansion
        expansion_x = width * 0.08
        expansion_y = height * 0.08

        left -= expansion_x
        right += expansion_x
        top -= expansion_y
        bottom += expansion_y

        # Calculate center point and dimensions
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        final_width = right - left
        final_height = bottom - top

        return int(center_x), int(center_y), int(final_width), int(final_height)