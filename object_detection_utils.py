from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from tensorflow.python.ops.init_ops_v2 import normal


def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (255, 255, 255)):
        """
        Initialize the ObjectDetectionUtils class.
        """
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
    
    def get_labels(self, labels_path: str) -> list:
        """
        Load labels from a file.
        """
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names

    def preprocess(self, frame: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        """
        Resize frame with unchanged aspect ratio using padding.
        """
        # Get frame dimensions
        img_h, img_w = frame.shape[:2]
        print(f"img_h: {img_h}, img_w: {img_w}")
        print(f"model_h: {model_h}, model_w: {model_w}")

        # Calculate scaling factor
        scale = min(model_w / img_w , model_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_img_w, new_img_h))
        
        # Create padding
        padded_frame = np.full((model_h, model_w, 3), self.padding_color, dtype=np.uint8)
        
        # Calculate padding coordinates
        x_offset = (model_w - new_img_w) // 2
        y_offset = (model_h - new_img_h) // 2
        
        # Place the resized frame in the center of the padded frame
        padded_frame[y_offset:y_offset+new_img_h, x_offset:x_offset+new_img_w] = resized_frame
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        #normalized_img = rgb_frame.astype(np.float32) / 255.0
        
        return rgb_frame

    def draw_detection(self, frame: np.ndarray, box: list, cls: int, score: float, color: tuple):
        """
        Draw box and label for one detection on the frame.
        """
        label = f"{self.labels[cls]}: {score:.2f}%"
        ymin, xmin, ymax, xmax = [int(coord) for coord in box]
        
        # Draw rectangle
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Add label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (xmin, ymin - 25), 
                     (xmin + label_size[0], ymin),
                     (0, 255, 0),
                     -1)  # Filled rectangle
        
        # Add label text (in white)
        cv2.putText(frame, 
                    label, 
                    (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0),
                    2)

    def visualize_frame(self, detections: dict, frame: np.ndarray, width: int, height: int, min_score: float = 0.45) -> np.ndarray:
        if detections['num_detections'] == 0:
            return frame
            
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']
        
        result_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                # Generate unique color for this class
                color = generate_color(classes[idx])
                
                # Get coordinates and scale to frame size
                ymin = int(boxes[idx][0] * frame_height)
                xmin = int(boxes[idx][1] * frame_width)
                ymax = int(boxes[idx][2] * frame_height)
                xmax = int(boxes[idx][3] * frame_width)
                
                # Draw bounding box
                cv2.rectangle(result_frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Prepare label text
                label = f"{self.labels[classes[idx]]}: {scores[idx]*100:.1f}%"
                
                # Get label size for background
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(result_frame, 
                            (xmin, ymin - label_h - baseline - 10),
                            (xmin + label_w, ymin),
                            color, 
                            -1)
                
                # Draw label text
                cv2.putText(result_frame,
                           label,
                           (xmin, ymin - baseline - 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (255, 255, 255),
                           2)
        
        return result_frame

    def extract_detections(self, input_data: dict, threshold: float = 0.5) -> dict:
        boxes, scores, classes = [], [], []
        num_detections = 0

        try:
            # Debug print
            #print("Input data structure:", {k: v.shape if hasattr(v, 'shape') else len(v) for k, v in input_data.items()})
            print("=============================")
            print(f"input_data.items(): {input_data.items()}")

            for layer_name, detections in input_data.items():
                print(f"layer_name: {layer_name}")
                # 解析检测结果
                for detection in detections.reshape(-1, detections.shape[-1]):  # 展平检测结果
                    print(f"detection:{detection}")
                    score = detection[4]  # 置信度
                    if score >= threshold:  # 过滤低于阈值的检测结果
                        boxes.append(detection[:4])  # 边界框坐标
                        scores.append(score)  # 置信度
                        classes.append(0)  # 假设只有一个类别
                        num_detections += 1

            # Debug print
            print(f"Extracted {num_detections} detections above threshold {threshold}")

        except Exception as e:
            print(f"Error in extract_detections: {str(e)}")
            return {'detection_boxes': [], 'detection_classes': [], 'detection_scores': [], 'num_detections': 0}

        return {
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
            'num_detections': num_detections
        }
