from PIL import Image, ImageDraw, ImageFont
import numpy as np


def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.

    Args:
        class_id (int): The class ID to generate a color for.

    Returns:
        tuple: A tuple representing an RGB color.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114), label_font: str = "LiberationSans-Regular.ttf"):
        """
        Initialize the ObjectDetectionUtils class.

        Args:
            labels_path (str): Path to the labels file.
            padding_color (tuple): RGB color for padding. Defaults to (114, 114, 114).
            label_font (str): Path to the font used for labeling. Defaults to "LiberationSans-Regular.ttf".
        """
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        self.label_font = label_font
    
    def get_labels(self, labels_path: str) -> list:
        """
        Load labels from a file.

        Args:
            labels_path (str): Path to the labels file.

        Returns:
            list: List of class names.
        """
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names

    def preprocess(self, image: Image.Image, model_w: int, model_h: int) -> Image.Image:
        """
        Resize image with unchanged aspect ratio using padding.

        Args:
            image (PIL.Image.Image): Input image.
            model_w (int): Model input width.
            model_h (int): Model input height.

        Returns:
            PIL.Image.Image: Preprocessed and padded image.
        """
        img_w, img_h = image.size
        scale = min(model_w / img_w, model_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        image = image.resize((new_img_w, new_img_h), Image.Resampling.BICUBIC)

        padded_image = Image.new('RGB', (model_w, model_h), self.padding_color)
        padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
        return padded_image

    def draw_detection(self, draw: ImageDraw.Draw, box: list, cls: int, score: float, color: tuple, scale_factor: float):
        """
        Draw box and label for one detection.

        Args:
            draw (ImageDraw.Draw): Draw object to draw on the image.
            box (list): Bounding box coordinates.
            cls (int): Class index.
            score (float): Detection score.
            color (tuple): Color for the bounding box.
            scale_factor (float): Scale factor for coordinates.
        """
        label = f"{self.labels[cls]}: {score:.2f}%"
        ymin, xmin, ymax, xmax = box
        font = ImageFont.truetype(self.label_font, size=15)
        draw.rectangle([(xmin * scale_factor, ymin * scale_factor), (xmax * scale_factor, ymax * scale_factor)], outline=color, width=2)
        draw.text((xmin * scale_factor + 4, ymin * scale_factor + 4), label, fill=color, font=font)

    def visualize(self, detections: dict, image: Image.Image, image_id: int, output_path: str, width: int, height: int, min_score: float = 0.45, scale_factor: float = 1):
        """
        Visualize detections on the image.

        Args:
            detections (dict): Detection results.
            image (PIL.Image.Image): Image to draw on.
            image_id (int): Image identifier.
            output_path (str): Path to save the output image.
            width (int): Image width.
            height (int): Image height.
            min_score (float): Minimum score threshold. Defaults to 0.45.
            scale_factor (float): Scale factor for coordinates. Defaults to 1.
        """
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']
        draw = ImageDraw.Draw(image)

        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                color = generate_color(classes[idx])
                scaled_box = [x * width if i % 2 == 0 else x * height for i, x in enumerate(boxes[idx])]
                self.draw_detection(draw, scaled_box, classes[idx], scores[idx] * 100.0, color, scale_factor)
                
        image.save(f'{output_path}/output_image{image_id}.jpg', 'JPEG')

    def extract_detections(self, input_data: dict, threshold: float = 0.3) -> dict:
        """解析YOLOv5模型的输出"""
        boxes = []
        scores = []
        classes = []
        num_detections = 0
        
        try:
            # YOLOv5输出格式为：[batch, num_anchors, xywh + confidence + num_classes]
            for layer_name, detections in input_data.items():
                # 重塑数组以处理所有检测
                batch_size, height, width, channels = detections.shape
                num_anchors = 3
                num_classes = 80  # COCO数据集的类别数

                # 重组检测结果
                detections = detections.reshape(batch_size, height * width * num_anchors, -1)
                
                for detection in detections[0]:  # 处理第一个batch
                    # YOLOv5输出格式：[x, y, w, h, confidence, class_scores...]
                    confidence = detection[4]  # objectness score
                    
                    if confidence >= threshold:
                        # 获取类别分数
                        class_scores = detection[5:]
                        class_id = np.argmax(class_scores)
                        class_score = class_scores[class_id]
                        
                        # 计算最终分数
                        score = float(confidence * class_score)
                        
                        if score >= threshold:
                            # 转换xywh到xyxy格式
                            x, y, w, h = detection[0:4]
                            x1 = x - w/2
                            y1 = y - h/2
                            x2 = x + w/2
                            y2 = y + h/2
                            
                            # 添加检测结果
                            boxes.append([y1, x1, y2, x2])  # 转换为[ymin, xmin, ymax, xmax]格式
                            scores.append(score)
                            classes.append(class_id)
                            num_detections += 1
                            
            # 转换为numpy数组
            if num_detections > 0:
                boxes = np.array(boxes, dtype=np.float32)
                scores = np.array(scores, dtype=np.float32)
                classes = np.array(classes, dtype=np.int32)
            else:
                boxes = np.array([], dtype=np.float32)
                scores = np.array([], dtype=np.float32)
                classes = np.array([], dtype=np.int32)
                
        except Exception as e:
            print(f"Error in extract_detections: {str(e)}")
            return {
                'detection_boxes': np.array([]),
                'detection_classes': np.array([]),
                'detection_scores': np.array([]),
                'num_detections': 0
            }

        return {
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
            'num_detections': num_detections
        }

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: First box in format [ymin, xmin, ymax, xmax]
            box2: Second box in format [ymin, xmin, ymax, xmax]
        
        Returns:
            float: IoU value
        """
        # Calculate intersection
        ymin = max(box1[0], box2[0])
        xmin = max(box1[1], box2[1])
        ymax = min(box1[2], box2[2])
        xmax = min(box1[3], box2[3])
        
        if xmin >= xmax or ymin >= ymax:
            return 0.0
        
        intersection = (xmax - xmin) * (ymax - ymin)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0