import numpy as np
import cv2
import tensorflow as tf

def get_label(class_id):
    labels = {0: "Mickey", 1: "Minnie"}
    return labels.get(class_id, "unlabeled")

def draw_detection(image, d, c, conf, color, scale_factor_x, scale_factor_y):
    """Draw box and label for 1 detection."""
    label = get_label(c)
    
    # 正确解析坐标顺序 [x1,y1,x2,y2]
    xmin, ymin, xmax, ymax = d
    
    # 转换到像素坐标
    height, width = image.shape[:2]
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    
    # 计算中心点
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)
    
    # 绘制边界框
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=2)
    
    # 绘制中心点
    cv2.circle(image, (center_x, center_y), 5, (0, 255, 255), -1)
    
    # 准备标签文本
    label_text = f"{label} {conf:.2f}"
    
    # 获取文本大小以绘制背景
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # 绘制标签背景
    label_ymin = max(ymin - text_height - 10, 0)
    label_ymax = max(ymin, text_height + 10)
    cv2.rectangle(image, 
                 (xmin, label_ymin),
                 (xmin + text_width + 10, label_ymax),
                 color, 
                 cv2.FILLED)
    
    # 绘制文本
    cv2.putText(image, 
                label_text,
                (xmin + 5, label_ymax - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness)
    
    # 添加坐标调试信息
    debug_text = f"({center_x},{center_y})"
    cv2.putText(image,
                debug_text,
                (center_x - 50, center_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)
    
    return label, image

def report_detections(detections, image, min_score=0.45, scale_factor_x=1, scale_factor_y=1):
    """Reports, saves and draws all confident detections"""
    COLORS = {
        0: (0, 255, 0),  # Mickey - 绿色
        1: (0, 0, 255)   # Minnie - 红色
    }
    
    LABELS = {
        0: "Mickey",
        1: "Minnie"
    }
    
    def safe_get(value):
        if isinstance(value, (np.ndarray, tf.Tensor)):
            return value.numpy() if hasattr(value, 'numpy') else value
        return value
    
    # 打印完整的检测结果
    print("\nRaw detection data:")
    for key in detections.keys():
        print(f"Key: {key}")
        if isinstance(detections[key], np.ndarray):
            print(f"Shape: {detections[key].shape}")
            print(f"Values: {detections[key]}")
    
    boxes = safe_get(detections['boxes'])
    scores = safe_get(detections['scores'])
    classes = safe_get(detections['classes'])
    num_detections = safe_get(detections['num_detections'])
    
    if len(boxes.shape) > 2:
        boxes = boxes[0]
        scores = scores[0]
        classes = classes[0]
    
    num_detections = int(num_detections)
    img_height, img_width = image.shape[:2]
    draw = image.copy()
    
    print(f"\nImage size: {img_width}x{img_height}")
    print(f"Number of detections: {num_detections}")
    print(f"Boxes shape: {boxes.shape}")
    
    # 绘制测试框
    cv2.rectangle(draw, (100, 100), (200, 200), (255, 0, 0), 2)
    
    for i in range(num_detections):
        if scores[i] >= min_score:
            try:
                bbox = boxes[i]
                print(f"\nProcessing detection {i}:")
                print(f"Raw bbox: {bbox}")
                
                # 从bbox中提取坐标
                if len(bbox) == 4:
                    # YOLO格式: [x_center, y_center, width, height]
                    x_center, y_center, width, height = bbox
                    
                    # 转换为像素坐标
                    x_center_px = int(x_center * img_width)
                    y_center_px = int(y_center * img_height)
                    width_px = int(width * img_width)
                    height_px = int(height * img_height)
                    
                    # 计算边界框坐标
                    x1 = int(x_center_px - width_px/2)
                    y1 = int(y_center_px - height_px/2)
                    x2 = int(x_center_px + width_px/2)
                    y2 = int(y_center_px + height_px/2)
                    
                    print(f"Center: ({x_center_px}, {y_center_px})")
                    print(f"Size: {width_px}x{height_px}")
                    print(f"Box corners: ({x1},{y1}) -> ({x2},{y2})")
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, img_width - 1))
                    x2 = max(0, min(x2, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    y2 = max(0, min(y2, img_height - 1))
                    
                    # 获取类别和颜色
                    class_id = int(classes[i])
                    color = COLORS.get(class_id, (255, 0, 0))
                    label = LABELS.get(class_id, f"Class {class_id}")
                    
                    # 绘制边界框，使用更粗的线条
                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3)
                    
                    # 绘制中心点
                    cv2.circle(draw, (x_center_px, y_center_px), 5, (0, 255, 255), -1)
                    
                    # 绘制标签
                    label_text = f"{label} {scores[i]:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    # 获取文本大小
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, font, font_scale, thickness)
                    
                    # 绘制标签背景
                    cv2.rectangle(draw, 
                                (x1, y1 - text_height - 5), 
                                (x1 + text_width + 5, y1), 
                                color, 
                                cv2.FILLED)
                    
                    # 绘制标签文本
                    cv2.putText(draw,
                               label_text,
                               (x1, y1 - 5),
                               font,
                               font_scale,
                               (255, 255, 255),
                               thickness)
                    
                    # 绘制调试信息
                    debug_text = f"({x1},{y1})->({x2},{y2})"
                    cv2.putText(draw,
                               debug_text,
                               (x_center_px - 50, y_center_px + 30),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.5,
                               (255, 255, 255),
                               1)
            
            except Exception as e:
                print(f"Error processing detection {i}: {str(e)}")
                continue
    
    return draw
