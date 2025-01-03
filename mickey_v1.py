import cv2
import os
import numpy as np
import time
from collections import deque


def read_bbox_file(filepath):
    """读取标注文件，返回Mickey的坐标信息（如果存在）
    Args:
        filepath (str): 标注文件路径
    Returns:
        tuple: (x_center, y_center, width, height, confidence) 或 None
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # 解析每一行数据，包含置信度
                label, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                # 只返回Mickey (label = 0) 的信息
                if int(label) == 0:
                    return x_center, y_center, width, height, confidence
    except Exception as e:
        print(f"Error reading bbox file: {e}")
        return None
    return None


def calculate_crop_coordinates(frame_width, frame_height, mickey_x, mickey_y, bbox_width, bbox_height,
                               target_width=1920, target_height=1080):
    """计算裁剪坐标，基于四种不同的情况
    Args:
        frame_width (int): 原始帧宽度
        frame_height (int): 原始帧高度
        mickey_x (int): Mickey中心点x坐标
        mickey_y (int): Mickey中心点y坐标
        bbox_width (int): 边界框宽度
        bbox_height (int): 边界框高度
        target_width (int): 目标宽度
        target_height (int): 目标高度
    Returns:
        tuple: (crop_x1, crop_y1, crop_x2, crop_y2) 裁剪坐标
    """
    # 计算扩展后的bbox尺寸（扩展8%）
    expansion_x = bbox_width * 0.08
    expansion_y = bbox_height * 0.08
    expanded_width = bbox_width + 2 * expansion_x
    expanded_height = bbox_height + 2 * expansion_y

    # 计算目标裁剪区域的尺寸，保持16:9比例
    aspect_ratio = target_width / target_height
    if expanded_width / expanded_height > aspect_ratio:
        crop_width = expanded_width
        crop_height = crop_width / aspect_ratio
    else:
        crop_height = expanded_height
        crop_width = crop_height * aspect_ratio

    # 初始化裁剪区域坐标
    crop_x1 = mickey_x - crop_width / 2
    crop_y1 = mickey_y - crop_height / 2
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    # 检查是否有足够的裁剪空间
    x_has_space = crop_width <= frame_width
    y_has_space = crop_height <= frame_height

    # 情况1：xy都够裁剪
    if x_has_space and y_has_space:
        # 保持当前计算的坐标
        pass

    # 情况2：x不够裁剪
    elif not x_has_space and y_has_space:
        crop_x1 = 0
        crop_x2 = frame_width
        # 保持y轴中心点
        crop_y1 = mickey_y - crop_height / 2
        crop_y2 = mickey_y + crop_height / 2

    # 情况3：y不够裁剪
    elif x_has_space and not y_has_space:
        crop_y1 = 0
        crop_y2 = frame_height
        # 保持x轴中心点
        crop_x1 = mickey_x - crop_width / 2
        crop_x2 = mickey_x + crop_width / 2

    # 情况4：xy都不够裁剪
    else:
        # 计算米奇所在的四分之一区域
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

    # 确保坐标在合理范围内
    crop_x1 = max(0, min(int(crop_x1), frame_width - 1))
    crop_y1 = max(0, min(int(crop_y1), frame_height - 1))
    crop_x2 = max(crop_x1 + 1, min(int(crop_x2), frame_width))
    crop_y2 = max(crop_y1 + 1, min(int(crop_y2), frame_height))

    return crop_x1, crop_y1, crop_x2, crop_y2


def draw_bbox_with_confidence(frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence):
    """在帧上绘制边界框和置信度标签
    Args:
        frame (np.ndarray): 视频帧
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 (int): 边界框坐标
        confidence (float): 置信度
    """
    # 绘制蓝色边界框
    cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 0, 0), 2)

    # 准备置信度标签文本
    label = f"Mickey {confidence:.2f}"

    # 获取文本大小以确定背景矩形的尺寸
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # 绘制深蓝色背景矩形
    cv2.rectangle(frame,
                  (bbox_x1, bbox_y1 - text_height - 10),
                  (bbox_x1 + text_width + 10, bbox_y1),
                  (255, 0, 0), -1)  # -1 表示填充矩形

    # 在背景上绘制白色文本
    cv2.putText(frame, label,
                (bbox_x1 + 5, bbox_y1 - 5),
                font, font_scale, (255, 255, 255), thickness)


def process_video(video_path, labels_dir):
    """处理视频主函数
    Args:
        video_path (str): 输入视频路径
        labels_dir (str): 标注文件目录路径
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建窗口显示视频
    cv2.namedWindow("Mickey Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mickey Detection", 1920, 1080)

    frame_number = 0
    max_time = 0
    processing_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # 构建对应的标注文件路径
        bbox_file = os.path.join(labels_dir, f"mickey_cut_{frame_number}.txt")
        if os.path.exists(bbox_file):
            # 读取Mickey的bbox信息
            bbox_info = read_bbox_file(bbox_file)
            if bbox_info:
                x_center, y_center, width, height, confidence = bbox_info

                # 只处理置信度大于0.8的检测结果
                if confidence > 0.8:
                    # 将相对坐标转换为绝对坐标
                    mickey_x = int(x_center * frame_width)
                    mickey_y = int(y_center * frame_height)
                    mickey_width = int(width * frame_width)
                    mickey_height = int(height * frame_height)

                    # 计算bbox的左上角和右下角坐标
                    bbox_x1 = int(mickey_x - mickey_width / 2)
                    bbox_y1 = int(mickey_y - mickey_height / 2)
                    bbox_x2 = int(mickey_x + mickey_width / 2)
                    bbox_y2 = int(mickey_y + mickey_height / 2)

                    # 绘制带有置信度的边界框
                    draw_bbox_with_confidence(frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence)

                    # 计算裁剪坐标
                    crop_x1, crop_y1, crop_x2, crop_y2 = calculate_crop_coordinates(
                        frame_width, frame_height, mickey_x, mickey_y,
                        mickey_width, mickey_height
                    )

                    # 裁剪并调整大小
                    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    resized_frame = cv2.resize(cropped_frame, (1920, 1080))
                else:
                    # 置信度低于阈值，直接缩放整帧
                    resized_frame = cv2.resize(frame, (1920, 1080))
            else:
                # 未检测到Mickey，直接缩放整帧
                resized_frame = cv2.resize(frame, (1920, 1080))
        else:
            # 标注文件不存在，直接缩放整帧
            resized_frame = cv2.resize(frame, (1920, 1080))

        end_time = time.time()
        elapsed_time = end_time - start_time
        processing_times.append(elapsed_time)
        max_time = max(max_time, elapsed_time)

        # 显示处理后的视频帧
        cv2.imshow("Mickey Detection", resized_frame)

        # 打印时间
        print(f"Frame {frame_number}: Processed in {elapsed_time:.4f} seconds")

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")

    # 打印最大时间
    print(f"Maximum processing time: {max_time:.4f} seconds")
    # 打印平均处理时间
    if processing_times:
        average_time = sum(processing_times) / len(processing_times)
        print(f"Average processing time per frame: {average_time:.4f} seconds")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "mickey_cut.mp4"  # 输入4K视频路径
    labels_dir = "./labels"  # 标注文件目录

    process_video(video_path, labels_dir)
