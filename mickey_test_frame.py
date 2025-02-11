import cv2
import os
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SceneSegment:
    start_frame: int
    end_frame: int
    bboxes: List[Tuple[float, float, float, float]]  # List of (x, y, w, h)


class SceneDetector:
    def __init__(self, max_gap_frames=3, min_confidence=0.85):
        """初始化 场景检测器

        Args:
            max_gap_frames (int): 分割场景前没有米奇的最大帧数
            min_confidence (float): 有效米奇检测的最小置信度阈值
        """
        self.max_gap_frames = max_gap_frames
        self.min_confidence = min_confidence
        self.current_scene = None
        self.scenes = []
        self.gap_count = 0

    def process_frame(self, frame_number: int, bbox_info: Optional[Tuple]) -> None:
        """处理单帧并更新场景信息

        Args:
            frame_number (int): 当前frame的序号
            bbox_info (tuple): (x_center, y_center, width, height, confidence) or None (bbox文件中的内容)
        """
        is_valid_detection = (
                bbox_info is not None and
                bbox_info[4] >= self.min_confidence
        )

        if is_valid_detection:
            x, y, w, h, _ = bbox_info
            if self.current_scene is None:
                # 新场景分段
                self.current_scene = SceneSegment(frame_number, frame_number, [(x, y, w, h)])
                self.gap_count = 0
            else:
                # 继续旧场景分段追加
                self.current_scene.end_frame = frame_number
                self.current_scene.bboxes.append((x, y, w, h))
                self.gap_count = 0
        else:
            if self.current_scene is not None:
                self.gap_count += 1
                if self.gap_count > self.max_gap_frames:
                    # 结束当前场景
                    self.scenes.append(self.current_scene)
                    self.current_scene = None
                    self.gap_count = 0

    def finalize(self) -> None:
        """完成场景检测并添加最后一个场景（如果存在）"""
        if self.current_scene is not None:
            self.scenes.append(self.current_scene)
            self.current_scene = None


def calculate_scene_bbox(scene: SceneSegment, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """计算当前场景里面bbox的最大范围

    Args:
        scene (SceneSegment): 场景分段（包含bbox信息）
        frame_width (int): 原始视频宽度
        frame_height (int): 原始视频高度

    Returns:
        tuple: (center_x, center_y, width, height) 绝对坐标
    """
    # 转换相对坐标 到 绝对坐标，并计算上下左右的坐标点
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

    # 计算bbox的union（最大值）
    left = min(corner[0] for corner in abs_corners)
    top = min(corner[1] for corner in abs_corners)
    right = max(corner[2] for corner in abs_corners)
    bottom = max(corner[3] for corner in abs_corners)

    # 计算宽度、高度
    width = right - left
    height = bottom - top

    # 增加8%向外扩展
    expansion_x = width * 0.08
    expansion_y = height * 0.08

    left -= expansion_x
    right += expansion_x
    top -= expansion_y
    bottom += expansion_y

    # 计算中心点和尺寸（长宽）
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    final_width = right - left
    final_height = bottom - top

    return int(center_x), int(center_y), int(final_width), int(final_height)


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


def calculate_crop_coordinates(frame_width, frame_height, mickey_x, mickey_y, bbox_width, bbox_height,
                               target_width=1920, target_height=1080, smoothing_buffer=None):
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
        smoothing_buffer (SmoothingBuffer): 平滑缓冲区对象
    Returns:
        tuple: (crop_x1, crop_y1, crop_x2, crop_y2) 裁剪坐标
    """
    # 使用平滑缓冲区更新和获取平滑后的位置
    if smoothing_buffer is not None:
        smoothing_buffer.update(mickey_x, mickey_y)
        smooth_x, smooth_y = smoothing_buffer.get_smooth_position()
        if smooth_x is not None and smooth_y is not None:
            mickey_x, mickey_y = smooth_x, smooth_y

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


import time

def process_video(video_path: str, labels_dir: str, output_path: str) -> None:
    """核心函数

    Args:
        video_path (str): 输入的视频路径
        labels_dir (str): 推理导出的label txt 路径
        output_path (str): 输出的视频路径
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

    scene_detector = SceneDetector()
    frame_number = 0
    processing_times = []

    # 第一遍：场景检测
    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break

        bbox_file = os.path.join(labels_dir, f"mickey_cut_{frame_number}.txt")
        bbox_info = None

        if os.path.exists(bbox_file):
            # 1. 读取每一帧的bbox信息
            bbox_info = read_bbox_file(bbox_file)
        # 2. 更新场景信息
        scene_detector.process_frame(frame_number, bbox_info)
        frame_number += 1

    scene_detector.finalize()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 第二遍：处理每个场景
    current_scene_idx = 0
    current_scene = scene_detector.scenes[current_scene_idx] if scene_detector.scenes else None
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        if current_scene and current_scene.start_frame <= frame_number <= current_scene.end_frame:
            # 读取当前帧的bbox信息用于显示
            bbox_file = os.path.join(labels_dir, f"mickey_cut_{frame_number}.txt")
            if os.path.exists(bbox_file):
                bbox_info = read_bbox_file(bbox_file)
                if bbox_info and bbox_info[4] >= scene_detector.min_confidence:
                    # 获取当前帧的bbox坐标和置信度
                    x_center, y_center, width, height, confidence = bbox_info

                    # 转换为绝对坐标
                    abs_x = int(x_center * frame_width)
                    abs_y = int(y_center * frame_height)
                    abs_width = int(width * frame_width)
                    abs_height = int(height * frame_height)

                    # 计算bbox的左上角和右下角坐标
                    bbox_x1 = int(abs_x - abs_width / 2)
                    bbox_y1 = int(abs_y - abs_height / 2)
                    bbox_x2 = int(abs_x + abs_width / 2)
                    bbox_y2 = int(abs_y + abs_height / 2)

                    # 1.绘制边界框和置信度
                    draw_bbox_with_confidence(frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence)

            # 2. 计算场景的最大bbox
            scene_bbox = calculate_scene_bbox(current_scene, frame_width, frame_height)
            mickey_x, mickey_y, mickey_width, mickey_height = scene_bbox

            # 3. 计算裁剪坐标
            crop_x1, crop_y1, crop_x2, crop_y2 = calculate_crop_coordinates(
                frame_width, frame_height, mickey_x, mickey_y,
                mickey_width, mickey_height
            )

            # 4. 裁剪并调整大小
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_frame = cv2.resize(cropped_frame, (1920, 1080))
            out.write(cropped_frame)

            end_time = time.time()
            elapsed_time = end_time - start_time
            processing_times.append(elapsed_time)
            print(f"Frame {frame_number}: Mickey detected, cropped, and resized in {elapsed_time:.4f} seconds")

            if frame_number == current_scene.end_frame and current_scene_idx < len(scene_detector.scenes) - 1:
                current_scene_idx += 1
                current_scene = scene_detector.scenes[current_scene_idx]
        else:
            resized_frame = cv2.resize(frame, (1920, 1080))
            out.write(resized_frame)

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 计算并打印平均处理时间
    if processing_times:
        average_time = sum(processing_times) / len(processing_times)
        print(f"\nAverage processing time per frame: {average_time:.4f} seconds")

    print("\nDetected scenes:")
    for i, scene in enumerate(scene_detector.scenes):
        print(f"Scene {i + 1}: frames {scene.start_frame}-{scene.end_frame} "
              f"(duration: {scene.end_frame - scene.start_frame + 1} frames)")


if __name__ == "__main__":
    video_path = "mickey_cut.mp4"
    labels_dir = "labels"
    output_path = "mickey_demo_frame.mp4"
    process_video(video_path, labels_dir, output_path)
