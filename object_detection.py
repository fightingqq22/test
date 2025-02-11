#!/usr/bin/env python3

import cv2
import argparse
import sys
import os
import numpy as np
from loguru import logger
import queue
import time
import traceback
import threading
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from object_detection_utils import ObjectDetectionUtils, SceneDetector, SceneSegment
from utils import HailoAsyncInference


def parse_args() -> argparse.Namespace:
    """Initialize argument parser for video detection."""
    parser = argparse.ArgumentParser(description="Video Detection Example")
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="mickey.hef"
    )
    parser.add_argument(
        "-i", "--input",
        help="Path to input video file or camera index (0 for webcam)",
        default="0"
    )
    parser.add_argument(
        "-l", "--labels",
        default="mickey_labels.txt",
        help="Path to labels file"
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold"
    )
    return parser.parse_args()


class VideoDetector:
    def __init__(self, net_path: str, labels_path: str, conf_thres: float, iou_thres: float):
        """Initialize video detector with model and parameters."""
        self.utils = ObjectDetectionUtils(labels_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize scene detector
        self.scene_detector = SceneDetector()
        
        # Initialize queues for async inference
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Initialize Hailo inference
        self.hailo_inference = HailoAsyncInference(
            net_path, self.input_queue, self.output_queue, batch_size=1
        )
        self.height, self.width, _ = self.hailo_inference.get_input_shape()
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.hailo_inference.run)
        self.inference_thread.daemon = True
        self.inference_thread.start()

    def detect_scenes(self, cap: cv2.VideoCapture) -> None:
        """First pass: Scene detection with detailed logging"""
        logger.info("Starting first pass: Scene detection...")
        
        frame_number = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 预处理
                processed_frame = self.utils.preprocess(frame)
                if processed_frame is None:
                    continue
                    
                # 推理
                self.input_queue.put([processed_frame])
                try:
                    result = self.output_queue.get(timeout=5)
                    if result is None:
                        continue
                except queue.Empty:
                    logger.warning(f"Frame {frame_number}/{total_frames}: Timeout waiting for inference")
                    continue
                    
                processed_frame, infer_results = result
                
                # 提取检测结果
                detections = self.utils.extract_detections(
                    infer_results, self.conf_thres, self.iou_thres
                )
                
                # 记录检测结果
                if detections['num_detections'] > 0:
                    detections_count += 1
                    logger.debug(f"Frame {frame_number}: Found {detections['num_detections']} detections")
                    for i in range(detections['num_detections']):
                        cls = detections['detection_classes'][i]
                        score = detections['detection_scores'][i]
                        if cls == 0:  # Mickey
                            logger.debug(f"  Mickey detected with confidence {score:.3f}")
                
                # 更新场景检测器
                self.scene_detector.process_frame(frame_number, detections)
                
                frame_number += 1
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    logger.info(f"Processing: {frame_number}/{total_frames} frames ({progress:.1f}%)")
            
            # 确保调用finalize
            logger.info("\nFirst pass completed, finalizing scene detection...")
            self.scene_detector.finalize()
            
            # 打印统计信息
            logger.info(f"\nProcessing summary:")
            logger.info(f"Total frames processed: {frame_number}")
            logger.info(f"Frames with detections: {detections_count}")
            logger.info(f"Number of scenes detected: {len(self.scene_detector.scenes)}")
            
            if not self.scene_detector.scenes:
                logger.warning("Warning: No scenes were detected!")
                logger.warning("Consider adjusting detection parameters:")
                logger.warning(f"Current confidence threshold: {self.conf_thres}")
                logger.warning(f"Current IoU threshold: {self.iou_thres}")
                logger.warning(f"Current scene gap threshold: {self.scene_detector.max_gap_frames}")
            
        except Exception as e:
            logger.error(f"Error during scene detection: {str(e)}")
            traceback.print_exc()
            
        finally:
            # 确保即使发生错误也会调用finalize
            if not self.scene_detector.scenes:
                self.scene_detector.finalize()


    def process_frame_in_scene(self, frame: np.ndarray, frame_number: int, scene: SceneSegment, detections: dict) -> np.ndarray:
        """Process frame within a detected scene."""
        try:
            orig_height, orig_width = frame.shape[:2]
            display_frame = frame.copy()

            # 1. 在原始frame上绘制检测信息
            for i in range(detections['num_detections']):
                cls = detections['detection_classes'][i]
                score = detections['detection_scores'][i]
                box = detections['detection_boxes'][i]
                
                if cls == 0 and score >= 0.5:  # Mickey class
                    # 转换为像素坐标
                    x1, y1, x2, y2 = map(int, [
                        box[0] * orig_width,
                        box[1] * orig_height,
                        box[2] * orig_width,
                        box[3] * orig_height
                    ])
                    
                    # 绘制检测框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # 准备置信度标签
                    label = f"Mickey {score:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8  # 调整字体大小
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # 绘制标签背景
                    cv2.rectangle(
                        display_frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        (255, 0, 0),
                        -1
                    )
                    
                    # 绘制标签文本
                    cv2.putText(
                        display_frame,
                        label,
                        (x1 + 5, y1 - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )

            # 2. 计算场景的最大bbox
            scene_bbox = self.utils.calculate_scene_bbox(scene, orig_width, orig_height)
            if scene_bbox is None:
                return cv2.resize(display_frame, (1920, 1080))
                
            mickey_x, mickey_y, mickey_width, mickey_height = scene_bbox
            
            # 3. 计算裁剪坐标
            crop_x1, crop_y1, crop_x2, crop_y2 = self.utils.calculate_crop_coordinates(
                orig_width, orig_height,
                mickey_x, mickey_y,
                mickey_width, mickey_height
            )
            
            # 4. 裁剪显示帧并调整大小
            cropped_frame = display_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            result_frame = cv2.resize(cropped_frame, (1920, 1080))
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {str(e)}")
            traceback.print_exc()
            return cv2.resize(frame, (1920, 1080))

    def process_video(self, cap: cv2.VideoCapture, out: cv2.VideoWriter) -> None:
        """Second pass: Process video with detection visualization."""
        logger.info("\nStarting second pass: Video processing...")
        
        frame_number = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_scene_idx = 0
        
        # 检查和打印场景信息
        if not self.scene_detector.scenes:
            logger.error("No scenes available for processing!")
            return
            
        logger.info(f"Found {len(self.scene_detector.scenes)} scenes:")
        for i, scene in enumerate(self.scene_detector.scenes):
            logger.info(f"Scene {i}: frames {scene.start_frame}-{scene.end_frame}")
        
        # 初始化当前场景
        current_scene = self.scene_detector.scenes[0]
        logger.info(f"\nStarting with scene 0: frames {current_scene.start_frame}-{current_scene.end_frame}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # 预处理
                processed_frame = self.utils.preprocess(frame)
                if processed_frame is None:
                    continue
                
                # 推理
                self.input_queue.put([processed_frame])
                try:
                    result = self.output_queue.get(timeout=5)
                    if result is None:
                        continue
                except queue.Empty:
                    logger.warning(f"Frame {frame_number}/{total_frames}: Inference timeout")
                    continue
                
                processed_frame, infer_results = result
                detections = self.utils.extract_detections(
                    infer_results, self.conf_thres, self.iou_thres
                )
                
                # 调试信息
                logger.debug(f"Frame {frame_number}: Processing with scene {current_scene_idx}")
                logger.debug(f"Scene range: {current_scene.start_frame}-{current_scene.end_frame}")
                
                # 场景处理判断
                if current_scene and current_scene.start_frame <= frame_number <= current_scene.end_frame:
                    logger.debug(f"Frame {frame_number}: In scene {current_scene_idx}")
                    result_frame = self.process_frame_in_scene(
                        frame, frame_number, current_scene, detections
                    )
                    
                    # 检查是否需要切换到下一个场景
                    if frame_number == current_scene.end_frame and current_scene_idx < len(self.scene_detector.scenes) - 1:
                        current_scene_idx += 1
                        current_scene = self.scene_detector.scenes[current_scene_idx]
                        logger.info(f"\nMoving to scene {current_scene_idx}: frames {current_scene.start_frame}-{current_scene.end_frame}")
                else:
                    # 检查是否需要更新当前场景
                    next_scene_idx = current_scene_idx + 1
                    if (next_scene_idx < len(self.scene_detector.scenes) and 
                        frame_number >= self.scene_detector.scenes[next_scene_idx].start_frame):
                        current_scene_idx = next_scene_idx
                        current_scene = self.scene_detector.scenes[current_scene_idx]
                        logger.info(f"\nUpdating to scene {current_scene_idx}: frames {current_scene.start_frame}-{current_scene.end_frame}")
                    
                    # 显示带检测框的原始帧
                    result_frame = frame.copy()
                    # 绘制检测框
                    if detections['num_detections'] > 0:
                        self.utils.draw_detection_opencv(result_frame, detections, frame.shape[1], frame.shape[0])
                    result_frame = cv2.resize(result_frame, (1920, 1080))
                
                # 显示和保存
                cv2.imshow('Detection Result', result_frame)
                out.write(result_frame)
                
                if frame_number % 30 == 0:
                    logger.info(f"Processing frame {frame_number}/{total_frames}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_number += 1
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_number}: {str(e)}")
                traceback.print_exc()
                
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    
    # Initialize video capture
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
        
    if not cap.isOpened():
        logger.error("Error opening video source")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Input video properties:")
    logger.info(f"Frame size: {frame_width}x{frame_height}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Total frames: {total_frames}")

    # Initialize detector
    detector = VideoDetector(
        args.net,
        args.labels,
        args.conf_thres,
        args.iou_thres
    )

    # Create output video filename
    if args.input.isdigit():
        output_path = f'output_camera_{args.input}.mp4'
    else:
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f'output_{input_name}.mp4'

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (1920, 1080)
    )

    try:
        # First pass: Scene detection
        detector.detect_scenes(cap)
        
        # Reset video capture for second pass
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Second pass: Process scenes
        detector.process_video(cap, out)
        
    except KeyboardInterrupt:
        logger.info("Processing stopped by user")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        traceback.print_exc()
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Video processing completed. Output saved to: {output_path}")

if __name__ == "__main__":
    main()