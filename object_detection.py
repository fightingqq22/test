#!/usr/bin/env python3

import cv2
import argparse
import sys
import os
import numpy as np
from loguru import logger
import queue
import traceback
import threading
from pathlib import Path
from object_detection_utils import ObjectDetectionUtils
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

    def process_frame(self, frame):
        """Process a single frame through the detection pipeline."""
        try:
            # Get original frame dimensions
            orig_height, orig_width = frame.shape[:2]
            print(f"\nProcessing frame: {orig_width}x{orig_height}")
            
            # Preprocess frame
            processed_frame = self.utils.preprocess(frame)
            if processed_frame is None:
                return frame
                
            # Put frame in input queue
            self.input_queue.put([processed_frame])
            
            # Get detection results
            try:
                result = self.output_queue.get(timeout=5)
                if result is None:
                    return frame
            except queue.Empty:
                print("Timeout waiting for inference results")
                return frame
                
            processed_frame, infer_results = result
            
            # Extract detections
            detections = self.utils.extract_detections(
                infer_results, self.conf_thres, self.iou_thres
            )
            
            print(f"\nDetections found: {detections['num_detections']}")
            if detections['num_detections'] > 0:
                print("Detection boxes:")
                for i in range(detections['num_detections']):
                    print(f"Box {i}: {detections['detection_boxes'][i]}")
            
            # Create PIL Image for visualization
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Visualize detections on frame
            annotated_frame = self.utils.visualize_video(
                detections, frame_rgb, orig_width, orig_height
            )
            
            # Convert back to BGR for OpenCV
            return cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame

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

    # Initialize detector
    detector = VideoDetector(
        args.net,
        args.labels,
        args.conf_thres,
        args.iou_thres
    )

    logger.info("Starting video detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            # Display result
            cv2.imshow('Detection Results', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Detection stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()