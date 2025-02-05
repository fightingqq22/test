#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
import numpy as np
from loguru import logger
import queue
import threading
from typing import List
from object_detection_utils import ObjectDetectionUtils

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, validate_images, divide_list_to_batches


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Video Detection Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="mickey.hef"
    )
    parser.add_argument(
        "-i", "--input", 
        default="mickey_1080P.mp4",
        help="Path to the input video file or camera index (default: 0 for webcam)"
    )
    parser.add_argument(
        "-b", "--batch_size", 
        default=1,
        type=int,
        required=False,
        help="Batch size for inference"
    )
    parser.add_argument(
        "-l", "--labels", 
        default="coco.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    if args.input.isdigit():
        args.input = int(args.input)
    elif not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video file not found: {args.input}")

    return args


def enqueue_frames(
    cap: cv2.VideoCapture,
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
    utils: ObjectDetectionUtils
) -> None:
    """
    Preprocess and enqueue video frames.
    """
    frames_batch = []
    orig_frames_batch = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Store original frame
        orig_frames_batch.append(frame)
        # Process frame for model input
        processed_frame = utils.preprocess(frame, width, height)
        frames_batch.append(processed_frame)

        if len(frames_batch) == batch_size:
            # Put both original and processed frames in queue
            input_queue.put((orig_frames_batch, frames_batch))
            frames_batch = []
            orig_frames_batch = []

    # Handle remaining frames
    if frames_batch:
        input_queue.put((orig_frames_batch, frames_batch))

    input_queue.put(None)  # Signal end of input


def process_output(output_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    while True:
        result = output_queue.get()
        if result is None:
            break

        frame, infer_results = result
        

        # Convert RGB back to BGR for visualization
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #cv2.imshow('Object Detection', frame)

        # Deals with the expanded results from hailort versions < 4.19.0
        if len(infer_results) == 1:
            infer_results = infer_results[0]
            
        # Print the detection results for debugging
        print("Infer results:", infer_results)
        
        detections = utils.extract_detections(infer_results)
        
        # Print extracted detections for debugging
        print("Extracted detections:", detections)
        
        # Visualize the frame
        visualized_frame = utils.visualize_frame(detections, frame, width, height)
        
        # Display the frame
        cv2.imshow('Object Detection111', visualized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    output_queue.task_done()


def infer(
    cap: cv2.VideoCapture,
    net_path: str,
    labels_path: str,
    batch_size: int
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    utils = ObjectDetectionUtils(labels_path)

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, 
        input_queue, 
        output_queue, 
        batch_size,
        send_original_frame=True  # This is important
    )
    height, width, _ = hailo_inference.get_input_shape()

    enqueue_thread = threading.Thread(
        target=enqueue_frames, 
        args=(cap, batch_size, input_queue, width, height, utils)
    )
    process_thread = threading.Thread(
        target=process_output, 
        args=(output_queue, width, height, utils)
    )

    enqueue_thread.start()
    process_thread.start()

    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)
    process_thread.join()

    logger.info('Inference completed successfully!')


def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()

    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error("Error opening video stream or file")
        return

    try:
        # Start the inference
        infer(cap, args.net, args.labels, args.batch_size)
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
