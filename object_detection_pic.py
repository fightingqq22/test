#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading
from PIL import Image
from typing import List
from object_detection_utils_pic import ObjectDetectionUtils

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_input_images, validate_images, divide_list_to_batches


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="mickey.hef"  # Changed to yolov5.hef
    )
    parser.add_argument(
        "-i", "--input", 
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size", 
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-l", "--labels", 
        default="mickey_labels.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )
    # Added YOLOv5 specific arguments
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="Confidence threshold for detection"
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold for detection"
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.labels):
        logger.info(f"Labels file not found: {args.labels}, creating default Mickey/Minnie labels")
        with open(args.labels, 'w') as f:
            f.write("Mickey\nMinnie")

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args


def enqueue_images(
    images: List[Image.Image],
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
    utils: ObjectDetectionUtils
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.
    """
    for batch in divide_list_to_batches(images, batch_size):
        processed_batch = []
        batch_array = []

        for image in batch:
            processed_image = utils.preprocess(image)
            processed_batch.append(processed_image)
            batch_array.append(np.array(processed_image))

        input_queue.put(processed_batch)

    input_queue.put(None)  # Add sentinel value to signal end of input


def process_output(
    output_queue: queue.Queue,
    output_path: Path,
    width: int,
    height: int,
    utils: ObjectDetectionUtils,
    conf_thres: float,
    iou_thres: float
) -> None:
    """
    Process and visualize the output results.
    """
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        processed_image, infer_results = result
        # Modified to pass confidence and IoU thresholds
        detections = utils.extract_detections(infer_results, conf_thres, iou_thres)
        utils.visualize(
            detections, processed_image, image_id,
            output_path, width, height
        )
        image_id += 1

    output_queue.task_done()


def infer(
    images: List[Image.Image],
    net_path: str,
    labels_path: str,
    batch_size: int,
    output_path: Path,
    conf_thres: float,
    iou_thres: float
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    utils = ObjectDetectionUtils(labels_path)

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size
    )
    height, width, _ = hailo_inference.get_input_shape()
    print(f"Model input shape: {width}x{height}")

    enqueue_thread = threading.Thread(
        target=enqueue_images, 
        args=(images, batch_size, input_queue, width, height, utils)
    )
    process_thread = threading.Thread(
        target=process_output, 
        args=(output_queue, output_path, width, height, utils, conf_thres, iou_thres)
    )

    enqueue_thread.start()
    process_thread.start()

    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    process_thread.join()

    logger.info(
        f'Inference was successful! Results have been saved in {output_path}'
    )


def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()

    # Load input images
    images = load_input_images(args.input)

    # Validate images
    try:
        validate_images(images, args.batch_size)
    except ValueError as e:
        logger.error(e)
        return

    # Create output directory if it doesn't exist
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)

    # Start the inference with YOLOv5 specific parameters
    infer(
        images, 
        args.net, 
        args.labels, 
        args.batch_size, 
        output_path,
        args.conf_thres,
        args.iou_thres
    )


if __name__ == "__main__":
    main()