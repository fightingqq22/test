import cv2
import os
import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import yolo
import yolox_stream_report_detections as report

# yolox_s_leaky input resolution
INPUT_RES_H = 640
INPUT_RES_W = 640

# Loading compiled HEFs to device:
model_name = 'mickey'
hef_path = '{}.hef'.format(model_name)
video_dir = '.'
video_path = os.path.join(video_dir, 'mickey.mp4')  # Use mickey.mp4 as the input video
hef = HEF(hef_path)
devices = Device.scan()

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Video file does not exist: {video_path}")
    exit()

with VDevice(device_ids=devices) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    height, width, channels = hef.get_input_vstream_infos()[0].shape

    cap = cv2.VideoCapture(video_path)  # Open the video file
    display_width = 1280  # Adjust based on your screen
    display_height = 720

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        exit()

    while True:
        # Read a frame from the video source
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print("Video has ended or could not read frame")
            break

        # Get height and width from capture
        orig_h, orig_w = frame.shape[:2]
        display_frame = cv2.resize(frame, (display_width, display_height))
    
        # 计算等比例缩放的大小
        orig_h, orig_w = frame.shape[:2]
        ratio = min(INPUT_RES_H / orig_h, INPUT_RES_W / orig_w)
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)

        # 执行等比例缩放
        resized_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 创建白色画布
        canvas = np.ones((INPUT_RES_H, INPUT_RES_W, 3), dtype=np.float32) * 255

        # 计算边界填充
        pad_h = (INPUT_RES_H - new_h) // 2
        pad_w = (INPUT_RES_W - new_w) // 2

        # 将缩放后的图像放在画布中央
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_img
    
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_data = {input_vstream_info.name: np.expand_dims(canvas, axis=0).astype(np.float32)}
            with network_group.activate(network_group_params):
                infer_results = infer_pipeline.infer(input_data)

        # Print the keys and shapes of infer_results for debugging
        print("Infer results keys and shapes:")
        for key, value in infer_results.items():
            print(f"Key: {key}, Shape: {value.shape}")

        # Create dictionary that returns layer name from tensor shape (required for postprocessing)
        layer_from_shape = {infer_results[key].shape: key for key in infer_results.keys()}

        # Postprocessing info for constructor as recommended in hailo_model_zoo/cfg/base/yolox.yaml
        anchors = {"strides": [32, 16, 8], "sizes": [[1, 1], [1, 1], [1, 1]]}
        yolox_post_proc = yolo.YoloPostProc(
            img_dims=(INPUT_RES_H, INPUT_RES_W),
            nms_iou_thresh=0.65,
            score_threshold=0.3,
            anchors=anchors,
            output_scheme=None,
            classes=2,
            labels_offset=1,
            meta_arch="yolo_v5",
            device_pre_post_layers=[]
        )

        # Order of insertion matters since we need the reorganized tensor to be in (BS,H,W,85) shape
        endnodes = [
            infer_results[layer_from_shape[(1, 40, 40, 21)]],
            infer_results[layer_from_shape[(1, 20, 20, 21)]],
            infer_results[layer_from_shape[(1, 80, 80, 21)]],
        ]
        hailo_preds = yolox_post_proc.yolo_postprocessing(endnodes)
        num_detections = int(hailo_preds['num_detections'])
        scores = hailo_preds["scores"][0].numpy()
        classes = hailo_preds["classes"][0].numpy()
        boxes = hailo_preds["boxes"][0].numpy()
        if scores[0] == 0:
            num_detections = 0
        preds_dict = {'scores': scores, 'classes': classes, 'boxes': boxes, 'num_detections': num_detections}
        display_frame = report.report_detections(preds_dict, display_frame, scale_factor_x=display_width/orig_w, scale_factor_y=display_height/orig_h)
        cv2.imshow('frame', display_frame)

        # Wait for a key event
        key = cv2.waitKey(1)

        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            break

# Release the video source and destroy all windows
cap.release()
cv2.destroyAllWindows()
