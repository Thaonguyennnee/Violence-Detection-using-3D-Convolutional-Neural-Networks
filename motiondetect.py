import time
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import torch
from numpy import random
import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def perform_object_detection(weights , source = 0, imgsz=640, conf_thres=0.25, iou_thres=0.2, device='', view_img=True,
                             save_txt=True, save_conf=False, nosave=False, classes=0, agnostic_nms=False, augment=False,
                             update=False, project='runs/detectnew', name='exp', exist_ok=False, no_trace=False):
    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # Increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # Load FP32 model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(imgsz, s=stride)  # Check img_size

    if no_trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # To FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # Initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')):
        view_img = check_imshow()
        cudnn.benchmark = True  # Set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    frame_index=0
    bbox_data = []
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        frame_index += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to FP16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # Detections per image
            if source.isnumeric():  # Batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # To Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain (whwh)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # Detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        bbox_data.append([frame_index, *xyxy])

    df = pd.DataFrame(bbox_data, columns=['frame', 'X1', 'Y1', 'X2', 'Y2']).astype(int)


    if nosave:
        return df

    # Stream results
    if view_img:
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond

    # Save results (image with detections)
    if dataset.mode == 'image':
        cv2.imwrite(save_path, im0)
    else:  # 'video' or 'stream'
        if vid_path != save_path:  # New video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # Release previous video writer
            if vid_cap:  # Video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # Stream
                fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path += '.mp4'
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        vid_writer.write(im0)

    return df

# Function to calculate optical flow and detect fast-moving objects
def detect_fast_moving_objects(video_path, threshold=2):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    frame_index = 0
    data = []

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame_index += 1

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate the magnitude of the optical flow
        magnitude = np.linalg.norm(flow, axis=-1)

        # Threshold the magnitude to detect fast-moving objects
        fast_moving_objects = (magnitude > threshold).astype(np.uint8) * 255

        # Find contours of the fast-moving objects
        contours, _ = cv2.findContours(fast_moving_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the fast-moving objects
        frame_copy = frame2.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            data.append([frame_index, x, y, x + w, y + h])

        # Display the frame with bounding boxes around fast-moving objects
        cv2.imshow('Fast-Moving Objects with Bounding Boxes', frame_copy)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # Press 'ESC' key to exit
            break

        prvs = next

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(data, columns=['FrameIndex', 'x1', 'y1', 'x2', 'y2'])
    return df

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Union area
    union = area_box1 + area_box2 - intersection

    # Calculate IoU
    iou = intersection / union

    return iou

def find_overlapping_regions(bbox_data_fast, bbox_data_yolov7, threshold = 0.5):
    overlapping_regions = []

    for i in range(len(bbox_data_fast)):
        fast_bbox = bbox_data_fast.iloc[i]
        for j in range(len(bbox_data_yolov7)):
            yolov7_bbox = bbox_data_yolov7.iloc[j]

            iou = calculate_iou(
                [fast_bbox['x1'], fast_bbox['y1'], fast_bbox['x2'], fast_bbox['y2']],
                [yolov7_bbox['X1'], yolov7_bbox['Y1'], yolov7_bbox['X2'], yolov7_bbox['Y2']]
            )

            if iou >= threshold:
                overlapping_regions.append(fast_bbox.to_dict())

    return pd.DataFrame(overlapping_regions)

def get_bbox():
    bbox = None
    cropped_frame = None

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        frame_copy = frame2.copy()  # Create a copy of the frame for drawing

        # Calculate the minimum bounding rectangle (MBR) for all overlapping regions
        if not overlapping_regions.empty:
            x1_min = overlapping_regions['x1'].min()
            y1_min = overlapping_regions['y1'].min()
            x2_max = overlapping_regions['x2'].max()
            y2_max = overlapping_regions['y2'].max()

            bbox = (x1_min, y1_min, x2_max, y2_max)

            # Create a bounding box for the MBR
            cv2.rectangle(frame_copy, (x1_min, y1_min), (x2_max, y2_max), (0, 0, 255), 2)



        if bbox is not None:
            cropped_frame = frame_copy[y1_min:y2_max, x1_min:x2_max]
            cv2.imshow('Overlapping Regions', cropped_frame)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # Press 'ESC' key to exit
            break

    # Close windows
    cap.release()
    cv2.destroyAllWindows()
    return bbox

if __name__ == '__main__':
    # frame_index = 0
    weights = 'yolov7.pt'
    for name in os.listdir('UCF_fighting_processed'):
        source = 'UCF_fighting_processed/' + name
        bbox_data_yolov7 = perform_object_detection(weights = weights, source = source)
        print(bbox_data_yolov7)

        # threshold = 0.5  # Adjust this threshold as needed
        bbox_data_fast = detect_fast_moving_objects(source, 2)
        print(bbox_data_fast)

        overlapping_regions = find_overlapping_regions(bbox_data_fast, bbox_data_yolov7)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Error: Could not open video.")
        ret, frame1 = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('', fourcc, 30, (frame1.shape[1], frame1.shape[0]))

        result = get_bbox()
        print(result)