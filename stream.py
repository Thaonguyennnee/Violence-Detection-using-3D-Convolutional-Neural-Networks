from network import C3D_modelv1
import numpy as np
import cv2
import pandas as pd
import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import time

torch.backends.cudnn.benchmark = True
import os
import shutil


def detect_fast_moving_objects(prvs, next, threshold=2):
    data = []

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 15, 1.2, 0)

    # Calculate the magnitude of the optical flow
    magnitude = np.linalg.norm(flow, axis=-1)

    # Threshold the magnitude to detect fast-moving objects
    fast_moving_objects = (magnitude > threshold).astype(np.uint8) * 255

    # Find contours of the fast-moving objects
    contours, _ = cv2.findContours(fast_moving_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the fast-moving objects
    # frame_copy = frame2.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        data.append([x, y, x + w, y + h])

    #     # Display the frame with bounding boxes around fast-moving objects
    #     cv2.imshow('Fast-Moving Objects with Bounding Boxes', frame_copy)
    #
    #     k = cv2.waitKey(30) & 0xFF
    #     if k == 27:  # Press 'ESC' key to exit
    #         break
    #
    #     prvs = next
    #
    # cap.release()
    # cv2.destroyAllWindows()

    df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2']).astype(int)
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


def find_overlapping_regions(bbox_data_fast, bbox_data_yolov7, threshold=0.5):
    overlapping_regions = []
    bbox = None
    df_bbox = None
    for j in range(len(bbox_data_yolov7)):
        yolov7_bbox = bbox_data_yolov7.iloc[j]
        for i in range(len(bbox_data_fast)):
            fast_bbox = bbox_data_fast.iloc[i]

            iou = calculate_iou(
                [fast_bbox['x1'], fast_bbox['y1'], fast_bbox['x2'], fast_bbox['y2']],
                [yolov7_bbox['X1'], yolov7_bbox['Y1'], yolov7_bbox['X2'], yolov7_bbox['Y2']]
            )

            if iou >= threshold:
                overlapping_regions.append(yolov7_bbox.to_dict())

    overlapping_regions = pd.DataFrame(overlapping_regions)
    print(overlapping_regions)

    if not overlapping_regions.empty:
        x1_min = overlapping_regions['X1'].min()
        y1_min = overlapping_regions['Y1'].min()
        x2_max = overlapping_regions['X2'].max()
        y2_max = overlapping_regions['Y2'].max()

        bbox = (x1_min, y1_min, x2_max, y2_max)
        df_bbox = pd.DataFrame(bbox)

    print(bbox)

    return bbox, df_bbox


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main(weights, source, imgsz=640, conf_thres=0.3, iou_thres=0.2, device='',
         classes=0, agnostic_nms=False, augment=False, no_trace=False):
    global y1_min, y2_max, x2_max, x1_min
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/2class.txt', 'r') as f:
        class_names = f.readlines()

    model_C3D = C3D_modelv1.C3D(num_classes=2)
    checkpoint = torch.load('run_new/run_10/models/C3D-traindata_epoch-19.pth.tar',
                            map_location=lambda storage, loc: storage)
    model_C3D.load_state_dict(checkpoint['state_dict'])
    model_C3D.to(device)
    model_C3D.eval()

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # # Initialize
    # set_logging()
    # device = select_device(device)
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # Load FP32 model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(imgsz, s=stride)  # Check img_size

    if no_trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # To FP16

    if webcam:
        view_img = check_imshow()
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    frame_index = 0
    start_time = time.time()
    clip = []
    l_bbox = []
    clipp = []

    for path, img, im0s, vid_cap in dataset:
        frame_index += 1
        # if frame_index % 2 == 0:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to FP16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # Detections per image
            print(frame_index)
            if source.isnumeric():  # Batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                clip.append(im0)
                # if len(clip) > 1:
                #     df1 = None
                #     prvs = cv2.cvtColor(clip[i - 1], cv2.COLOR_BGR2GRAY)
                #     next = cv2.cvtColor(clip[i], cv2.COLOR_BGR2GRAY)
                #     df1 = detect_fast_moving_objects(prvs, next, threshold=50)
                #
                #     for index, row in df1.iterrows():
                #         x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                #         cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Draw a green bounding box
                #     # Write results
                bbox_data = []
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox_data.append([*xyxy])
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green bounding box
                df = pd.DataFrame(bbox_data, columns=['X1', 'Y1', 'X2', 'Y2']).astype(int)
                # print(df)

                    # bbox, df_bbox = find_overlapping_regions(df1, df, threshold=0.4) #bbox of each frame
                    # # print(bbox)
                    # if bbox is not None:
                    #     x1, y1, x2, y2 = bbox
                    #     cv2.rectangle(im0, (x1, y1), (x2, y2), (225, 0, 0), 2)  # Draw a green bounding box
                    #
                    #     # print(bbox)
                    #     l_bbox.append([x1, y1, x2, y2])
                    #     print(l_bbox)
                    #     list_bbox = pd.DataFrame(l_bbox, columns=['x1', 'y1', 'x2', 'y2']).astype(int)
                    #     print(list_bbox)
                    #     if not list_bbox.empty:
                    #         x1_min = list_bbox['x1'].min()
                    #         y1_min = list_bbox['y1'].min()
                    #         x2_max = list_bbox['x2'].max()
                    #         y2_max = list_bbox['y2'].max()
                    #
                    #         bbox = (x1_min, y1_min, x2_max, y2_max)
                    #
                    #     cropped_frame = im0[y1_min:y2_max, x1_min:x2_max]

                    # clipp = []
                    #
                    # while True:
                    #     ret, frame = cap.read()
                    #     if not ret:
                    #         break

                # tmp_ = cv2.resize(im0, (112, 112))
                # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                # clipp.append(tmp)
                #
                # if len(clipp) == 16:
                #     inputs = np.array(clipp).astype(np.float32)
                #     inputs = np.expand_dims(inputs, axis=0)
                #     inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                #     inputs = torch.from_numpy(inputs)
                #     inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
                #
                #     with torch.no_grad():
                #         outputs = model_C3D.forward(inputs)
                #
                #     probs = torch.nn.Softmax(dim=1)(outputs)
                #     label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                #     # if probs[0][1].detach().cpu().item() > 0.3:
                #     cv2.putText(im0, class_names[label].split(' ')[-1].strip(), (20, 20),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #                 (0, 0, 255), 1)
                #     cv2.putText(im0, "prob: %.4f" % probs[0][label], (20, 40),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #                 (0, 0, 255), 1)
                #
                #     clipp.pop(0)
                #     # l_bbox.pop(0)
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = frame_index / elapsed_time

                # Display the FPS on the frame
                cv2.putText(im0, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # yield frame
                cv2.imshow('result', im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    return


if __name__ == '__main__':
    weights = 'yolov7.pt'
    for name in os.listdir('UCF_fighting_processed'):
        source = r'0'
        t1 = time.time()
        bbox_data_yolov7 = main(weights=weights, source=source)
        print(bbox_data_yolov7)
