import torch
import pandas as pd
import cv2
from pathlib import Path
from utils.general import check_img_size, non_max_suppression, scale_coords, select_device, TracedModel, set_logging
from utils.datasets import LoadImages, LoadStreams

def perform_object_detection(weights, source=0, imgsz=640, conf_thres=0.25, iou_thres=0.2, device='', classes=0,
                             agnostic_nms=False, augment=False, no_trace=False):
    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if no_trace:
        model = TracedModel(model, device, imgsz)

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    frame_index = 0
    bbox_data = []

    for path, img, im0s, _ in dataset:
        frame_index += 1
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):
            p = Path(path)
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    bbox_data.append([frame_index, *xyxy])

                    # Display bounding boxes on the image
                    im0s = plot_one_box(xyxy, im0s, label=f'{names[int(cls)]} {conf:.2f}', color=colors[int(cls)])

        # Display the image with bounding boxes
        cv2.imshow('Result', im0s)
        cv2.waitKey(1)

    cv2.destroyAllWindows()  # Close the display window

    df = pd.DataFrame(bbox_data, columns=['frame', 'X1', 'Y1', 'X2', 'Y2']).astype(int)

    return df
