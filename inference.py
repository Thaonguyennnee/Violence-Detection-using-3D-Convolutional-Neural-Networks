import torch
import numpy as np
from network import C3D_modelv1
import cv2
import time
torch.backends.cudnn.benchmark = True
import os
import shutil

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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/2class.txt', 'r') as f:
        class_names = f.readlines()
    #    f.close()
    # init model
    model = C3D_modelv1.C3D(num_classes=2)
    checkpoint = torch.load('run_new/run_10/models/C3D-traindata_epoch-19.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    # path = 'UCF_fighting/Fighting002_x264.mp4'
    # filename = os.listdir(path)
    # for name in filename:
    #     video = path + '/' + name
    #     cap = cv2.VideoCapture(video)
    #     retaining = True
    video = 'static/UCF_fighting/Fighting002_x264.mp4'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    y = 0
    n = 600



    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            if probs[0][1].detach().cpu().item() > 0.5:
                colour = (0, 0, 255)
                y += 1
                n = 0

            else:
                n += 1
                colour = (0, 255, 0)

            # cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            #             colour, 1)
            # cv2.putText(frame, "%.4f" % probs[0][label], (80, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            #             colour, 1)
            if n < 250:
                cv2.putText(frame, "Warning", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1)
            if y >30 and n < 100:
                cv2.putText(frame, "Warning: Violence", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0,0,225), 1)
            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()









