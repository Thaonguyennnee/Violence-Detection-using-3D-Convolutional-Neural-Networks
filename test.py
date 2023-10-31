import torch
import numpy as np
from network import C3D_model
import cv2
import time
torch.backends.cudnn.benchmark = True
import os
import shutil
from sklearn.metrics import f1_score, precision_score, recall_score

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
    model = C3D_model.C3D(num_classes=2)
    checkpoint = torch.load('/home/seino_tuanpn7/ThaoNguyen/video-recognition/C3D-traindata_epoch-19.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()


    pathvideos = '/home/seino_tuanpn7/violence/data/violence-data'
    pathtest = '/home/seino_tuanpn7/ThaoNguyen/video-recognition/traindata/test'
    pathval = '/home/seino_tuanpn7/ThaoNguyen/video-recognition/traindata/val'
    video_extensions = ['.mp4', '.avi', '.mpg', '.mov']
    y_true = []
    y_pred = []
    incorrect = []
    foldername = os.listdir(pathtest)
    for n in foldername:
        filename = os.listdir(pathtest + '/' + n)
        print(n)
        for name in filename:
            for ext in video_extensions:
                video = pathvideos + '/' + n + '/' + name + ext
                cap = cv2.VideoCapture(video)
                if cap.isOpened():
                    video_paths = video
                    
    # print(y_true)
    # print(len(y_true))
            cap = cv2.VideoCapture(video_paths)
            retaining = True

            clip = []
            label = 0

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
                    if probs[0][1].detach().cpu().item() > 0.8:
                        print(label)
                        y_pred.append(label)
                        retaining = False
                        break
                    clip.pop(0)

                cv2.imshow('result', frame)
                cv2.waitKey(1)
            if label != 1:
                y_pred.append(label)
                print(label)
            if n == 'normal':
                true_label = 0
                y_true.append(true_label)
            else:
                true_label = 1
                y_true.append(true_label)
            if true_label != label:
                incorrect.append(video_paths)
    #                 print(video)


            cap.release()
            cv2.destroyAllWindows()

    y_true.remove(0)
    y_true.remove(0)
    print(len(y_true)-len(y_pred))
    print(f1_score(y_true,y_pred))
    print(recall_score(y_true,y_pred))
    print(precision_score(y_true,y_pred))
    print(incorrect)



if __name__ == '__main__':
    main()









