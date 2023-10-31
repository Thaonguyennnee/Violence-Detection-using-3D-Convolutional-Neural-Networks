import torch
import numpy as np
from network import C3D_modelv1
import cv2
import time

torch.backends.cudnn.benchmark = True

def video_detection(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/2class.txt', 'r') as f:
        class_names = f.readlines()

    model = C3D_modelv1.C3D(num_classes=2)
    checkpoint = torch.load('run_new/run_10/models/C3D-traindata_epoch-19.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Initialize the camera feed (default camera, you can specify a camera index)
    cap = cv2.VideoCapture(path)  # Use camera index 0 for the default camera

    clip = []
    y = 0
    n = 600

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if path != 0:
            frame = cv2.resize(frame, (1920, 1080))

        tmp_ = cv2.resize(frame, (112, 112))
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

            if probs[0][1].detach().cpu().item() > 0.65:
                y += 1
                n = 0

            else:
                n += 1

            if probs[0][1].detach().cpu().item() > 0.5:
                colour = (0, 0, 255)
            else:
                colour = (0, 255, 0)

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        colour, 2)
            cv2.putText(frame, "%.4f" % probs[0][label], (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        colour, 2)
            if n < 250:
                cv2.putText(frame, "Warning", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 255), 2)
            if y > 35 and n < 60:
                cv2.putText(frame, "Warning: Violence", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 225), 2)
            clip.pop(0)
        yield frame
    #     cv2.imshow('result', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
