from PIL import Image
import os
import cv2
import numpy as np

# Set the path to the folder containing the images
input_folder = 'Mydata/test/normal'

import shutil
filename = os.listdir(input_folder)
for name in filename:
    n = len(os.listdir(path + '/' + name))
    if n<= 16:
        # shutil.rmtree(path+'/'+name)
        print(name)
