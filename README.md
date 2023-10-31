# Fight Detection using 3D CNNs
## Introduction
This project applies a 3D Convolutional Neural network to detect fighting in surveillance cameras implemented using PyTorch. The data set is compiled from many sources on the internet.
Re-implement the C3D model for the custom dataset 2 layers described in the repo: [pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Webapp](#Webapp)
- [Future work](#Futurework)

## Installation
Anaconda and Python 3.8. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/Thaonguyennnee/Violence-Detection-using-3D-Convolutional-Neural-Networks.git
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

2. Install Project Dependencies:
    ```Shell
    pip install -r requirements.txt
    ```

## Usage

1. Configure your dataset and pre-trained model path in
[mypath.py](mypath.py).

2. Testing:

    To test the model, please do:
    ```Shell
    python inference.py
    ```
    
3. Training:
   To test the model, please do:
    ```Shell
    python train.py
    ```

## Webapp
A basic web app with 2 functions detects from video and camera. If a fight is detected, there is a mail alert sent to the user.
```Shell
python app.py
```
<div align="center">
    <a href="./">
        <img src="figure/Screenshot 2023-10-31 104012.png" width="50%"/>
    </a>
    <a href="./">
        <img src="figure/mail_alert.png" width="39%"/>
    </a>
</div>

## Futurework
The new pipeline preprocessing to pay attention to people who moving based on paper: [LOCALIZATION GUIDED FIGHT ACTION DETECTION IN SURVEILLANCE VIDEOS](https://weiyaolin.github.io/pdf/LocalizationGuided.pdf)
```Shell
python stream.py
```
![preprocessing](figure/new_preprocess.png)
