# Fight Detection using 3D CNNs
## Introduction
This project applies a 3D Convolutional Neural network to detect fighting in surveillance cameras implemented using PyTorch. The data set is compiled from many sources on the internet.
Re-implement the C3D model for the custom dataset 2 layers described in the repo: [pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

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


3. Configure your dataset and pretrained model path in
[mypath.py](https://github.com/jfzhang95/pytorch-video-recognition/blob/master/mypath.py).

4. You can choose different models and datasets in
[train.py](https://github.com/jfzhang95/pytorch-video-recognition/blob/master/train.py).

    To train the model, please do:
    ```Shell
    python train.py
    ```


## Usage

Provide instructions on how to use your project.

## Contributing

Tell others how they can contribute to your project.

## License

This project is licensed under the [License Name] - see the [LICENSE](LICENSE) file for details.
