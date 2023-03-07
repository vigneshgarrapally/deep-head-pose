

# Deep Head Pose Fork #

This is a fork of the [deep-head-pose](https://github.com/prakhargoyal106/deep-head-pose) repository with some additional scripts added.

<div align="center">
  <img src="https://i.imgur.com/K7jhHOg.png" width="380"><br><br>
</div>

## Introduction ##

**Deep Head Pose** is a project that uses deep learning to estimate the pose of a human head in an image or video. It's based on a Convolutional Neural Network (CNN) architecture that takes an image as input and outputs the yaw, pitch, and roll angles of the head.

**Hopenet** is an accurate and easy to use head pose estimation network. Models have been trained on the 300W-LP dataset and have been tested on real data with good qualitative performance.

For details about the method and quantitative results please check the CVPR Workshop [paper](https://arxiv.org/abs/1710.00925).

<div align="center">
<img src="conan-cruise.gif" /><br><br>
</div>


## Requirements

To run the scripts in this repository, you'll need the following:

* Python 3.6 or higher
* PyTorch 1.0 or higher
* OpenCV-Python

You can install the Python dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

This repository contains several scripts to train and test. Detailed instructions for using the scripts in this repository can be found [here](code/README.md).

Please open an issue if you have an problem.

## Acknowledgements

The original Deep Head Pose repository was created by [Prakhar Goyal](https://github.com/prakhargoyal106). Thanks to the author for providing the pre-trained model weights and sample code for head pose estimation.

