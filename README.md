### 1. Overview

This project compares classical denoising methods (Gaussian and Median) with DnCNN under a unified and reproducible protocol for Gaussian noise removal at σ = 25. The pipeline includes consistent preprocessing, training or inference, and quantitative as well as qualitative evaluation.

### 2. Dataset

BSDS300 was used with 200 images for training and 100 images for testing. From the training set, 40 images were held out for validation. Images are either 321×481 or 481×321. All images were converted to grayscale and corrupted with synthetic Gaussian noise at σ = 25.

### 3. Methods

Classical baselines rely on OpenCV GaussianBlur and medianBlur with controlled kernel sizes and a fixed sigma for GaussianBlur.
DnCNN experiments include a pretrained 17-layer model and custom variants that change depth, number of feature maps, kernel size, and number of epochs while keeping training settings consistent. Portrait images were rotated to normalize shape and a single channel setting was used.

### 4. Results

DnCNN provided the best overall denoising quality. Classical Gaussian and Median filters were faster but tended to blur fine details. Increasing the depth of DnCNN improved performance only up to a point; beyond a certain number of layers the model overfit the small BSDS300 dataset and generalization degraded.

### 5. Responsibilities

I designed the entire pipeline from noise injection to training, inference, and evaluation. I implemented fair hyperparameter searches for both classical baselines and DnCNN, handled preprocessing including shape normalization and grayscale conversion, and conducted quantitative and qualitative analyses with failure case summaries.

### 6. Environment

Python 3.10 with Anaconda. Libraries include PyTorch, OpenCV, NumPy, Pandas, and Matplotlib.
GPU used: NVIDIA GeForce RTX 3050 Laptop GPU.

Model Reference: https://github.com/SaoYan/DnCNN-PyTorch.git

Dataset Reference: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
