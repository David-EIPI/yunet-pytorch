## yunet-pytorch

Reproduction of YuNet in PyTorch. The model is trained and quantized for use in embedded applications.

YuNet is an image detection network originally proposed for fast face detection. This repository reproduces the training and inference code using recent versions of PyTorch and supporting Python modules. Some simplifications were made because a general-purpose face detection system is not required. The intended use is to detect whether a human face is present in the camera view so that a face recognition model can be executed only when appropriate.

The model architecture accepts tuning parameters such as the number of hidden convolution layers and the number of feature pyramid layers. For the trained model included in this repository, these parameters were chosen to match the original YuNet architecture.

Input resolution is flexible, as is typical for PyTorch models. The provided weights were trained using RGB images of resolution 160×160.

The current configuration contains 73,381 trainable parameters.

The model architecture supports landmark prediction, but the training dataset provides only bounding box annotations. The trained weights therefore produce confidence scores and bounding boxes only.

Training was performed from scratch using the SoloFace dataset, which contains curated single-face images.

Objectness and confidence losses are combined because only a single object class is present. The total loss function is a linear combination of binary cross-entropy and Enhanced IoU (EIoU) losses.

## Training

Training was performed on a GPU with 4 GB of memory using the following SGD optimizer configuration:  
Initial learning rate: 0.01  
Momentum: 0.9  
Weight decay: 0.0005  

A cyclic learning rate scheduler was used, varying the learning rate between approximately 0.0002 and 0.02 with a period of about five epochs.

Batch size was set to 128.

The SoloFace dataset already includes augmentation such as mirroring and scaling. In addition, grayscale versions of all training images were added in order to reduce model dependence on color information.

Dataset sizes:

Training set: 79,804 images  
Validation set: 434 images

## Files

`yunet_model.py`  
Model definition and architecture implementation.

`yunet_train.py`  
Training script. Training data must be downloaded and placed in the `data` directory. The `setup.sh` script in this repository can be used to download and unpack the SoloFace dataset. Also, the directory must contain `dataset.py`, which loads the images and parses the annotations.

`predict.py`  
Basic inference script. The script runs the pretrained model on image files specified on the command line. Input images can be any format supported by OpenCV (e.g., JPG or PNG). The script performs anchor decoding and outputs the confidence value and bounding box coordinates. The detected bounding box is also drawn on the original image.

Since the model is intended for images containing a single face, non-maximum suppression is not performed.

`quantize_test.py`  
Script for quantizing the pretrained model for use with `esp-dl`. Quantization uses the `esp_ppq` package. Both int8 and int16 quantization modes are supported, although accuracy has been tested only for the int16 configuration.

For best results, the calibration dataset should contain images from the same source that will be used with the quantized model.

`weights/yunet_v3.pth`  
Weights obtained after training the model for 1000 epochs on the SoloFace dataset.

## Quantization and Embedded Use

Quantization is implemented using the `esp_ppq` package and produces models compatible with the `esp-dl` inference framework.

Quantization modes supported by the script:

int8  
int16

The quantization method used by `esp_ppq` is symmetric by default.

The quantization script accepts the target platform as a parameter. Supported targets include ESP32-S3 and ESP32-P4, as supported by `esp_ppq` and `esp-dl`.

The model has not yet been deployed on an ESP device, so inference latency and memory footprint are not currently available.

## Metrics

With the confidence threshold set to 0.995, the quantized model performs as follows.

SoloFace test set: 3732 images  
Accuracy: 88%  
Precision: 94%  
Recall: 80%
Bounding box prediction IoU loss: 0.32  
Average IoU with ground truth boxes: 0.73

Private dataset obtained from a security camera: 81 face images  
Accuracy: 84%  
Precision: 88%  
Recall: 79%
Bounding box prediction IoU loss: 0.34  
Average IoU with ground truth boxes: 0.72

## References

YuNet  
Originally developed by Wei Wu et al.  
Shared by OpenCV Zoo:  
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Original implementation:  
https://github.com/ShiqiYu/libfacedetection

Paper:  
https://link.springer.com/article/10.1007/s11633-023-1423-y

SoloFace  
SoloFace: A Single-Face Dataset for Resource-Constrained Face Detection and Tracking  
DOI: https://doi.org/10.5281/zenodo.14474899

EIoU  
A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization  
DOI: https://doi.org/10.1109/TIP.2021.3077144