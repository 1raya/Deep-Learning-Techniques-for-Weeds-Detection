# Deep-Learning-Techniques-for-Weeds-Detection


## Table of Content
* [Introduction](#Introduction)
* [Environment Requirements](#Environment-Requirements)
* [Dataset](#Dataset)
* [Model Architecture](#Model-Architecture)
* [Model Training](#Model-Training)
* [Conclusion](#Conclusion)

## Introduction  
Weeds pose a significant threat to agricultural productivity, causing yield losses and increasing the need for herbicides. Traditional weed detection methods are time-consuming and labor-intensive, making them inefficient for large-scale farming operations. In recent years, deep learning techniques have shown promising results in various computer vision tasks, including object detection. This project aim to use deep learning models to detect weeds, compare their performance, and provide additional analysis using a newly created dataset

## Environment Requirements:

* gitpython>=3.1.30
* opencv-python>=4.1.1
* torch>=1.8.0
* torchvision>=0.9.0
* Pillow
* matplotlib
* pandas
* seaborn
* Google Colab Pro
* Roboflow


# Dataset
In this project, we used an open-access [Weeds and Corn dataset from Roboflow](https://universe.roboflow.com/secomindai/weeddetection-kvotz) . This dataset includes manually annotated images that can be employed to detect weeds. The dataset consist of 1,268 RGB images divided into two classes: weeds and corn images. The dataset has been annotated with bounding box annotations. Through augmentation, the dataset has been expanded to contain 3042 instances, with each training example having three outputs. Specifically, for the "crop" class, maximum zoom of 20%, minimum zoom of 0%, and rotations ranging from -15° to +15° have been applied.

## New Dataset
We expanded our analysis by evaluating the performance of YOLOv7 on a newly collected dataset, specifically captured from Okra and Eggplant fields. The dataset consists of 3 classes (Okra, Eggplant, Weeds) and 950 images captured by an iPhone 11 Pro and iPhone 12 Ultra Wide camera from May 16th to 18th, 2023. The data was collected between 11 AM and 7 PM, including various weather and light conditions. 

### Prepare and Label image data:
We used the advanced Smart Polygon feature in Roboflow Annotate, which is powered by the Segment Anything Model (SAM). This cloud-based solution allowed us to apply polygon annotations with improved speed, ease, and precision directly within the Roboflow UI. To prepare each image for analysis, pre-processing steps were performed, including autoorientation of pixel data by stripping EXIF-orientation and resizing to 640x640 dimensions using stretch transformation. A set of augmentation techniques was applied to generate two versions of each source image. These techniques included a 50% probability of horizontal flip, a 50% probability of vertical flip, random rotation within the range of -15 to +15 degrees, and random shear horizontally and vertically within the range of -15°to +15°. As a result of these augmentations, the total number of images in the dataset increased to 1596.

# Model Architecture
YOLO model considered as a One-stage object detectors, In this project we trained different versions of the real-time object detection YOLO models (YOLOv5, YOLOv6, YOLOv7, and YOLOv8). These YOLO models apply a single neural network to the full image as shown in below Figure.


# Model Training

# Evaluation

# Conclusion
