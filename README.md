# Ultralytics Yolo11 Custom Object Detection

Repo contains an example of how to train a custom Yolo11 object detection model

## Assumptions

* The source_data directory contains the images and labels file and so custom labeling is considered done.

* Training will be on a local Mac with Mac Silicon 
* 

## Resources

https://docs.ultralytics.com/modes/train/

### YoloV11: How to train for object detection on a custom dataset

Roboflow Example

https://www.youtube.com/watch?v=etjkjZoG2F0

### YouTube Tutorials

https://youtu.be/A1V8yYlGEkI?si=nxBmp_zCrAOtw8FT



## Setup

* mkdir <project dir>
* cd <project dir>
* uv init .
* uv add ultralytics

## Creating the train/test/val split

The data was originally in the source_data directory, without a train/test/val split..

The script, split_data.py, was used to create the train/test/val split which was placed in the modeling_data directory.

## Dataset

The dataset is stored in the modeling_data directory and is split into train, val, and test sets.

