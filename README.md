## Introduction

A Keras implementation of YOLOv3.

---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.
    

## IOU Tracker
Python implementation of the IOU Tracker described in the AVSS 2017 paper
[High-Speed Tracking-by-Detection Without Using Image Information](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf).

This project is released under the MIT License (details in LICENSE file).
If you think our work is useful in your research, please consider citing:

```
@INPROCEEDINGS{1517Bochinski2017,
	AUTHOR = {Erik Bochinski and Volker Eiselein and Thomas Sikora},
	TITLE = {High-Speed Tracking-by-Detection Without Using Image Information},
	BOOKTITLE = {International Workshop on Traffic and Street Surveillance for Safety and Security at IEEE AVSS 2017},
	YEAR = {2017},
	MONTH = aug,
	ADDRESS = {Lecce, Italy},
	URL = {http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf},
	}
```

## Usage
Basic demo script:
```
$ ./det_tracking.py -h
usage: det_tracking.py [-h] -d DETECTION_PATH -o OUTPUT_PATH [-sl SIGMA_L]
               [-sh SIGMA_H] [-si SIGMA_IOU] [-tm T_MIN] [-fo FRAMEORDER]

optional arguments:
  -h, --help            show this help message and exit
  -d DETECTION_PATH, --detection_path DETECTION_PATH
                        full path to CSV file containing the detections
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output path to store the tracking results
  -sl SIGMA_L, --sigma_l SIGMA_L
                        low detection threshold
  -sh SIGMA_H, --sigma_h SIGMA_H
                        high detection threshold
  -si SIGMA_IOU, --sigma_iou SIGMA_IOU
                        intersection-over-union threshold
  -tm T_MIN, --t_min T_MIN
                        minimum track length
  -fo FRAMEORDER, --frameorder FRAMEORDER
                        convert the output to output in frame order.
```
Input data format:
    '''frame,track_id,x,y,w,h,confidence,-1,-1,-1,class_id'''