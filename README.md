# Contour-Detection
## Overview
This is my implementation for the contour-detection task given to me during batch13 vinAI interview. In short, this model aim to detect the bounding box, the center, the number of circle for each plot in the image. The main method used here is YOLO - You Only Look Once. I have coded it from scratch (TinyYOLO) and utilized ultralytics library so i can provide a comprehensive comparison between each model but it turn out that my scratch model has to faced some problems.

### Overview of the dataset
<p align="center">
  <img src="images/overview/contour_000085.png" width="200" />
  <img src="images/overview/contour_000088.png" width="200" />
  <img src="images/overview/contour_000093.png" width="200" />
  <img src="images/overview/contour_000099.png" width="200" />
</p>

## Challenge
- Overlap circle : Some generated images have their contour plot overlap on each other, making detection task more challenging.
- Metrics : Intuitively, the number of circle in each plot can be somehow perceived as "class" and we can do some no-brain move like fit the whole yolo model with "number of circle" as a categorical label but i have tried to implement yolo model with MSE, MAE, ... loss function but it messed up the yolo's paper loss function so much that at the end, i can hardly compare loss of my yolo scratch model with ultralytic model.
- Code : The code is a little bit messy as i have to debug a lot.

## Result
As i mentioned above, it's problematic to compare 2 models which have different training loss and training regime so i will use ultralytic's metric - which will of course lower my scratch model performance.

| Model Name          | Box Loss | mAP@50  | Center Loss (MSE) |
| ------------------- | -------: | ------: | -----------------: |
| Scratch (Tiny YOLO) |   0.3120 | 0.76412 |             2.861  |
| YOLOv8n             |   0.1140 | 0.99416 |             0.534  |
| YOLOv8s             |   **0.0914** | 0.99475 |             **0.419**  |
| YOLO11n             |   0.1014 | 0.99416 |             0.462  |
| YOLO11s             |   0.1017 | **0.99488** |             0.453  |

### Scratch model (TinyYOLO) demo - click for a better view
<p align="center">
  <img src="images/yolo_scratch/output_image.png" width="200" />
  <img src="images/yolo_scratch/output_image1.png" width="200" />
  <img src="images/yolo_scratch/output_image2.png" width="200" />
  <img src="images/yolo_scratch/output_image3.png" width="200" />
</p>

### YOLOV8n demo
<p align="center">
  <img src="images/yolov8n/contour_000236.jpg" width="200" />
  <img src="images/yolov8n/contour_000252.jpg" width="200" />
  <img src="images/yolov8n/contour_000254.jpg" width="200" />
  <img src="images/yolov8n/contour_000265.jpg" width="200" />
</p>

### YOLOV8s demo
<p align="center">
  <img src="images/yolov8s/contour_000014.jpg" width="200" />
  <img src="images/yolov8s/contour_000019.jpg" width="200" />
  <img src="images/yolov8s/contour_000023.jpg" width="200" />
  <img src="images/yolov8s/contour_000036.jpg" width="200" />
</p>

### YOLO11n demo
<p align="center">
  <img src="images/yolo11n/contour_015790.jpg" width="200" />
  <img src="images/yolo11n/contour_015801.jpg" width="200" />
  <img src="images/yolo11n/contour_015811.jpg" width="200" />
  <img src="images/yolo11n/contour_015820.jpg" width="200" />
</p>

### YOLO11s demo
<p align="center">
  <img src="images/yolo11s/contour_016444.jpg" width="200" />
  <img src="images/yolo11s/contour_016450.jpg" width="200" />
  <img src="images/yolo11s/contour_016457.jpg" width="200" />
  <img src="images/yolo11s/contour_016460.jpg" width="200" />
</p>

## Installation
Clone this repo and install with
```
pip install -r requirements.txt
```
Anyway i have modified my local ultralytics library so i do not think you can re-run this code properly, but you can re-run the scratch yolo model without any issues

## Training
To train best model run : 
```
python train.py
```
The model will have name “blue_cross.pth” and the log will be saved at “log.txt”
## Evaluation
to eval the best model run : 
```
python eval.py
```
## File description
eval.py: evaluate function used in training phase <br>
data.py: set up buffer and create dataloader <br>
test.py: evaluate the win rate of blue and red agent after training <br>
utils.py : some auxiliary functions that control the agent <br>
train.py: you can set up some config for enviroment, call function train from Trainer to run cross_q algorithm 

For further details on environment setup and agent interactions, please refer to the MAgent2 documentation.
