# Contour-Detection
## Overview
This is my implementation for the contour-detection task given to me during batch13 vinAI interview. In short, this model aim to detect the bounding box, the center, the number of circle for each plot in the image. The main method used here is YOLO - You Only Look Once. I have coded it from scratch (TinyYOLO) and utilized ultralytics library so i can provide a comprehensive comparison between each model but it turn out that my scratch model has to faced some problems.

### Overview of the dataset

## Challenge
- Overlap circle : Some generated images have their contour plot overlap on each other, making detection task more challenging.
- Metrics : Intuitively, the number of circle in each plot can be somehow perceived as "class" and we can do some no-brain move like fit the whole yolo model with "number of circle" as a categorical label but i have tried to implement yolo model with MSE, MAE, ... loss function but it messed up the yolo's paper loss function so much that at the end, i can hardly compare loss of my yolo scratch model with ultralytic model.
- Code : The code is a little bit messy as i have to debug a lot.

## Result

## Some demo result

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
