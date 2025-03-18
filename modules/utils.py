import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import torch
import torchvision
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from random import randint, choice, shuffle, choices
import cv2
import torch
from torch import nn
import torch.nn.functional as F

BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
S = 13
BOX = 5
CLS = 1
H, W = 416, 416
OUTPUT_THRESH = 0.7
# pre-defined anchor-boxes
ANCHOR_BOXS = [[1.08,1.19],
               [3.42,4.41],
               [6.63,11.38],
               [9.42,5.11],
               [16.62,10.52]]


def plot_img(img, size=(7,7)):
    plt.figure(figsize=size)
    plt.imshow(img[:,:,::-1])
    plt.show()
    plt.axis('off')  # Optional: removes axes
    plt.savefig('output_image.png')

scale = 770

def visualize_bbox(img, boxes, thickness=2, color=BOX_COLOR, draw_center=True):
    img_copy = img.cpu().permute(1,2,0).numpy() if isinstance(img, torch.Tensor) else img.copy()
    for box in boxes:
        x,y,w,h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_copy = cv2.rectangle(
            img_copy,
            (x,y),(x+w, y+h),
            color, thickness)
        if draw_center:
            center = (x+w//2, y+h//2)
            img_copy = cv2.circle(img_copy, center=center, radius=3, color=(0,255,0), thickness=2)
        
        conf = float(box[-2])
        cls_id = int(box[-1])
        
        # Prepare label text with class ID and confidence
        label = f"{cls_id}, {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
                img_copy, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img_copy

from tqdm import tqdm

def output_tensor_to_boxes(boxes_tensor):
    cell_w, cell_h = W/S, H/S
    boxes = []
    probs = []
    
    for i in range(S):
        for j in range(S):
            for b in range(BOX):
                anchor_wh = torch.tensor(ANCHOR_BOXS[b])
                data = boxes_tensor[i,j,b]
                xy = torch.sigmoid(data[:2])
                wh = torch.exp(data[2:4])*anchor_wh
                obj_prob = torch.sigmoid(data[4:5])
                cls_prob = torch.softmax(data[5:], dim=-1)
                combine_prob = obj_prob*max(cls_prob)
                
                if combine_prob > OUTPUT_THRESH:
                    x_center, y_center, w, h = xy[0], xy[1], wh[0], wh[1]
                    x, y = x_center+j-w/2, y_center+i-h/2
                    x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                    box = [x,y,w,h, combine_prob, torch.round(data[5])]
                    boxes.append(box)
    return boxes


def overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def compute_iou(box1, box2):
    """Compute IOU between box1 and box2"""
    x1,y1,w1,h1 = box1[0], box1[1], box1[2], box1[3]
    x2,y2,w2,h2 = box2[0], box2[1], box2[2], box2[3]
    
    ## if box2 is inside box1
    if (x1 < x2) and (y1<y2) and (w1>w2) and (h1>h2):
        return 1
    
    area1, area2 = w1*h1, w2*h2
    intersect_w = overlap((x1,x1+w1), (x2,x2+w2))
    intersect_h = overlap((y1,y1+h1), (y2,y2+w2))
    intersect_area = intersect_w*intersect_h
    iou = intersect_area/(area1 + area2 - intersect_area)
    return iou

def nonmax_suppression(boxes, IOU_THRESH = 0.3):
    """remove ovelap bboxes"""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    for i, current_box in enumerate(boxes):
        if current_box[4] <= 0:
            continue
        for j in range(i+1, len(boxes)):
            iou = compute_iou(current_box, boxes[j])
            if iou > IOU_THRESH:
                boxes[j][4] = 0
    boxes = [box for box in boxes if box[4] > 0]
    return boxes

