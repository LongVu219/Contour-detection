import torch
from dataset import *
from model import *

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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def post_process_output(output):
    """Convert output of model to pred_xywh"""
    # xy
    xy = torch.sigmoid(output[:,:,:,:,:2]+1e-6)

    # wh
    wh = output[:,:,:,:,2:4]
    anchors_wh = torch.Tensor(ANCHOR_BOXS).view(1,1,1,BOX,2).to(device)
    wh = torch.exp(wh)*anchors_wh
    
    # objectness confidence
    obj_prob = torch.sigmoid(output[:,:,:,:,4:5]+1e-6)
    
    # class distribution
    cls_dist = output[:,:,:,:,5]
    return xy, wh, obj_prob, cls_dist

def post_process_target(target_tensor):
    xy = target_tensor[:,:,:,:,:2]
    wh = target_tensor[:,:,:,:,2:4]
    obj_prob = target_tensor[:,:,:,:,4:5]
    cls_dist = target_tensor[:,:,:,:,5]
    return xy, wh, obj_prob, cls_dist

def square_error(output, target):
    return (output-target)**2

def custom_loss(output_tensor, target_tensor):
    cell_size = (W/S, H/S)
    NOOB_W, CONF_W, XY_W, PROB_W, WH_W = 2.0, 10.0, 0.5, 1.0, 0.1

    pred_xy, pred_wh, pred_obj_prob, pred_cls_dist = post_process_output(output_tensor)
    true_xy, true_wh, true_obj_prob, true_cls_dist = post_process_target(target_tensor)
    
    pred_cls_dist = pred_cls_dist.unsqueeze(-1)
    true_cls_dist = true_cls_dist.unsqueeze(-1)

    #print(pred_xy.shape, pred_wh.shape, pred_obj_prob.shape, pred_cls_dist.shape)
    #print(true_xy.shape, true_wh.shape, true_obj_prob.shape, true_cls_dist.shape)

    # cal pred_bbox area
    pred_ul = pred_xy - 0.5*pred_wh
    pred_br = pred_xy + 0.5*pred_wh
    pred_area = pred_wh[:,:,:,:,0]*pred_wh[:,:,:,:,1]
    
    # cal true_box area
    true_ul = true_xy - 0.5*true_wh
    true_br = true_xy + 0.5*true_wh
    true_area = true_wh[:,:,:,:,0]*true_wh[:,:,:,:,1]

    # cal iou
    intersect_ul = torch.max(pred_ul, true_ul)
    intersect_br = torch.min(pred_br, true_br)
    intersect_wh = intersect_br - intersect_ul
    intersect_area = intersect_wh[:,:,:,:,0]*intersect_wh[:,:,:,:,1]
    
    # get the best matching bounding box
    iou = intersect_area/(pred_area + true_area - intersect_area)
    max_iou = torch.max(iou, dim=3, keepdim=True)[0]
    best_box_index =  torch.unsqueeze(torch.eq(iou, max_iou).float(), dim=-1)
    true_box_conf = best_box_index*true_obj_prob
    
    xy_loss =  (square_error(pred_xy, true_xy)*true_box_conf*XY_W).sum()
    wh_loss =  (square_error(pred_wh, true_wh)*true_box_conf*WH_W).sum()
    obj_loss = (square_error(pred_obj_prob, true_obj_prob)*(CONF_W*true_box_conf + NOOB_W*(1-true_box_conf))).sum()
    
    #print(pred_cls_dist.shape, true_cls_dist.shape, true_box_conf.shape, true_obj_prob.shape)
    #import sys
    #sys.exit() 
    
    cls_loss = (square_error(pred_cls_dist, true_cls_dist)*true_box_conf*PROB_W).sum()

    # Final loss
    total_loss = xy_loss + wh_loss + obj_loss + cls_loss
    return total_loss, (xy_loss + wh_loss, obj_loss, cls_loss)

