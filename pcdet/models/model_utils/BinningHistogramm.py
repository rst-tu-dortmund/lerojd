import torch
import numpy as np
import os

# from ...ops.iou3d_nms.iou3d_nms_utils import boxes_aligned_iou3d_gpu
from ...ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def match_bbox(cls_preds, box_preds, gt, final_pred_labels):
    np_cls_preds = cls_preds.cpu().numpy()
    np_gt = gt.cpu().numpy()

    assert cls_preds.shape[0] == box_preds.shape[0]

    ious = np.zeros([gt.shape[0],1])
    matched_cls = np.zeros([gt.shape[0],3])
    gt_cls = np.zeros([gt.shape[0],3])

    for j in range(gt.shape[0]):
        for i in range(cls_preds.shape[0]):
            iou = boxes_iou3d_gpu(box_preds[i].unsqueeze(0), gt[j, :7].unsqueeze(0))

            np_iou = iou.cpu().numpy()

            if np_iou > ious[j]:
                ious[j] = np_iou
                matched_cls[j] =  np_cls_preds[i]

    gt_cls[np.arange(len(gt_cls)),np_gt[:,7].astype(int)-1] = 1
    
    
    return matched_cls, gt_cls



