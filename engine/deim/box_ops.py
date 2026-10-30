"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved 
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py   
"""     

import torch
from torch import Tensor    
from torchvision.ops.boxes import box_area
   

def box_cxcywh_to_xyxy(x):  
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))] 
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)  

    
# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):   
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] 
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
   
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union   
 

def generalized_box_iou(boxes1, boxes2):     
    """
    Generalized IoU from https://giou.stanford.edu/     
 
    The boxes should be in [x0, y0, x1, y1] format    
    
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results 
    # so do an early check   
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()  
    iou, union = box_iou(boxes1, boxes2)     

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])    
     
    wh = (rb - lt).clamp(min=0)  # [N,M,2]  
    area = wh[:, :, 0] * wh[:, :, 1] 

    return iou - (area - union) / area


def distance_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Distance IoU (DIoU) from "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    https://arxiv.org/abs/1911.08287
    
    DIoU = IoU - (center_distance^2) / (diagonal_length^2)
    
    The boxes should be in [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix
    """
    iou, union = box_iou(boxes1, boxes2)
    
    # 计算中心点
    b1_cx = (boxes1[:, None, 0] + boxes1[:, None, 2]) / 2
    b1_cy = (boxes1[:, None, 1] + boxes1[:, None, 3]) / 2
    b2_cx = (boxes2[:, 0] + boxes2[:, 2]) / 2
    b2_cy = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    # 中心点距离的平方
    center_dist_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
    
    # 最小包围框的对角线长度的平方
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    diag_sq = ((rb - lt) ** 2).sum(dim=-1)
    
    diou = iou - center_dist_sq / (diag_sq + eps)
    return diou


def complete_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Complete IoU (CIoU) from "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    https://arxiv.org/abs/1911.08287
    
    CIoU = IoU - (center_distance^2) / (diagonal_length^2) - alpha * v
    where v measures aspect ratio consistency, alpha is a trade-off parameter
    
    The boxes should be in [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix
    """
    iou, union = box_iou(boxes1, boxes2)
    
    # DIoU 部分
    b1_cx = (boxes1[:, None, 0] + boxes1[:, None, 2]) / 2
    b1_cy = (boxes1[:, None, 1] + boxes1[:, None, 3]) / 2
    b2_cx = (boxes2[:, 0] + boxes2[:, 2]) / 2
    b2_cy = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    center_dist_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    diag_sq = ((rb - lt) ** 2).sum(dim=-1)
    
    # 计算宽高和 aspect ratio
    b1_w = boxes1[:, None, 2] - boxes1[:, None, 0]
    b1_h = boxes1[:, None, 3] - boxes1[:, None, 1]
    b2_w = boxes2[:, 2] - boxes2[:, 0]
    b2_h = boxes2[:, 3] - boxes2[:, 1]
    
    # v = (4/pi^2) * (arctan(w1/h1) - arctan(w2/h2))^2
    v = (4 / (torch.pi ** 2)) * (torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps))) ** 2
    
    # alpha = v / ((1 - IoU) + v)
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)
    
    ciou = iou - center_dist_sq / (diag_sq + eps) - alpha * v
    return ciou


def efficient_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Efficient IoU (EIoU) from "Focal and Efficient IoU Loss for Accurate Bounding Box Regression"
    https://arxiv.org/abs/2101.08158
    
    EIoU = IoU + (width_distance^2) / (C_w^2) + (height_distance^2) / (C_h^2)
    
    独立惩罚宽度、高度和中心点距离，相比 CIoU 更简单且避免 aspect ratio 带来的退化问题
    
    The boxes should be in [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix
    """
    iou, union = box_iou(boxes1, boxes2)
    
    # 中心点距离（分别计算 x 和 y）
    b1_cx = (boxes1[:, None, 0] + boxes1[:, None, 2]) / 2
    b1_cy = (boxes1[:, None, 1] + boxes1[:, None, 3]) / 2
    b2_cx = (boxes2[:, 0] + boxes2[:, 2]) / 2
    b2_cy = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    # 宽度和高度
    b1_w = boxes1[:, None, 2] - boxes1[:, None, 0]
    b1_h = boxes1[:, None, 3] - boxes1[:, None, 1]
    b2_w = boxes2[:, 2] - boxes2[:, 0]
    b2_h = boxes2[:, 3] - boxes2[:, 1]
    
    # 最小包围框
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    # 包围框的宽度和高度
    c_w = rb[:, :, 0] - lt[:, :, 0]
    c_h = rb[:, :, 1] - lt[:, :, 1]
    
    # 中心点距离惩罚（分解到 x 和 y）
    center_dist_x = (b1_cx - b2_cx) ** 2
    center_dist_y = (b1_cy - b2_cy) ** 2
    
    # 宽度和高度差异惩罚
    width_dist = (b1_w - b2_w) ** 2
    height_dist = (b1_h - b2_h) ** 2
    
    # EIoU 公式（注意：EIou 论文中是 1 - IoU + dist_terms，所以返回负值用于损失）
    # 返回 1 - EIoU 可以直接作为损失，或者返回 EIoU 作为相似度度量
    # 这里返回 EIoU 相似度（越大越好），使用时 loss = 1 - eiou
    eiou = iou - center_dist_x / (c_w ** 2 + eps) - center_dist_y / (c_h ** 2 + eps) \
           - width_dist / (c_w ** 2 + eps) - height_dist / (c_h ** 2 + eps)
    
    return eiou


# IoU 变体映射表，用于通过名称选择
IOU_FUNCTIONS = {
    'iou': box_iou,
    'giou': generalized_box_iou,
    'diou': distance_box_iou,
    'ciou': complete_box_iou,
    'eiou': efficient_box_iou,
}


def get_iou_function(iou_type='giou'):
    """
    根据名称获取 IoU 函数
    
    Args:
        iou_type: str, 可选 'iou', 'giou', 'diou', 'ciou', 'eiou'
    
    Returns:
        IoU 计算函数
    """
    if iou_type not in IOU_FUNCTIONS:
        raise ValueError(f"Unknown IoU type: {iou_type}. Available: {list(IOU_FUNCTIONS.keys())}")
    return IOU_FUNCTIONS[iou_type]


def compute_iou_loss(boxes1, boxes2, iou_type='giou', reduction='mean', eps=1e-7):
    """
    统一的 IoU 损失计算接口
    
    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [M, 4] in xyxy format
        iou_type: str, IoU 类型
        reduction: str, 'mean', 'sum', or 'none'
        eps: float, 数值稳定性
    
    Returns:
        IoU 损失值（对于非 IoU 类型，返回 1 - iou_metric）
    """
    iou_fn = get_iou_function(iou_type)
    
    if iou_type == 'iou':
        iou, _ = iou_fn(boxes1, boxes2)
        loss = 1 - iou
    else:
        # GIoU, DIoU, CIoU, EIoU 都返回度量值（越大越好），需要 1 - metric 转为损失
        metric = iou_fn(boxes1, boxes2, eps=eps) if iou_type in ['diou', 'ciou', 'eiou'] else iou_fn(boxes1, boxes2)
        loss = 1 - metric
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss
    
    return loss
 
   
def masks_to_boxes(masks):   
    """Compute the bounding boxes around the provided masks
 
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device) 
   
    h, w = masks.shape[-2:]     

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))  
    x_max = x_mask.flatten(1).max(-1)[0]   
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]   

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]   
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
   
    return torch.stack([x_min, y_min, x_max, y_max], 1)