"""    
DEIM: DETR with Improved Matching for Fast Convergence    
Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved. 
"""

import torch
import torch.nn as nn    
import torch.distributed
import torch.nn.functional as F 
import torchvision   

import copy 
import numpy as np

from .utils import DensityMapGenerator, visualize_density_map, point_sample, \
    get_uncertain_point_coords_with_randomness, calculate_uncertainty, sigmoid_ce_loss_jit, dice_loss_jit, \
    MaskVisualizer
from .dfine_utils import bbox2distance
from .box_ops import (box_cxcywh_to_xyxy, box_iou, generalized_box_iou,
                      distance_box_iou, complete_box_iou, efficient_box_iou,
                      get_iou_function)
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized 
from ..core import register   

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
SEG_MASK_LOSS_VIS = False   
SEG_MASK_STEP_VIS = 200     

@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """ 
    __share__ = ['num_classes', ] 
    __inject__ = ['matcher', ]     
     
    def __init__(self, \
        matcher,
        weight_dict,   
        losses, 
        alpha=0.2,     
        gamma=2.0,
        num_classes=80,    
        reg_max=32,   
        boxes_weight_format=None, 
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        no_weight_vfl_epoch=-1,
        ccm_params=None,  
        density_recall_penalty=1.1,
        density_precision_penalty=1.3,    
        mask_point_sample_ratio=8,
        # === ACL (Adaptive Curriculum Learning)新增参数 ===
        weight_dict_stage1=None,
        weight_dict_stage2=None,
        matcher_stage1=None,
        matcher_stage2=None,
        flat_epoch=60,
        # === ACL 平滑过渡参数 ===
        transition_mode='hard',  # 'hard' 或 'smooth'
        transition_start=None,    # 过渡开始epoch（默认等于flat_epoch）
        transition_end=None,      # 过渡结束epoch（默认等于flat_epoch）
        gamma_stage1=None,        # 阶段1的gamma
        gamma_stage2=None,         # 阶段2的gamma
        # === IoU 变体选择参数 ===
        iou_type='giou',           # IoU类型: 'iou', 'giou', 'diou', 'ciou', 'eiou'
        ):
        """Create the criterion.
        Parameters:   
            matcher: module able to compute a matching between targets and proposals.     
            weight_dict: dict containing as key the names of the losses and as values their relative weight. 
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.    
            boxes_weight_format: format for boxes weight (iou, ).     
        """     
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict  
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format     
        self.share_matched_indices = share_matched_indices     
        self.alpha = alpha   
        self.gamma = gamma 
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max   
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha 
        self.use_uni_set = use_uni_set 
        self.no_weight_vfl_epoch = no_weight_vfl_epoch
        self.epoch = 0
        self.ccm_params = ccm_params  
        self.density_recall_penalty = density_recall_penalty
        self.density_precision_penalty = density_precision_penalty
        self.mask_point_sample_ratio = mask_point_sample_ratio

        # === ACL阶段配置 (支持平滑渐进) ===
        self.weight_dict_stage1 = weight_dict_stage1
        self.weight_dict_stage2 = weight_dict_stage2
        self.matcher_stage1 = matcher_stage1
        self.matcher_stage2 = matcher_stage2
        self.flat_epoch = flat_epoch
        
        # 平滑过渡配置（新增）
        self.transition_start = transition_start if transition_start is not None else flat_epoch
        self.transition_end = transition_end if transition_end is not None else flat_epoch
        self.transition_mode = transition_mode  # 'hard' 或 'smooth'
        self.gamma_stage1 = gamma_stage1
        self.gamma_stage2 = gamma_stage2
        
        # IoU 变体配置
        self.iou_type = iou_type
        self.iou_fn = get_iou_function(iou_type)
        print(f"{GREEN}IoU type: {iou_type}{RESET}")
        
        self.use_acl = (weight_dict_stage1 is not None and 
                        weight_dict_stage2 is not None)
        if self.use_acl:
            if self.transition_mode == 'smooth' and self.transition_start < self.transition_end:
                print(f"{GREEN}ACL Smooth Transition enabled:{RESET}")
                print(f"  Stage1: epoch < {self.transition_start}")
                print(f"  Transition: epoch [{self.transition_start}, {self.transition_end}]")
                print(f"  Stage2: epoch >= {self.transition_end}")
            else:
                print(f"{GREEN}ACL Hard Switch enabled: stage1 (epoch < {flat_epoch}), stage2 (epoch >= {flat_epoch}){RESET}")
            print(f"  Stage1 weight_dict: {weight_dict_stage1}")
            print(f"  Stage2 weight_dict: {weight_dict_stage2}")
            self._acl_stage1_printed = False
            self._acl_stage2_printed = False
            self._acl_transition_last_epoch = -1
  
        if SEG_MASK_LOSS_VIS:   
            self.visualizer = MaskVisualizer()     
        else: 
            self.visualizer = None 
        self.vis_counter = 0  
  
        if self.no_weight_vfl_epoch != -1: 
            print(RED + f"no_weight_vfl_epoch set {self.no_weight_vfl_epoch}" + RESET)

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']    
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)  
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        return {'loss_cardinality': card_err}
    
    def loss_densitymap(self, outputs, targets, indices, num_boxes):     
        assert 'pred_densitymap' in outputs  
        pred_densitymap = outputs['pred_densitymap'] # bs, 1, h, w   
        bs, c, h, w = pred_densitymap.size()  
        dmg = DensityMapGenerator([h, w])
        density_loss = []
        for b in range(bs):
            boxes = targets[b]["boxes"].cpu().detach().numpy()
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h   
  
            gt_densitymap = torch.from_numpy(dmg.generate_from_boxes(boxes, method='gaussian')).to(pred_densitymap.device)
            gt_densitymap /= (gt_densitymap.max() + 1e-6) # 归一化策略修正   
    
            density_map = pred_densitymap[b].squeeze() / (pred_densitymap.max() + 1e-6) # 归一化策略修正   
            diff = density_map - gt_densitymap 
            underestimation_mask = (density_map < gt_densitymap).float()
            overestimation_mask = (density_map > gt_densitymap).float()
            penalty_weight = 1 + self.density_recall_penalty * gt_densitymap * underestimation_mask + self.density_precision_penalty * gt_densitymap * overestimation_mask
            density_loss.append((penalty_weight * (diff ** 2)).mean())
   
            # image = np.zeros((h, w, 3), dtype=np.uint8)    
            # visualize_density_map(image, gt_densitymap.cpu().detach().numpy(), boxes, save_path=f'result_{b}.png')
  
        return {'loss_densitymap': sum(density_loss) / len(density_loss)}
    
    def loss_densitycount(self, outputs, targets, indices, num_boxes):   
        assert 'pred_densitymap' in outputs
        pred_densitymap = outputs['pred_densitymap'].sum(dim=[1,2,3]) / 300 # bs

        targets_count = [targets[i]['labels'].shape[0] / 300 for i in range(len(targets))]
        targets_count = torch.tensor(targets_count, dtype=torch.float).to(pred_densitymap.device)
   
        return {'loss_densitycount': F.smooth_l1_loss(pred_densitymap, targets_count)}
  
    def loss_ccm(self, outputs, targets, indices, num_boxes):
        assert 'pred_bbox_number' in outputs
        ccm_targets = []    
        for i in range(len(targets)):     
            tgt_num = targets[i]['labels'].shape[0]   
            t = 0
            for j in range(len(self.ccm_params)):    
                if tgt_num >= self.ccm_params[j]:
                    t = j + 1    
            ccm_targets.append(t)
        ccm_targets = torch.tensor(ccm_targets, dtype=torch.int64).to(outputs['pred_bbox_number'].device)
    
        loss = torch.nn.CrossEntropyLoss()(outputs['pred_bbox_number'], ccm_targets) 

        return {'loss_ccm': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs 
        src_logits = outputs['pred_logits']    
        idx = self._get_src_permutation_idx(indices) 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)  
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}
 
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):   
        assert 'pred_boxes' in outputs 
        idx = self._get_src_permutation_idx(indices)     
        if values is None:     
            src_boxes = outputs['pred_boxes'][idx]    
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()   
        else:
            ious = values 

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])   
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o    
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]  

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)  
        target_score_o[idx] = ious.to(target_score_o.dtype) 
        target_score = target_score_o.unsqueeze(-1) * target     
     
        pred_score = F.sigmoid(src_logits).detach()    
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score    
 
        if self.no_weight_vfl_epoch == -1 or self.epoch >= self.no_weight_vfl_epoch:     
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none') 
        else:
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes  
        return {'loss_vfl': loss}  
 
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None): 
        assert 'pred_boxes' in outputs  
        idx = self._get_src_permutation_idx(indices)    
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]   
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))    
            ious = torch.diag(ious).detach()  
        else:   
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) 
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
  
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)    
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target     

        pred_score = F.sigmoid(src_logits).detach()  
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:   
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target     

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        if self.no_weight_vfl_epoch == -1 or self.epoch >= self.no_weight_vfl_epoch: 
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        else:    
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none')    
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}     
  
    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):     
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """     
        assert 'pred_boxes' in outputs 
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]    
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}    
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # 使用配置的 IoU 变体
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # 计算 IoU 度量（注意：iou_fn 返回的是度量值，越大越好）
        if self.iou_type == 'iou':
            iou_metric, _ = self.iou_fn(src_boxes_xyxy, target_boxes_xyxy)
            iou_metric = torch.diag(iou_metric)
        else:
            iou_metric = torch.diag(self.iou_fn(src_boxes_xyxy, target_boxes_xyxy))
        
        loss_giou = 1 - iou_metric
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
  
        return losses    
     
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss    
            and Decoupled Distillation Focal (DDF) Loss. """     

        losses = {}     
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)  
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
 
            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets
  
            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(     
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)  
  
            if 'teacher_corners' in outputs:  
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners): 
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]
    
                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)   
                    mask[idx] = True    
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)  

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()    
    
                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)   
                    if 'is_dn' not in outputs: 
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0 
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0  
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)     

        return losses

    def loss_masks_ps(self, outputs, targets, indices, num_boxes):
        """Compute BCE-with-logits and Dice losses for segmentation masks on matched pairs.
        Expects outputs to contain 'pred_masks' of shape [B, Q, H, W] and targets with key 'masks'.
        """ 
        assert 'pred_masks' in outputs, "pred_masks missing in model outputs"     
        pred_masks = outputs['pred_masks']  # [B, Q, H, W] 
        # gather matched prediction masks     
        idx = self._get_src_permutation_idx(indices)     
        src_masks = pred_masks[idx]  # [N, H, W]   
        # handle no matches
        if src_masks.numel() == 0:     
            return {   
                'loss_mask_ps_ce': src_masks.sum(),
                'loss_mask_ps_dice': src_masks.sum(),     
            }   
        # gather matched target masks    
        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, Ht, Wt]
     
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks.unsqueeze(1)  
        target_masks = target_masks.unsqueeze(1).float()
 
        num_points = max(src_masks.shape[-2], src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio)
    
        with torch.no_grad():    
            # sample point_coords   
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,     
                lambda logits: calculate_uncertainty(logits),  
                num_points,     
                3,
                0.75, 
            ) 
            # get gt labels
            point_labels = point_sample(
                target_masks, 
                point_coords,
                align_corners=False,
                mode="nearest",    
            ).squeeze(1)

        point_logits = point_sample(  
            src_masks,     
            point_coords,
            align_corners=False,     
        ).squeeze(1)

        if self.visualizer and self.vis_counter % SEG_MASK_STEP_VIS == 0:
            target_masks = target_masks.squeeze(1)
            N, H, W = target_masks.shape
        
            # 上采样target masks
            src_masks = F.interpolate(     
                src_masks,  # [N, 1, Ht, Wt]     
                size=(H, W),  # 下采样到预测mask的尺寸
                mode='nearest'
            ).squeeze(1)  # [N, H, W]

            # 创建边界框mask，只保留box内的区域 
            box_masks = torch.zeros_like(target_masks)  # [N, Ht, Wt]
    
            target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, 4] (cx, cy, w, h)    
            for i in range(N):
                cx, cy, w_box, h_box = target_boxes[i]  
                # 转换归一化坐标到像素坐标（在预测mask的尺寸上）
                x1 = int((cx - w_box / 2) * W)
                y1 = int((cy - h_box / 2) * H)
                x2 = int((cx + w_box / 2) * W)    
                y2 = int((cy + h_box / 2) * H)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(0, min(x2, W))  
                y2 = max(0, min(y2, H))
                
                # 在box区域内设置为1
                box_masks[i, y1:y2, x1:x2] = 1.0

            with torch.no_grad():
                self.visualizer.visualize_batch(
                    src_masks, 
                    target_masks,  
                    box_masks,   
                    target_boxes,
                    max_samples=16
                )
        self.vis_counter += 1
  
        losses = {
            "loss_mask_ps_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_ps_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }
    
        del src_masks
        del target_masks  
        return losses  

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices  
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])     
        src_idx = torch.cat([src for (src, _) in indices]) 
        return batch_idx, src_idx     
     
    def _get_tgt_permutation_idx(self, indices):    
        # permute targets following indices 
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
  
    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []     
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]
   
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)  
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}   
            for idx in unique_sorted:    
                row_idx, col_idx = idx[0].item(), idx[1].item()     
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx     
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)  
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)  
            results.append((final_rows.long(), final_cols.long()))  
        return results
    
    def _clear_cache(self): 
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None  
        self.num_pos, self.num_neg = None, None
 
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {  
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl, 
            'mal': self.loss_labels_mal,
            'local': self.loss_local,    
            'cardinality': self.loss_cardinality, 
            'ccm': self.loss_ccm, 
            'densitymap': self.loss_densitymap,
            'densitycount': self.loss_densitycount,
            'mask_ps': self.loss_masks_ps     
        }    
        assert loss in loss_map, f'do you really want to compute {loss} loss?'   
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.     
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.   
                      The expected keys in each dict depends on the losses applied, see each loss' doc 
        """
        epoch = kwargs.get('epoch', 0)

        # === ACL阶段切换逻辑（支持平滑渐进）===
        if self.use_acl:
            # 判断是否处于过渡阶段
            in_transition = (self.transition_mode == 'smooth' and 
                            self.transition_start < self.transition_end and
                            self.transition_start <= epoch < self.transition_end)
            
            if in_transition:
                # 平滑过渡：线性插值计算权重
                alpha = (epoch - self.transition_start) / (self.transition_end - self.transition_start)
                alpha = max(0.0, min(1.0, alpha))  #  clamp to [0, 1]
                
                # 插值计算当前权重
                self.weight_dict = {}
                for key in self.weight_dict_stage1.keys():
                    v1 = self.weight_dict_stage1[key]
                    v2 = self.weight_dict_stage2[key]
                    self.weight_dict[key] = v1 * (1 - alpha) + v2 * alpha
                
                # 插值matcher配置
                current_matcher_config = {}
                if self.matcher_stage1 and self.matcher_stage2:
                    for key in self.matcher_stage1.keys():
                        v1 = self.matcher_stage1[key]
                        v2 = self.matcher_stage2[key]
                        current_matcher_config[key] = v1 * (1 - alpha) + v2 * alpha
                
                # 插值gamma
                if self.gamma_stage1 is not None and self.gamma_stage2 is not None:
                    self.gamma = self.gamma_stage1 * (1 - alpha) + self.gamma_stage2 * alpha
                
                # 过渡状态：每5个epoch打印一次（每个epoch只打印一次）
                if epoch % 5 == 0 and epoch != self._acl_transition_last_epoch:
                    self._acl_transition_last_epoch = epoch
                    print(f"{YELLOW}ACL Transition epoch {epoch}: alpha={alpha:.2f}{RESET}")
                    print(f"  Current weights: {self.weight_dict}")
                    
            elif epoch < self.transition_start:
                # Stage 1: 分类主导
                self.weight_dict = self.weight_dict_stage1
                current_matcher_config = self.matcher_stage1
                if self.gamma_stage1 is not None:
                    self.gamma = self.gamma_stage1
                    
                # 打印进入Stage 1提示（仅一次）
                if not self._acl_stage1_printed:
                    self._acl_stage1_printed = True
                    print(f"{GREEN}=== ACL Stage 1: epoch {epoch} ==={RESET}")
                    print(f"  weight_dict: {self.weight_dict}")
                    
            else:
                # Stage 2: 定位主导（仅第一次进入时打印）
                if not self._acl_stage2_printed:
                    self._acl_stage2_printed = True
                    print(f"{YELLOW}=== ACL Stage Switch: epoch {epoch}, entering Stage 2 ==={RESET}")
                    print(f"  New weight_dict: {self.weight_dict_stage2}")
                    if self.matcher_stage2:
                        print(f"  New matcher config: {self.matcher_stage2}")
                
                self.weight_dict = self.weight_dict_stage2
                current_matcher_config = self.matcher_stage2
                if self.gamma_stage2 is not None:
                    self.gamma = self.gamma_stage2
            
            # 动态更新匹配器配置（如果提供了）
            if current_matcher_config is not None and hasattr(self, 'matcher'):
                if hasattr(self.matcher, 'cost_class'):
                    self.matcher.cost_class = current_matcher_config.get('cost_class', 2)
                    self.matcher.cost_bbox = current_matcher_config.get('cost_bbox', 5)
                    self.matcher.cost_giou = current_matcher_config.get('cost_giou', 2)

        # 处理Dynamic-Query的多batch问题   
        num_queries_list = outputs.get('num_queries_list', None)
        loss_filter = ['ccm', 'densitymap', 'densitycount']
  
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets 
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch, num_queries_list=num_queries_list)['indices']    
        self._clear_cache() 
   
        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:   
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']  
            if 'pre_outputs' in outputs:    
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets, epoch=epoch, num_queries_list=num_queries_list)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)    
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets, epoch=epoch, num_queries_list=num_queries_list)['indices']    
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():     
                torch.distributed.all_reduce(num_boxes_go)    
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item() 
        else:
            assert 'aux_outputs' in outputs, ''
    
        # Compute the average number of target boxes accross all nodes, for normalization purposes    
        num_boxes = sum(len(t["labels"]) for t in targets)  
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)     
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)   
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
  
        # Compute all the requested losses, main loss
        losses = {}   
        for loss in self.losses:    
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])   
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)    
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.  
        if 'aux_outputs' in outputs:     
            for i, aux_outputs in enumerate(outputs['aux_outputs']):    
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']     
                for loss in self.losses:
                    if loss in loss_filter and 'mask' not in loss: 
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local']) 
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}  
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)  
   
        # In case of auxiliary traditional head output at first decoder layer. just for dfine    
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']    
            for loss in self.losses:
                if loss in loss_filter or 'mask' in loss:    
                    continue
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])   
                indices_in = indices_go if use_uni_set else cached_indices[-1]   
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes   
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)    
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}  
                l_dict = {k + '_pre': v for k, v in l_dict.items()}   
                losses.update(l_dict)     

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:    
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic'] 
            if class_agnostic:
                orig_num_classes = self.num_classes    
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)  
                for t in enc_targets:  
                    t['labels'] = torch.zeros_like(t["labels"])     
            else:
                enc_targets = targets   

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):   
                for loss in self.losses:
                    if loss in loss_filter: 
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes   
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)    
  
            if class_agnostic:
                self.num_classes = orig_num_classes     
  
        # In case of cdn auxiliary losses.    
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']     
    
            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss     
                    aux_outputs['is_dn'] = True  
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss in loss_filter:   
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)  
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)   
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}   
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)  

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    if loss in loss_filter or 'mask' in loss:     
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}     
                    losses.update(l_dict)   

        # For debugging Objects365 pre-train.     
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}    
        return losses 
     
    def get_loss_meta_info(self, loss, outputs, targets, indices):     
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]     
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)     
 
        # 支持多种 IoU 变体作为权重
        if self.boxes_weight_format in ['iou', 'giou', 'diou', 'ciou', 'eiou']:
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes.detach())
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            
            # 根据配置选择 IoU 函数
            if self.boxes_weight_format == 'iou':
                iou_metric, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
                iou = torch.diag(iou_metric)
            elif self.boxes_weight_format == 'giou':
                iou = torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
            elif self.boxes_weight_format == 'diou':
                iou = torch.diag(distance_box_iou(src_boxes_xyxy, target_boxes_xyxy))
            elif self.boxes_weight_format == 'ciou':
                iou = torch.diag(complete_box_iou(src_boxes_xyxy, target_boxes_xyxy))
            elif self.boxes_weight_format == 'eiou':
                iou = torch.diag(efficient_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        else:
            raise AttributeError(f"Unknown boxes_weight_format: {self.boxes_weight_format}")

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou} 
        elif loss in ('vfl', 'mal'):   
            meta = {'values': iou}
        else:  
            meta = {}

        return meta  
 
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices     
        """  
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets] 
        device = targets[0]['labels'].device

        dn_match_indices = []   
        for i, num_gt in enumerate(num_gts):  
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx) 
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:     
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))   
 
        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()   
        return torch.abs(loss)  


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):    
        dis_left = label.long() 
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)   

        if weight is not None:
            weight = weight.float()    
            loss = loss * weight

        if avg_factor is not None:    
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':   
            loss = loss.sum()     

        return loss    

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]   
        return opt_list

    def set_epoch(self, epoch):
        self.epoch = epoch    
        if self.no_weight_vfl_epoch != -1 and self.epoch == self.no_weight_vfl_epoch:
            print(RED + f"Epoch:[{self.epoch}]>=[{self.no_weight_vfl_epoch}] using weight in vfl/mal." + RESET) 
