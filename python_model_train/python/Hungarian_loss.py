import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianDETRLoss(nn.Module):
    def __init__(self, num_classes=10, no_object_weight=0.1, class_cost_weight=1.0, bbox_cost_weight=5.0,
        giou_cost_weight=2.0, bbox_loss_weight=5.0, giou_loss_weight=2.0, image_width=960,image_height=540):
        super().__init__()
        self.num_classes = num_classes
        # Class 0-9 = actual objects
        # Class 10 = no-object
        self.no_object_class = num_classes
        self.class_cost_weight = class_cost_weight
        self.bbox_cost_weight = bbox_cost_weight
        self.giou_cost_weight = giou_cost_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.image_width = image_width
        self.image_height = image_height
        # Classification weight
        class_weights = torch.ones(num_classes + 1)
        # Reduce contribution of no-object class
        class_weights[self.no_object_class] = no_object_weight
        self.register_buffer("class_weights", class_weights)
    # ---------------------------------------------------------
    # xyxy -> cxcywh
    # ---------------------------------------------------------
    def xyxy_to_cxcywh(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    # ---------------------------------------------------------
    # Normalize ground-truth boxes
    # ---------------------------------------------------------
    def normalize_boxes(self, boxes):

        boxes = boxes.clone()
        boxes[:, 0] /= self.image_width
        boxes[:, 1] /= self.image_height
        boxes[:, 2] /= self.image_width
        boxes[:, 3] /= self.image_height
        return boxes
    # cxcywh -> xyxy
    def cxcywh_to_xyxy(self, boxes):
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2],dim=-1)
    # Pairwise GIoU
    # boxes1 -> [N,4]
    # boxes2 -> [M,4]
    # output -> [N,M]
    def generalized_iou(self, boxes1, boxes2):
        boxes1 = self.cxcywh_to_xyxy(boxes1)
        boxes2 = self.cxcywh_to_xyxy(boxes2)
        # boxes1: [N,4]
        # boxes2: [M,4]
        area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)*(boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
        area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)*(boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
        # Intersection
        lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[..., 0] * wh[..., 1]
        # Union
        union = (area1[:, None]+ area2[None, :]- intersection)
        iou = intersection / (union + 1e-7)
        # Smallest enclosing box
        enclosing_lt = torch.minimum(boxes1[:, None, :2],boxes2[None, :, :2])
        enclosing_rb = torch.maximum(boxes1[:, None, 2:],boxes2[None, :, 2:])
        enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
        enclosing_area = (enclosing_wh[..., 0]*enclosing_wh[..., 1])
        giou = iou - ((enclosing_area - union)/(enclosing_area + 1e-7))
        return giou

    # ---------------------------------------------------------
    # Hungarian matching for ONE image
    # ---------------------------------------------------------
    def match(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        # pred_logits:
        # [100, 11]
        # pred_boxes:
        # [100, 4]
        # gt_labels:
        # [N]
        # gt_boxes:
        # [N, 4]
        num_predictions = pred_logits.shape[0]
        num_targets = gt_labels.shape[0]
        # No objects in image
        if num_targets == 0:
            return (torch.empty(0,dtype=torch.long, device=pred_logits.device),
                torch.empty(0, dtype=torch.long, device=pred_logits.device)
            )
        # Classification cost
        probabilities = pred_logits.softmax(dim=-1)
        # [100, N]
        class_cost = -probabilities[:, gt_labels]
        # Bounding box L1 cost
        bbox_cost = torch.cdist(pred_boxes, gt_boxes, p=1)
        # [100,N]
        # GIoU cost
        giou = self.generalized_iou(pred_boxes, gt_boxes)
        # GIoU should be maximized.
        # Hungarian algorithm minimizes cost.
        giou_cost = -giou
        # Total matching cost
        total_cost = (
            self.class_cost_weight * class_cost
            +
            self.bbox_cost_weight * bbox_cost
            +
            self.giou_cost_weight * giou_cost
        )
        # Hungarian matching does not need gradients.
        total_cost = total_cost.detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(total_cost)
        row_indices = torch.tensor(row_indices, dtype=torch.long, device=pred_logits.device)
        col_indices = torch.tensor(col_indices, dtype=torch.long, device=pred_logits.device)
        return row_indices, col_indices

    # Forward
    def forward(self, pred_logits, pred_boxes, targets):
        batch_size = pred_logits.shape[0]
        total_class_loss = torch.tensor(0.0, device=pred_logits.device)
        total_bbox_loss = torch.tensor(0.0, device=pred_logits.device)
        total_giou_loss = torch.tensor(0.0, device=pred_logits.device)
        total_objects = 0
        # -----------------------------------------------------
        # Process every image independently
        # -----------------------------------------------------
        for b in range(batch_size):
            # Predictions for image b
            logits = pred_logits[b]
            boxes = pred_boxes[b]
            # Ground truth
            gt_labels = targets[b]["labels"].to(pred_logits.device)
            gt_boxes = targets[b]["boxes"].to(pred_logits.device)
            # Convert GT:
            # xyxy pixels
            #      ↓
            # cxcywh pixels
            #      ↓
            # normalized cxcywh
            gt_boxes = self.xyxy_to_cxcywh(gt_boxes)
            gt_boxes = self.normalize_boxes(gt_boxes)
            num_gt = gt_labels.shape[0]
            total_objects += num_gt
            # Hungarian matching
            matched_pred, matched_gt = self.match(logits, boxes, gt_labels, gt_boxes)
            # Classification targets for ALL queries
            target_classes = torch.full((logits.shape[0],), self.no_object_class, dtype=torch.long, device=logits.device)
            # Put real classes into matched queries
            if len(matched_pred) > 0:
                target_classes[matched_pred] = (gt_labels[matched_gt])
            # Classification loss for ALL 100 queries
            class_loss = F.cross_entropy(logits, target_classes, weight=self.class_weights)
            total_class_loss += class_loss
            # Box losses ONLY for matched predictions
            if len(matched_pred) > 0:
                matched_boxes = boxes[matched_pred]
                matched_gt_boxes = gt_boxes[matched_gt]
                # L1
                bbox_loss = F.l1_loss(matched_boxes, matched_gt_boxes, reduction="sum")
                # GIoU
                matched_giou = self.generalized_iou(matched_boxes,matched_gt_boxes)
                # Diagonal = corresponding pairs
                giou_loss = (1.0 - torch.diagonal(matched_giou)).sum()
                total_bbox_loss += bbox_loss
                total_giou_loss += giou_loss
        # Normalize
        total_objects = max(total_objects,1)
        total_class_loss /= batch_size
        total_bbox_loss /= total_objects
        total_giou_loss /= total_objects
        # Final DETR loss
        total_loss = (total_class_loss + self.bbox_loss_weight * total_bbox_loss + self.giou_loss_weight * total_giou_loss)
        return {
            "loss": total_loss,
            "classification_loss": total_class_loss,
            "bbox_loss": total_bbox_loss,
            "giou_loss": total_giou_loss
        }