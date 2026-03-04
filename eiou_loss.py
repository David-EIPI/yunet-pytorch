import torch
import torch.nn as nn
import torch.nn.functional as F

# Computes detection confidence loss and Extended IoU loss (from Peng et al., 2021)

def eiou_loss(pred, target, smooth_point=0.1, eps=1e-7):
    r"""Implementation of paper 'Extended-IoU Loss: A Systematic IoU-Related
     Method: Beyond Simplified Regression for Better Localization,

     <https://ieeexplore.ieee.org/abstract/document/9429909> '.

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    tx1, ty1, tx2, ty2 = target[..., 0], target[..., 1], target[..., 2], target[..., 3]

    # extent top left
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)

    # intersection coordinates
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    # extra
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)

    # Intersection
    intersection = (ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1) - (
        ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (
            iy1 - ey1)
    # Union
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (
        ty2 - ty1) - intersection + eps
    # IoU
    ious = 1 - (intersection / union)

    # Smooth-EIoU
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * (ious**2) / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point)
    return loss

def safe_cat_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    # Concatenates tensors in a list (possibly empty).
    # Needed to avoid exception thrown from torch.cat for empty input lists.

    if tensors == []:
        return torch.tensor([])
    else:
        return torch.cat(tensors)

from utils import decode_boxes

class easyLoss(nn.Module):
    def __init__(self, anchors: torch.Tensor, batch_size: int, device='cpu'):
        super().__init__()
        self.anchors = anchors.to(device)
        self.anchors_tile = torch.tile(self.anchors, (batch_size, 1))
        self.device = device

    def forward(self, predictions: tuple, targets: tuple) -> tuple:

        # ground truth inputs
        images, boxes, anchor_indexes, cls = targets
        boxes_dev = [ b.to(self.device) for b in boxes ]

        # predictions is a tuple (NxCx1, NxCx4), C is the total number of boxes on all levels, box (4x floats) + confidence (1x float)
        pred_boxes = predictions[1].to(device = self.device)
        pred_conf = predictions[0].to(device = self.device)


# Prepare a flattened array of anchor indexes into the flattened arrays of predictions filtered by non-zero class presence
        ai_idx = [ i * pred_conf.shape[1] + ai for i,ai in enumerate(anchor_indexes) if ai.numel() > 0 ]

        ai_idx_t = safe_cat_tensors(ai_idx).to(dtype = torch.long, device=self.device)

        cl_idx = [ torch.tensor( [ cls[i] ] * ai.numel(), dtype = torch.int ) for i,ai in enumerate(anchor_indexes) ]
        cl_idx_t = torch.cat(cl_idx).to(dtype = torch.int, device=self.device)

        gt_boxes = [ b for i,b in enumerate(boxes) if cls[i] > 0 ]
        gt_boxes_t = safe_cat_tensors( gt_boxes ).to(device=self.device)

        batch_idx = [ torch.tensor( [ i ] * ai.numel(), dtype = torch.int ) for i,ai in enumerate(ai_idx) ]
        batch_idx_t = safe_cat_tensors( batch_idx ).to(device = self.device, dtype = torch.int)


# Calculate confidence loss
        gt_conf = torch.zeros_like(pred_conf, requires_grad=False, device = self.device)
        gt_conf.put_(ai_idx_t, torch.ones_like(ai_idx_t, device = gt_conf.device, dtype=gt_conf.dtype))

        gt_conf = gt_conf.to(device = self.device)

        box_neg_mask = torch.empty_like(pred_conf, requires_grad=False, device = self.device).fill_(0.002)

        box_neg_mask.put_(ai_idx_t, torch.ones_like(ai_idx_t, device = gt_conf.device, dtype=gt_conf.dtype))

        conf = torch.sigmoid(pred_conf)
        loss_c = F.binary_cross_entropy(conf, gt_conf, reduction = 'none')
        loss_c *= box_neg_mask
        loss_c = torch.sum(loss_c)


# Calculate box loss
        loss_e = torch.zeros((1,), dtype=torch.float32, device=self.device)

        if ai_idx_t.numel() > 0:

# Compare gt boxes to the decoded predictions. Copy gt boxes to match predictions count
# (there may be > 1 if several anchor boxes were matched to this gt)

            pb = pred_boxes.flatten(0,-2)
            decoded = decode_boxes( pb[ai_idx_t], self.anchors_tile[ai_idx_t] )
            el = eiou_loss(decoded, gt_boxes_t)
            best_losses = torch.zeros_like(el, requires_grad=False, device = self.device).scatter_reduce(0, batch_idx_t, el, reduce="amin", include_self=False)
            loss_e += torch.sum(best_losses)


        return loss_c, loss_e
