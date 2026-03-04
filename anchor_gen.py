import numpy as np
import torch
from torch.nn.modules.utils import _pair

class AnchorGenerator:
    def __init__(self, num_level_anchors, image_size = (1.0, 1.0)):
        super().__init__()
        size = _pair(image_size)
        self.im_width, self.im_height = size
        self.num_anchors = [a for a in num_level_anchors]

    # offset = 0 for top/left, 0.5 for center, 1.0 for bottom/right
    def generate_anchors(self, offsets = 0, as_boxes = False):
        ofs = _pair(offsets)
        anchors = []
        for l in self.num_anchors:
            step_h = self.im_height / l
            step_w = self.im_width / l
            x_ofs, y_ofs = ofs
            x_ofs *= step_w
            y_ofs *= step_h
            level_anchors = np.mgrid[x_ofs:self.im_width + x_ofs:step_w, y_ofs:self.im_height + y_ofs:step_h]
            level_anchors = np.reshape(np.transpose(level_anchors, [1,2,0]),[-1,2])
            if as_boxes:
                box = np.asarray([step_w, step_h])
                boxes = np.tile(box, (level_anchors.shape[0], 1))
                level_anchors = np.concatenate([level_anchors, boxes], axis = 1)

            anchors.append(level_anchors )
        return np.concatenate(anchors)

    # return center points, optionally with box width and height
    def generate_anchor_points(self, offsets = 0, as_boxes = False):
        anchors = self.generate_anchors(offsets, as_boxes)
        return torch.tensor(anchors, dtype=torch.float32)

    # return boxes as topleft and bottomright corners
    def generate_anchor_boxes(self):
    # Get centered boxes
        anchors = self.generate_anchors(0.5, True)
    # Convert to box representation
        conversion_mat = np.array([
            [ 1, 0, 1, 0 ],
            [ 0, 1, 0, 1 ],
            [ -0.5, 0, 0.5, 0 ],
            [ 0, -0.5, 0, 0.5 ]
        ])
        boxes = anchors @ conversion_mat
    # Clip to image dimensions
        boxes = np.clip(boxes, 0, [ self.im_width, self.im_height, self.im_width, self.im_height])
        return torch.tensor(boxes, dtype=torch.float32)


import torchvision
def find_best_anchor_boxes(
            t_label : torch.Tensor,
            anchor_boxes: torch.Tensor,
            threshold: float) -> list[int]:

    iou_scores = torchvision.ops.box_iou(t_label, anchor_boxes)
    best_boxes = torch.nonzero(iou_scores > threshold, as_tuple = True)[1]
    if best_boxes.shape[0] == 0:
        best_boxes = torch.argmax(iou_scores).unsqueeze(0)

    return best_boxes
