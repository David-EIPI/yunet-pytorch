import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path

import sys
mylib_path = "../"
sys.path.insert(0, mylib_path)

net_device = 'cuda'
data_image_size = 160
batch_size=64
anchors_level0 = data_image_size // 4

checkpoint_file_name = "yunet_v3_cp.pth"
bestmodel_file_name = "yunet_v3.pth"

lr_scheduler = "cycle" # other options "linear" and "plateau"

# If preload_data is set dataloader is iterated before the training loop
# and all data (images and labels) are cached in RAM.
# Otherwise dataloader is iterated normally during training loop.
preload_data = True

from yunet_model import YunetModel
from eiou_loss import eiou_loss
from utils import prompt_load_if_exists


import torchvision
from data.dataset import EasyFaceDataset, combined_datasets, od_collate_fn
from anchor_gen import AnchorGenerator
from utils import decode_boxes

# Precompute nearest anchors and append to the images and GT labels
class AnchoredDataset(EasyFaceDataset):
    boxes : torch.Tensor | None = None

    def __init__(self, annotations_path: Path, image_size: int):
        super().__init__(annotations_path, image_size)
        self.boxes = __class__.boxes

    def __getitem__(self, idx:int)->tuple:
        image, label, name, cls = super().__getitem__(idx)
        t_label = torch.as_tensor(label, dtype = torch.float32)

        iou_scores = torchvision.ops.box_iou(t_label, self.boxes)
        if cls > 0:
            best_boxes = torch.nonzero(iou_scores > 0.5, as_tuple = True)[1]
            if best_boxes.shape[0] == 0:
                best_boxes = torch.argmax(iou_scores).unsqueeze(0)
        else:
            best_boxes = torch.tensor([])

        return image, label, best_boxes, cls

anchor_gen = AnchorGenerator([anchors_level0, anchors_level0//2, anchors_level0//4])
AnchoredDataset.boxes = anchor_gen.generate_anchor_boxes()


# Load data from the Soloface dataset

datasets = combined_datasets(data_image_size, AnchoredDataset)
train_dataset = datasets["train"]
val_dataset = datasets["val"]
test_dataset = datasets["test"]
train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=od_collate_fn
)

val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=od_collate_fn
)


def safe_cat_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    # Concatenates tensors in a list (possibly empty).
    # Needed to avoid exception thrown from torch.cat for empty input lists.

    if tensors == []:
        return torch.tensor([])
    else:
        return torch.cat(tensors)

# Computes detection confidence loss and Extended IoU loss (from Peng et al., 2021)

class easyLoss(nn.Module):
    def __init__(self, anchors: torch.Tensor, device='cpu'):
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

# Create the loss function
criterion = easyLoss(AnchoredDataset.boxes)#, device=net_device)

# Create the model. Weights are random-initialized
net = YunetModel(1, tfpn_levels=3, device=net_device)


import torch.optim as optim
lr = [0.01]
optimizer = optim.SGD(net.parameters(), lr=lr[0], momentum=0.9, weight_decay=0.0005)

# Select learning rate scheduler
if lr_scheduler == "cycle":
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, lr[0]/50, lr[0]*2, 4000, mode='triangular')
elif lr_scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, min_lr = 1e-6, factor = 0.5 )
else:
    scheduler = optim.lr_scheduler.LinearLR(optimizer, 0.02, 1.0, 4000)


min_val_loss = 10000
max_idle_epochs=150
epochs_since_last_improvement=0
start_epoch = 0

if prompt_load_if_exists(checkpoint_file_name):
    checkpoint = torch.load(checkpoint_file_name, weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    epochs_since_last_improvement = checkpoint['epochs_since_last_improvement']
    loss = checkpoint['loss']
    min_val_loss = checkpoint['min_val_loss']
    net.train()


# Cache the data
data_cache = list()

if preload_data:
    count = 0
    for i, data in enumerate(train_dataloader):
        images, labels, idx, cls = data
        data_cache.insert(i, (images, labels, idx, cls))
        count += 1

    print(f'{count} batches loaded')


from datetime import timedelta
import time
dt1 = timedelta(seconds=time.monotonic())

loss_ratio = 2

if preload_data:
    iterable_dataset = data_cache
    n_avg = count // 2
else:
    iterable_dataset = train_dataloader
    n_avg = 100

for epoch in range(start_epoch, 2):
    dt_r = timedelta(seconds=time.monotonic())

    running_loss_e = 0
    running_loss_c = 0
    train_loss = 0
    net.train()

    for i, data in enumerate(iterable_dataset):
        images, labels, idx, cls = data
        images = images.to(net_device)
        optimizer.zero_grad()
        preds = net(images)
        loss_c, loss_e = criterion(preds, (images, labels, idx, cls))
        loss = (loss_ratio * loss_e + loss_c) / len(labels)
        loss.backward()
        optimizer.step()
        running_loss_e +=  loss_ratio * loss_e.item() / len(labels)
        running_loss_c += loss_c.item() / len(labels)
        train_loss += loss.item()
        scheduler.step()
        lr = scheduler.get_last_lr()
        if i % n_avg == 0: #n_avg-1:
            print(f'[{epoch + 1}, {i + 1:5d}] lr: {lr[0]:.5g} loss_e: {running_loss_e / n_avg:.3f} loss_c: {running_loss_c / n_avg:.3f}')
            running_loss_c = 0.0
            running_loss_e = 0.0

    val_e = 0
    val_c = 0
    nitems = 0
    net.eval()
    for i, data in enumerate(val_dataloader):
        images, labels, idx, cls = data
        images = images.to(net_device)
        preds = net(images)
        loss_c, loss_e = criterion(preds, (images, labels, idx, cls))
        loss = (loss_ratio * loss_e + loss_c) / len(labels)
        val_c += loss_c.item() / len(labels)
        val_e += loss_ratio * loss_e.item() / len(labels)
        nitems += 1

    val_loss = (val_c + val_e) / nitems

    epochs_since_last_improvement += 1
    if val_loss < min_val_loss:
        print(f'Saving model with l={val_loss:.5f}')
        min_val_loss = val_loss
        epochs_since_last_improvement = 0
        torch.save(net.state_dict(), bestmodel_file_name)

    dt2 = timedelta(seconds=time.monotonic())
    print(f"Epoch time:{(dt2 - dt_r)} Avg: {(dt2 - dt1)/(epoch+1)} Val: {val_e/nitems:.3f} {val_c/nitems:.3f} {val_loss:.3f}")
    torch.save({
            'epoch': epoch + 1,
            "epochs_since_last_improvement" : epochs_since_last_improvement,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            "min_val_loss": min_val_loss,
            }, checkpoint_file_name)

    with open("yunet_train_v3_epoch_log", "a") as f:
        f.write("{}\t{}\t{}\t{}\t{}\n".format(epoch+1, val_loss, val_c, val_e, train_loss / count))


    if epochs_since_last_improvement > max_idle_epochs:
        print(f'No improvements for {epochs_since_last_improvement} epochs.')
        break

