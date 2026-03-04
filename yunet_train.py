import torch
from collections import OrderedDict

from pathlib import Path
from tqdm import tqdm

import sys

net_device = 'cuda'
data_image_size = 160
batch_size=128
anchors_level0 = data_image_size // 4

checkpoint_file_name = "yunet_v3_cp.pth"
bestmodel_file_name = "yunet_v3.pth"

lr_scheduler = "cycle" # other options "linear" and "plateau"

# If preload_data is set dataloader is iterated before the training loop
# and all data (images and labels) are cached in RAM.
# Otherwise dataloader is iterated normally during training loop.
preload_data = True

import torchvision
from data.dataset import combined_datasets
from anchor_gen import AnchorGenerator, find_best_anchor_boxes

from yunet_model import YunetModel
from eiou_loss import easyLoss
from utils import prompt_load_if_exists



anchor_gen = AnchorGenerator([anchors_level0, anchors_level0//2, anchors_level0//4])
anchor_boxes = anchor_gen.generate_anchor_boxes()

def training_collate_fn(batch: list) -> tuple:
    """ Collate function for object detection.

    Args:
        batch (list): List of tuples containing images and targets.

    Function:
        Appends nearest anchors found for each image. Anchors are returned as indexes.

    Returns:
        tuple: Batch of images and targets.
    """
    imgs = []
    targets = []
    indexes = []
    classes = []
    for sample in batch:
        imgs.append(sample[0])

        t_label = torch.FloatTensor(sample[1])
        targets.append(t_label)

        classes.append(sample[3])

        if 0 == sample[3]: # No detected objects
            best_boxes = torch.tensor([])
        else: # Objects are present - assign anchor boxes
            best_boxes = find_best_anchor_boxes(t_label, anchor_boxes, 0.5)

        indexes.append(best_boxes)

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets, indexes, classes

# Load data from the Soloface dataset

datasets = combined_datasets(data_image_size)
train_dataset = datasets["train"]
val_dataset = datasets["val"]
test_dataset = datasets["test"]
train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=training_collate_fn
)

val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=training_collate_fn
)


# Create the loss function
criterion = easyLoss(anchor_boxes, batch_size)#, device=net_device)

# Create the model. Weights are random-initialized
net = YunetModel(1, tfpn_levels=3, device=net_device)

# Setup optimizer
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


# Cache training data in RAM
data_cache = list()

if preload_data:
    count = 0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Preloading data"):
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

