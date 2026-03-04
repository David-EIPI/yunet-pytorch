"""
Load previously trained model, quantize it and run inference on the test set.
"""
import sys
import argparse
from typing import Iterable
import importlib
from pathlib import Path

from tqdm import tqdm

import torch
import numpy as np
import torchvision

net_device = 'cuda'

model_weights_file = "weights/yunet_v3.pth"
ESPDL_MODEL_PATH = "yunet_v3.espdl"

batch_size = 1

from yunet_model import YunetModel

model = YunetModel(1, tfpn_levels=3, device = net_device)
model.load_state_dict(torch.load(model_weights_file, map_location=net_device, weights_only=True))
net = model.eval()

# Name of the dataset module
using_dataset="data.mydataset.dataset"
#using_dataset="data.dataset"

# Load dataset and supporting callables
dataset_module = importlib.import_module(using_dataset)

combined_datasets = getattr(dataset_module, "combined_datasets")

data_image_size = 160
anchors_level0 = data_image_size // 4


from anchor_gen import AnchorGenerator, find_best_anchor_boxes
from eiou_loss import easyLoss
from utils import decode_boxes

anchor_gen = AnchorGenerator([anchors_level0,anchors_level0//2,anchors_level0//4])
anchor_boxes = anchor_gen.generate_anchor_boxes()

# Create the loss function
criterion = easyLoss(anchor_boxes, batch_size)

datasets = combined_datasets(data_image_size)
test_dataset = datasets["test"]


def test_collate_fn(batch: list) -> tuple:
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
    names = []
    for sample in batch:
        imgs.append(sample[0])

        t_label = torch.FloatTensor(sample[1])
        targets.append(t_label)

        names.append(sample[2])
        classes.append(sample[3])

        if 0 == sample[3]: # No detected objects
            best_boxes = torch.tensor([])
        else: # Objects are present - assign anchor boxes
            best_boxes = find_best_anchor_boxes(t_label, anchor_boxes, 0.5)

        indexes.append(best_boxes)

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets, indexes, classes, names


test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=test_collate_fn
)


from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor

INPUT_SHAPE = [3, data_image_size, data_image_size]

class BaseInferencer:
    def __init__(self, args, dataset):
        # get quantization config.
        self.num_of_bits = args.bits
        self.target = args.target
        # Load a pretrained mobilenet v2 model
        self.model = YunetModel(1, tfpn_levels=3, device = net_device)
        self.model.load_state_dict(torch.load(model_weights_file, map_location=net_device, weights_only=True))
        self.model = self.model.to(net_device)
        self.calibration_dataset = dataset

    def load_calibration_dataset(self) -> Iterable:
        return self.calibration_dataset

    def __call__(self):
        def collate_fn(batch: list) -> torch.Tensor:
            return batch[0].to(net_device)

        # create a setting for quantizing your network with ESPDL.
        quant_setting = QuantizationSettingFactory.espdl_setting()

        # Load training data for creating a calibration dataloader.
        calibration_dataset = self.load_calibration_dataset()
        calibration_dataloader = torch.utils.data.DataLoader(dataset=calibration_dataset, batch_size=batch_size, shuffle=False)

        # quantize your model.
        self.quant_ppq_graph = espdl_quantize_torch(
            model=self.model,
            espdl_export_file=ESPDL_MODEL_PATH,
            calib_dataloader=calibration_dataloader,
            calib_steps=8,
            input_shape=[1] + INPUT_SHAPE,
            target=self.target,
            num_of_bits=self.num_of_bits,
            collate_fn=collate_fn,
            setting=quant_setting,
            device=net_device,
            error_report=False,
            skip_export=False,
            export_test_values=True,
            verbose=1,
        )
        return self.quant_ppq_graph

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-t",
    "--target",
    type=str,
    default="esp32s3",
    help="esp32p4, esp32s3 or c, (defaults: esp32s3).",
)
parser.add_argument(
    "-b",
    "--bits",
    type=int,
    default=16,
    help="the number of bits, support 8 or 16, (defaults: 8).",
)
args = parser.parse_args()

inferencer = BaseInferencer(args, test_dataset)
quant_ppq_graph = inferencer()

print(f"""Quantized model exported: {ESPDL_MODEL_PATH}
    Target: {args.target}
    Resolution: {args.bits} bits\n""" )

loss_ratio = 2
pred_boxes_list = []
pred_conf_list = []
cls = []
labels = []
fnames = []
loss_a = 0
loss_count = 0
with torch.no_grad():
    executor = TorchExecutor(graph=quant_ppq_graph, device=net_device)
    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Test set"):
        images, labels_b, idx, cls_b, names = data
        images = images.to(net_device)
        preds = executor(images)

        loss_c, loss_e = criterion(preds, (images, labels_b, idx, cls_b))
        loss = (loss_ratio * loss_e + loss_c)
        loss_a += loss.item()
        loss_count += len(labels_b)
        pred_boxes_list.append(preds[1])
        pred_conf_list.append(preds[0])
        cls.extend(cls_b)
        fnames.extend(names)
        labels.extend(labels_b)

pred_boxes = torch.cat(pred_boxes_list)
pred_conf = torch.cat(pred_conf_list)
del(pred_boxes_list)
del(pred_conf_list)

print("loss = {}".format(loss_a / loss_count))

test_e = 0
test_c = 0
nitems = 0
thresh = 0.995
tp = 0
tn = 0
fp = 0
fn = 0
nsamp = pred_conf.shape[0]
print(f"{nsamp=}")

conf = torch.sigmoid(pred_conf)
conf0 = torch.reshape(conf, (-1,))
pos = torch.argwhere(conf0 > thresh).ravel()
pos_data = pos // pred_conf.shape[1]
pred_c = torch.zeros((1, len(cls)), dtype=torch.int)
pred_c[:,pos_data] = 1
cls_t = torch.tensor(cls)
tp += torch.sum(torch.logical_and(cls_t, pred_c))
fp += torch.sum(torch.logical_and(torch.logical_xor(cls_t, pred_c), pred_c ))
fn += torch.sum(torch.logical_and(torch.logical_xor(cls_t, pred_c), cls_t ))
tn += torch.sum(torch.logical_not(torch.logical_or(cls_t, pred_c)))

all_tn = tp + fp + fn + tn
accuracy = (tp + tn) / all_tn
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f"{all_tn=}\n{tp=}\n{fp=}\n{fn=}\n{tn=}\n{accuracy=}\n{precision=}\n{recall=}")


# Select samples with detected boxes
selected_boxes = pred_boxes[pos_data,:,:]
# Isolate confidence values for samples that have some confs > threshold
detection_confs = conf[pos_data,:,:].squeeze(-1)
# Find max confidence for each sample
max_confs = torch.argmax(detection_confs, 1)
detection_boxes = selected_boxes[torch.arange(selected_boxes.shape[0]), max_confs, :]
tiled_anchors = anchor_boxes.repeat([detection_boxes.shape[0], 1, 1]).to(device=net_device)
detection_anchors = tiled_anchors[torch.arange(tiled_anchors.shape[0], device=net_device), max_confs, :]
decoded_boxes = decode_boxes(detection_boxes, detection_anchors)
labels_t = torch.cat(labels).to(device=net_device)[pos_data]
with torch.no_grad():
    boxes_iou = torch.mean(torch.diagonal(torchvision.ops.box_iou(decoded_boxes, labels_t))).item()
    boxes_loss_g = torchvision.ops.generalized_box_iou_loss(decoded_boxes, labels_t, "mean").item()
    print(f"{boxes_iou=:.3f}\n{boxes_loss_g=:.3f}")

