"""
   Run inference with the Yunet-pytorch, print detected face bounding box and save image with the box drawn.
"""
import torch
import numpy as np
import cv2 as cv
from torchvision import transforms
import torchvision
from pathlib import Path
import sys

from datetime import timedelta
import time

from yunet_model import YunetModel
from anchor_gen import AnchorGenerator
from utils import decode_boxes

net_device = 'cpu'
model_weights_file = "weights/yunet_v3.pth"

model = YunetModel(1, tfpn_levels=3, device = net_device)
model.load_state_dict(torch.load(model_weights_file, map_location=net_device, weights_only=True))
net = model.eval()


data_image_size = 160
anchors_level0 = data_image_size // 4

anchor_gen = AnchorGenerator([anchors_level0,anchors_level0//2,anchors_level0//4])
anchors = anchor_gen.generate_anchor_boxes().to(net_device)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((data_image_size, data_image_size))
        ])


def run_prediction(path: str)->dict:
    img = cv.imread(path)
    if img.shape[2] < 3:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    img_tensor = transform(img.copy()).to(net_device)

    img_tensor.unsqueeze_(0)

    dt1 = timedelta(seconds=time.monotonic())

    with torch.no_grad():
        preds = net(img_tensor)

    dt2 = timedelta(seconds=time.monotonic())

    decoded = decode_boxes(preds[1], anchors)
    confs = torch.sigmoid(preds[0])
    conf0 = torch.reshape(confs, (-1,))
    sorted_confs = torch.argsort(conf0, descending=True)
    top = sorted_confs[0]

    if conf0[top] > 0.9:
        box = torch.flatten(decoded, 0, -2)[top]
        imgsize = np.repeat(img.shape[0:2][::-1], 2)
        box_np = box.cpu().numpy() * imgsize
        box_np = np.asarray(box_np, np.int32)
        cv.rectangle(img,(box_np[0], box_np[1]),(box_np[2], box_np[3]),(0,255,0),3)
        return { "conf" : conf0[top].item(), "box" : box_np, "image" : img, "time" : (dt2-dt1) }
    else:
        return dict()


for path in sys.argv[1:]:

    print("*** {}".format(Path(path).name))
    results = run_prediction(path)

    if results == {}:
        print("Not detected.")
    else:
        outpath = "box_" + Path(path).name
        print("Confidence: {:.3f}".format(results["conf"]))
        print("Box: " + str(results["box"]))
        cv.imwrite(outpath, results["image"])
        print("Inference time: {}\n".format(results["time"]))

