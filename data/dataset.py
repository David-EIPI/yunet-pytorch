import sys
from pathlib import Path
import json
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

base_directory = "soloface-detection-dataset"

# Path is the base path to the dataset.
# Inside it should have images and labels directories.
# Labels are in json format.
# Boxes are given with normalized top-left and bottom-right coordinates.

class EasyFaceDataset(torch.utils.data.Dataset):


    def __init__(self, annotations_path: Path, image_size: int):
    # Load labels as dictionary { image_file : label in YOLO format }
        self.image_path = annotations_path / "images"
        label_path = annotations_path / "labels"

    # Verify that corresponding image files exist and not empty
        json_annotations = [x for _,_,f in label_path.walk() for x in f if (self.image_path /(Path(x).stem + ".jpg")).stat().st_size>0]

    # Sorted keys = file names
        self.sorted_images = []
    # Labels (class + box positions)
        self.sorted_labels = []

        for label_file in json_annotations:
            with (label_path / label_file).open("r") as f:
                data = json.load(f)
                bbox = data["bbox"]
                bx = [ float(x) for x in bbox ]
                cls = int(data["class"])
                img = Path(label_file).stem + ".jpg"
                self.sorted_images.append(img)
                self.sorted_labels.append( [ cls, bx[0], bx[1], bx[2], bx[3] ] )


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((image_size, image_size))
        ])
        self.image_size = image_size

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.image_path / self.sorted_images[idx]
        img = self.load_image(img_path)
        label = self.sorted_labels[idx]
        cls = label[0]
        padded_out = self.resize_pad(img, label[1:])
        img = padded_out['image']
        label = padded_out['label']

        return self.transform(img.copy()), np.clip(label, 0, 1), self.sorted_images[idx], cls

    def __len__(self) -> int:
        return len(self.sorted_labels)


    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = cv.imread(path)
        if img.shape[2] < 3:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        return img

    @staticmethod
    def resize_pad(img: np.ndarray, label: list[float]) -> dict:

        h, w = img.shape[:2]

    # Compute symmetric padding to make square
        pad_top = pad_bottom = pad_left = pad_right = 0
        new_w = w
        new_h = h
        if h < w:
            d = w - h
            pad_top = d // 2
            pad_bottom = d - pad_top
            new_h = w
        elif w < h:
            d = h - w
            pad_left = d // 2
            pad_right = d - pad_left
            new_w = h

    # Choose border color: 128 for gray border
        if img.ndim == 2:
            border_val = 128
        else:
            border_val = tuple([128] * img.shape[2])

        padded_image = cv.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv.BORDER_CONSTANT, value=border_val
        )

    # Convert original label to np attau
        bbox = np.asarray(label, dtype=np.float32)

        scale_orig = np.array([w, h, w, h], dtype=np.float32)
        padding    = np.array([pad_left, pad_top, pad_left, pad_top], dtype=np.float32)
        max_orig   = scale_orig  # [w, h, w, h]
        min_orig   = np.zeros(4, dtype=np.float32)

    # Convert to original pixel coords, clip, then shift by padding
        bbox_px = np.clip(bbox * scale_orig, min_orig, max_orig) + padding

    # New (padded) dims
        scale_new = np.array([new_w, new_h, new_w, new_h], dtype=np.float32)

    # Convert back to normalized TLBR in the padded (square) image
        new_label = np.clip(bbox_px / scale_new, 0.0, 1.0)

        new_label = np.atleast_2d(new_label)
        return { "image" : padded_image, "label" : new_label }

    @staticmethod
    def convert_labels(yolo_box: np.ndarray) -> np.ndarray:
# Convert YOLO format to box format (also normalized)
        target = yolo_box.copy()
        target[:, 0:2] -= yolo_box[:, 2:4]/2
        target[:, 2:4] = yolo_box[:, 0:2] + yolo_box[:, 2:4] / 2
        return target



def od_collate_fn(batch: list) -> tuple:
    """Collate function for object detection.

    Args:
        batch (list): List of tuples containing images and targets.

    Returns:
        tuple: Batch of images and targets.
    """
    imgs = []
    targets = []
    indexes = []
    classes = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        indexes.append(sample[2])
        classes.append(sample[3])

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets, indexes, classes

def combined_datasets(size: int, dataset_class: type[EasyFaceDataset] = EasyFaceDataset)->dict:

    here = Path(__file__).parent

    train_dataset = dataset_class(Path(here) / base_directory / "train", size)
    val_dataset = dataset_class(Path(here) / base_directory / "val", size)
    test_dataset = dataset_class(Path(here) / base_directory / "test", size)

    return { "train" : train_dataset, "val" : val_dataset, "test": test_dataset }



# ──────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":

    datasets = combined_datasets(256)

    train_dataset = datasets["train"]
    test_dataset = datasets["test"]


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=od_collate_fn
    )

    dataiter = iter(train_dataloader)
    images, labels, idx, cls = next(dataiter)
    print(">", labels)

