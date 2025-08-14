# @plawanrath

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO


class CocoDetectionDataset(Dataset):
    """
    Returns image (PIL or transformed Tensor) and a target dict with:
      boxes: FloatTensor [N,4] in xyxy (absolute, original image scale)
      labels: LongTensor [N] (raw COCO category_id values)
      area: FloatTensor [N]
      image_id: LongTensor [1]
      iscrowd: LongTensor [N]
    """
    def __init__(self, root: str, ann_file: str, transform=None):
        self.root = Path(root)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform  # keep None; we resize/normalize later in the training loop

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        coco = self.coco
        img_id = int(self.ids[index])

        # load image info
        img_info = coco.loadImgs(img_id)[0]        # FIX: take [0]
        file_name = img_info["file_name"]
        width, height = img_info["width"], img_info["height"]

        # load image
        img = Image.open(self.root / file_name).convert("RGB")

        # load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO xywh
            # convert to xyxy
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            # clip to image bounds (defensive)
            xmin = max(0.0, min(xmin, width - 1.0))
            ymin = max(0.0, min(ymin, height - 1.0))
            xmax = max(0.0, min(xmax, width - 1.0))
            ymax = max(0.0, min(ymax, height - 1.0))
            # skip degenerate boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        # Optional *visual-only* transforms (don’t change boxes here)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
    

class UnlabeledImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading unlabeled images for MAE pre-training.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Fetches an image by index, applies transformations, and returns it.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Warning: Skipping corrupted image file: {img_path} ({e})")
            # Return a dummy tensor of the correct size if an image is corrupted
            # This prevents the training from crashing.
            return (torch.zeros(3, 224, 224),)

        if self.transform:
            image = self.transform(image)

        return (image,)
