# @plawanrath

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
