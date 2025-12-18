import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'groundtruth')
        
        # Smart file loading (ignores hidden files)
        all_files = sorted(os.listdir(self.images_dir))
        self.ids = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) and not f.startswith('.')]
        
        # Verify masks exist
        valid_ids = []
        for f in self.ids:
            if self._find_mask(f):
                valid_ids.append(f)
        self.ids = valid_ids

    def _find_mask(self, filename):
        # exact match
        if os.path.exists(os.path.join(self.masks_dir, filename)):
            return os.path.join(self.masks_dir, filename)
        # partial match (e.g. image.jpg -> image.png)
        base = os.path.splitext(filename)[0]
        for ext in ['.png', '.tif', '.jpg', '_mask.png']:
            cand = os.path.join(self.masks_dir, base + ext)
            if os.path.exists(cand): return cand
        return None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        filename = self.ids[idx]
        img_path = os.path.join(self.images_dir, filename)
        mask_path = self._find_mask(filename)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long().unsqueeze(0)

        if isinstance(mask, torch.Tensor):
            mask = mask.float()
            if mask.ndim == 2: mask = mask.unsqueeze(0)

        return {'image': image, 'mask': mask}
