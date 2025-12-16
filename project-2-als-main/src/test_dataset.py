import torch
import os
import cv2 

class ValDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for loading images and their corresponding masks.
    Args:
        root_dir (str): Root directory containing 'images' and 'groundtruth' subdirectories.
        transform (callable, optional): Optional transform to be applied on a sample (expects a dict with 'image' and 'mask').
    Attributes:
        images_dir (str): Path to the directory containing images.
        masks_dir (str): Path to the directory containing masks.
        image_filenames (List[str]): Sorted list of image filenames.
        mask_filenames (List[str]): Sorted list of mask filenames.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Loads and returns the image and mask at the specified index, applying transforms if provided.
    Raises:
        FileNotFoundError: If an image or mask file is not found at the expected path.
    """

    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'groundtruth')

        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.mask_filenames = sorted(os.listdir(self.masks_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):

        image_name = self.image_filenames[index]
        mask_name = self.mask_filenames[index]

        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")

        return {
            'image' : image, 
            'mask' : mask
            }
    
