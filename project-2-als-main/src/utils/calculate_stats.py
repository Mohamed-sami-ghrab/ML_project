import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import src.dataset as dataset 

def compute_dataset_stats(root_dir, image_size=400, batch_size=16, num_workers=4):
    """
    Computes the mean and standard deviation of the dataset at the given path.
    
    Args:
        root_dir (str): Path to the dataset root (e.g., './data/training/').
        image_size (int): Size to resize images to (matches your training config).
        batch_size (int): Batch size for loading.
        num_workers (int): Number of worker threads for loading.
        
    Returns:
        tuple: (mean, std) where each is a tuple of 3 floats (R, G, B).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing stats for dataset at: {root_dir}")

    stats_transform = A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])

    ds = dataset.Dataset(root_dir=root_dir, transform=stats_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channels_sum = torch.zeros(3).to(device)
    channels_squared_sum = torch.zeros(3).to(device)
    num_batches = 0
    
    print("Iterating through dataset...")
    for batch in tqdm(loader):
        data = batch['image'].to(device).float() / 255.0 
        
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    mean = tuple(mean.cpu().tolist())
    std = tuple(std.cpu().tolist())
    
    print(f"\nRESULTS for {root_dir}:")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print("-" * 30)
    print(f"Config Copy-Paste:")
    print(f"'dataset_mean': ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}),")
    print(f"'dataset_std':  ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}),")
    
    return mean, std

if __name__ == "__main__":
    mean, std = compute_dataset_stats('./data/training/')