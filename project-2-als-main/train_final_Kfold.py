import os
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Imports
from src.loss import DiceBCELoss
import src.dataset as dataset
import src.models.unet as unet
from src.utils.metrics import calculate_f1

# --- CONFIG FOR 0.95 SCORE ---
CONFIG = {
    'epoch': 55,             # 55 epochs to ensure full convergence
    'learning_rate': 3e-4,   # Optimal starting LR
    'batch_size': 16,        
    'image_size': 400,
    'n_folds': 5,            # Ensemble of 5 models is key
    'seed': 42,
    'num_workers': 0 if os.name == 'nt' else 4
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_transforms():
    train_transform = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        # Distortions
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1)
        ], p=0.3),
        # Color
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1),
        ], p=0.4),
        # Cutout (Forces context learning)
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, val_transform

def train_one_fold(fold_idx, train_idx, val_idx, device):
    print(f"\n--- Fold {fold_idx+1}/{CONFIG['n_folds']} ---")
    train_tf, val_tf = get_transforms()
    
    ds = dataset.Dataset(root_dir='./data/training/', transform=None)
    
    train_dataset = dataset.Dataset(root_dir='./data/training/', transform=train_tf)
    train_dataset.ids = [ds.ids[i] for i in train_idx]
    
    val_dataset = dataset.Dataset(root_dir='./data/training/', transform=val_tf)
    val_dataset.ids = [ds.ids[i] for i in val_idx]

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=True)

    encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model = unet.UNet(encoder=encoder, width=CONFIG['image_size'], height=CONFIG['image_size'], 
                      initial_channels=3, dropout=0.1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-3)
    # Cosine Scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    loss_fn = DiceBCELoss(alpha=0.5, beta=0.5) # Balanced loss
    
    scaler = GradScaler()
    best_f1 = 0.0
    
    for epoch in range(CONFIG['epoch']):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for batch in loop:
            x = batch['image'].to(device).float()
            y = batch['mask'].to(device).float()
            if len(y.shape) == 3: y = y.unsqueeze(1)
            y = y / 255.0

            optimizer.zero_grad()
            with autocast():
                preds = model(x)
                loss = loss_fn(preds, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_f1 = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['image'].to(device).float()
                y = batch['mask'].to(device).float()
                if len(y.shape) == 3: y = y.unsqueeze(1)
                y = y / 255.0
                with autocast():
                    preds = model(x)
                val_f1 += calculate_f1(preds, y).item()

        avg_val_f1 = val_f1 / len(val_loader)
        scheduler.step(epoch + avg_val_f1)

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            torch.save(model.state_dict(), f"checkpoints/fold_{fold_idx}_best_resnet34.pth")
            print(f"   Epoch {epoch+1} | New Best F1: {best_f1:.4f}")

def main():
    set_seed(CONFIG['seed'])
    os.makedirs("checkpoints", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get all indices first
    full_ds = dataset.Dataset(root_dir='./data/training/', transform=None)
    all_indices = np.arange(len(full_ds))
    print(f"Training on {len(all_indices)} images using {device}")
    
    kfold = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
        train_one_fold(fold_idx, train_idx, val_idx, device)

if __name__ == "__main__":
    main()