import json
import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # Progress bar

# Import your custom modules
from src.loss import DiceBCELoss
import src.dataset as dataset
import src.models.unet as unet
from src.utils.calculate_stats import compute_dataset_stats
from src.utils.metrics import calculate_f1

# --- CONFIGURATION ---
CONFIG = {
    'epoch': 100,  # 100 is usually enough if converging well
    'learning_rate': 1e-4,
    'batch_size': 16, # Reduced slightly to prevent GPU OOM on smaller cards
    'image_size': 400,
    'architecture': "UNet-ResNet34",
    'val_split': 0.2,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'seed': 42,
    'num_workers': 0 if os.name == 'nt' else 4 # 0 for Windows, 4 for Linux
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train():
    set_seed(CONFIG['seed'])
    
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Training on Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 2. Prepare Directories
    run_name = f"{CONFIG['architecture']}_local_run"
    os.makedirs("checkpoints", exist_ok=True)
    
    # 3. Data Loading & Augmentation
    print("⏳ Calculating dataset stats...")
    # Assumes data is in ./data/training/images and ./data/training/groundtruth
    try:
        mean, std = compute_dataset_stats(root_dir='./data/training/')
    except:
        print("   Warning: Could not calc stats, using ImageNet defaults.")
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_transform = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Load full dataset and split
    full_dataset = dataset.Dataset(root_dir='./data/training/', transform=None)
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    
    raw_train_set, raw_val_set = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms using wrapper classes or by setting them if Dataset supports it
    # Since existing Dataset class applies transform in __getitem__, we need to handle the split carefully.
    # A cleaner way given your existing Dataset class structure:
    train_set = dataset.Dataset(root_dir='./data/training/', transform=train_transform)
    val_set = dataset.Dataset(root_dir='./data/training/', transform=val_transform)
    
    # Use indices from random_split to create Subsets
    train_loader = DataLoader(torch.utils.data.Subset(train_set, raw_train_set.indices), 
                              batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(val_set, raw_val_set.indices), 
                            batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=True)

    print(f"📊 Data Loaded: {len(train_loader.dataset)} Training, {len(val_loader.dataset)} Validation")

    # 4. Model Setup
    encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model = unet.UNet(encoder=encoder, 
                      width=CONFIG['image_size'], 
                      height=CONFIG['image_size'], 
                      initial_channels=3, 
                      dropout=CONFIG['dropout']).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # Weighted Loss: 80% Dice (F1), 20% BCE (Pixel Accuracy)
    loss_fn = DiceBCELoss(alpha=0.8, beta=0.2) 

    # 5. Training Loop
    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    print("🚀 Starting Training...")
    
    for epoch in range(CONFIG['epoch']):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0.0
        
        # Training
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epoch']} [Train]", leave=False)
        for batch in loop:
            x = batch['image'].to(device).float()
            y = batch['mask'].to(device).float()
            
            if len(y.shape) == 3:
                y = y.unsqueeze(1)
            y = y / 255.0

            preds = model(x)
            loss = loss_fn(preds, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_f1 = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['image'].to(device).float()
                y = batch['mask'].to(device).float()
                
                if len(y.shape) == 3:
                    y = y.unsqueeze(1)
                y = y / 255.0
                
                preds = model(x)
                loss = loss_fn(preds, y)
                f1 = calculate_f1(preds, y)
                
                epoch_val_loss += loss.item()
                epoch_val_f1 += f1.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_f1 = epoch_val_f1 / len(val_loader)
        
        scheduler.step(avg_val_f1)
        
        # Logging
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(avg_val_f1)
        
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} | T: {epoch_duration:.0f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {avg_val_f1:.4f}")

        # Save Best Model
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), f"checkpoints/{run_name}_best.pth")
            print(f"   🏆 New Best F1: {best_val_f1:.4f} -> Saved!")

    # Save final model
    torch.save(model.state_dict(), f"checkpoints/{run_name}_last.pth")
    print(f"\nTraining Finished. Best F1: {best_val_f1:.4f}")
    print(f"Model saved to checkpoints/{run_name}_best.pth")

if __name__ == "__main__":
    train()