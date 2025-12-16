import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Subset
import wandb
from src.loss import DiceBCELoss

import src.dataset as dataset
import random
import numpy as np

from src.utils.calculate_stats import compute_dataset_stats
from src.utils.metrics import calculate_f1

random_seed = random.randint(0, 2**32 - 1)

CONFIG = {
    'epoch': 1000,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'image_size': 400,
    'architecture': "Pretrained-SMP-ResNet34", 
    'dataset': "Full Dataset",
    'val_split': 0.2,
    'weight_decay' : 1e-4,
    'dropout': 0.4,
    'seed' : random_seed,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

run_name = f"{CONFIG['architecture']}_seed{CONFIG['seed']}"
os.makedirs("checkpoints", exist_ok=True)

with open(f"checkpoints/{run_name}_config.json", "w") as f:
    json.dump(CONFIG, f, indent=4)

wandb.init(
    project="unet",
    config=CONFIG,
    name=f"{CONFIG['architecture']}-FineTune"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
mean, std = compute_dataset_stats(root_dir='./data/training/')

train_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GridDistortion(p=0.1),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

train_dataset = dataset.Dataset(root_dir='./data/training/', transform=train_transform)
val_dataset   = dataset.Dataset(root_dir='./data/training/', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20)

loss_fn = DiceBCELoss()
wandb.watch(model, log="all", log_freq=100)

print(f"Starting Training on {device}...")
best_val_f1 = 0.0

for epoch in range(CONFIG['epoch']):
    
    #train
    model.train()
    epoch_train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
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

    avg_train_loss = epoch_train_loss / len(train_loader)
    

    # Start validation 
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_f1 = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x = batch['image'].to(device).float()
            y = batch['mask'].to(device).float()
            
            if len(y.shape) == 3: y = y.unsqueeze(1)
            y = y / 255.0
            
            preds = model(x)
            loss = loss_fn(preds, y)
            
            f1 = calculate_f1(preds, y)
            
            epoch_val_loss += loss.item()
            epoch_val_f1 += f1.item()
            
            if batch_idx == 0 and epoch % 5 == 0:
                visual_pred = (torch.sigmoid(preds[0]) > 0.5).float()
                wandb.log({
                    "Val Examples": [
                        wandb.Image(x[0], caption=f"Val Input Ep{epoch}"),
                        wandb.Image(y[0], caption=f"Val Truth Ep{epoch}"),
                        wandb.Image(visual_pred, caption=f"Val Pred Ep{epoch}")
                    ]
                })

    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_val_f1 = epoch_val_f1 / len(val_loader)
    
    scheduler.step(avg_val_f1)
    
    print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val F1: {avg_val_f1:.4f}")
    
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_f1": avg_val_f1
    })

    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        torch.save(model.state_dict(), f"checkpoints/{run_name}_best.pth")
        print(f"New Best Model Saved (F1: {best_val_f1:.4f})")

    torch.save(model.state_dict(), f"checkpoints/{run_name}_last.pth")

print("Training Finished.")
wandb.finish()