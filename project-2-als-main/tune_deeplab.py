import json
import os
import random
import numpy as np
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
import torchvision.models.segmentation as segmentation
from torch.utils.data import DataLoader

from src.loss import DiceBCELoss
import src.dataset as dataset
from src.utils.calculate_stats import compute_dataset_stats
from src.utils.metrics import calculate_f1

random_seed = random.randint(0, 2**32 - 1)

CONFIG = {
    'epoch': 500,
    'learning_rate': 1e-4, 
    'batch_size': 4, 
    'image_size': 400,
    'architecture': "DeepLabV3-ResNet101",
    'dataset': "Full Dataset",
    'weight_decay': 1e-4,
    'seed': random_seed,
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

wandb.init(project="deeplab-finetune", config=CONFIG, name=run_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

mean, std = compute_dataset_stats(root_dir="./data/training/")

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

train_set = dataset.Dataset(root_dir='./data/training/', transform=train_transform)
val_set = dataset.Dataset(root_dir='./data/validation/', transform=val_transform)

train_loader = DataLoader(train_set, 
                          batch_size=CONFIG['batch_size'], 
                          shuffle=True, 
                          num_workers=4, 
                          pin_memory=True, 
                          drop_last=True)

val_loader = DataLoader(val_set, 
                        batch_size=CONFIG['batch_size'], 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True)

def get_deeplab_model(device):
    model = segmentation.deeplabv3_resnet101(weights=segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    return model.to(device)

model = get_deeplab_model(device)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

loss_fn = DiceBCELoss()

print(f"Starting DeepLab Training on {device}...")
best_val_f1 = 0.0

for epoch in range(CONFIG['epoch']):
    
    model.train()
    epoch_train_loss = 0.0
    
    for batch in train_loader:
        x = batch['image'].to(device).float()
        
        y = batch['mask'].to(device).float()
        if len(y.shape) == 3: 
            y = y.unsqueeze(1)
        y = y / 255.0

        optimizer.zero_grad()
        
        outputs = model(x)
        
        loss_main = loss_fn(outputs['out'], y)
        
        loss_aux = loss_fn(outputs['aux'], y)
        
        total_loss = loss_main + 0.4 * loss_aux
        
        
        epoch_train_loss += total_loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_f1 = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x = batch['image'].to(device).float()
            y = batch['mask'].to(device).float()
            if len(y.shape) == 3: 
                y = y.unsqueeze(1)
            y = y / 255.0
            
            outputs = model(x)
            pred_tensor = outputs['out']
            loss = loss_fn(pred_tensor, y)
            
            f1 = calculate_f1(pred_tensor, y)
            
            epoch_val_loss += loss.item()
            epoch_val_f1 += f1.item()
            
            if batch_idx == 0 and epoch % 5 == 0:
                visual_pred = (torch.sigmoid(pred_tensor[0]) > 0.5).float()
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
    
    print(f"Ep {epoch+1} | Loss: {avg_train_loss:.4f} | Val F1: {avg_val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_f1": avg_val_f1,
        "lr": optimizer.param_groups[0]['lr']
    })

    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        torch.save(model.state_dict(), f"checkpoints/{run_name}_best.pth")
        print(f"--> New Best Model Saved (F1: {best_val_f1:.4f})")

    torch.save(model.state_dict(), f"checkpoints/{run_name}_last.pth")

print("Training Finished.")
wandb.finish()