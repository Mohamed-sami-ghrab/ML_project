import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.models import ResNet
import src.dataset as dataset
import torchvision.utils as vutils

train_transform = A.Compose([
    A.Resize(400, 400),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.5),

    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(400, 400),
    A.Normalize(),
    ToTensorV2(),
])

import src.models.unet as unet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

sanity_transform = A.Compose([
    A.Resize(400, 400),
    ToTensorV2(),
])

sanity_ds = dataset.Dataset(root_dir='./data/training/', transform=sanity_transform)
loader = DataLoader(sanity_ds, batch_size=1, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch = next(iter(loader))
x = batch['image'].to(device).float() / 255.0
y = batch['mask'].to(device).unsqueeze(1).float()

y = (y / 255.0).float()

encoder = models.resnet18(pretrained=True)

model = unet.UNet(encoder=encoder, width=400, height=400, initial_channels=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

loss_fn = nn.BCEWithLogitsLoss()

print("Starting Sanity Check...")
for epoch in range(100):
    model.train()
    
    preds = model(x)
    loss = loss_fn(preds, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("Saving results...")

probs = torch.sigmoid(preds)
pred_mask = (probs > 0.5).float()

vutils.save_image(x, "sanity_check_input.png")
vutils.save_image(y, "sanity_check_ground_truth.png")
vutils.save_image(pred_mask, "sanity_check_prediction.png")

print("Saved: sanity_check_input.png, sanity_check_ground_truth.png, sanity_check_prediction.png")
