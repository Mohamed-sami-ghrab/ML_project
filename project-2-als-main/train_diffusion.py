import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import wandb
import src.dataset as dataset
import src.models.diffusion_unet as diffunet

CONFIG = {
    "TIMESTEPS": 500,
    "START_BETA": 0.0001,
    "END_BETA": 0.02,
    "LR": 1e-4,
    "EPOCHS": 20000,           
    "IMAGE_SIZE": 400,        
    "BATCH_SIZE": 8,        
    "NUM_WORKERS": 4,        
    "VAL_INTERVAL": 10,      
    "SAVE_INTERVAL": 50,     
    "NUM_GEN_RUNS": 3        
}

wandb.init(project="segdiff-satellite", config=CONFIG)
config = wandb.config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

train_transform = A.Compose([
    # A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
    ToTensorV2(),
])

val_transform = A.Compose([
    # A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    ToTensorV2(),
])

full_dataset = dataset.Dataset(root_dir='./data/training/', transform=train_transform)

train_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

val_loader = DataLoader(full_dataset, batch_size=4, shuffle=True) # visualize 4 images
fixed_val_batch = next(iter(val_loader)) 

encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
time_embedding_dim = 256

model = diffunet.DiffusionUNet(
    encoder=encoder, 
    width=config.IMAGE_SIZE, 
    height=config.IMAGE_SIZE, 
    initial_channels=4, 
    time_embedder=diffunet.SinusoidalPositionEmbeddings(dim=time_embedding_dim)
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-2)
criterion = nn.MSELoss() 

betas = torch.linspace(config.START_BETA, config.END_BETA, config.TIMESTEPS, device=device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

def get_noisy_sample(t, x0, noise):
    sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha * x0 + sqrt_one_minus * noise 

def denoising(model, x, timesteps, device):
    model.eval()
    with torch.no_grad():
        
        new_mask = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=device)
        
        def update_x(current_noisy_mask, t_index, predicted_noise):
            sqrt_alpha_t = torch.sqrt(alphas[t_index]).view(1, 1, 1, 1)
            one_minus_alpha_t = (1 - alphas[t_index]).view(1, 1, 1, 1)
            sqrt_one_minus_alphas_c_t = sqrt_one_minus_alphas_cumprod[t_index].view(1, 1, 1, 1)
            
            alphas_cp_t = alphas_cumprod[t_index].view(1, 1, 1, 1)
            if t_index >= 1:
                alphas_cp_t_minus_one = alphas_cumprod[t_index - 1].view(1, 1, 1, 1)
            else:
                alphas_cp_t_minus_one = torch.tensor(1.0, device=device).view(1, 1, 1, 1)

            if t_index > 0:
                posterior_std = torch.sqrt(betas[t_index] * (1 - alphas_cp_t_minus_one) / (1 - alphas_cp_t))
                noise = torch.randn_like(current_noisy_mask)
            else:
                posterior_std = 0
                noise = 0
            
            mean_pred = (current_noisy_mask - predicted_noise * one_minus_alpha_t / sqrt_one_minus_alphas_c_t) / sqrt_alpha_t
            return mean_pred + posterior_std * noise

        for time in range(timesteps - 1, -1, -1):
            t_batch = torch.full((x.shape[0],), time, device=device, dtype=torch.long)
            
            pred = model(new_mask, x, t_batch)
            new_mask = update_x(new_mask, time, pred)

        return new_mask

print("Starting Training Loop...")

for epoch in range(config.EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        x_train = (batch_data['image'].to(device).float() / 255.0) * 2.0 - 1.0
        
        raw_mask = batch_data['mask'].to(device).float()
        if len(raw_mask.shape) == 3:
            raw_mask = raw_mask.unsqueeze(1)
        y_train = (raw_mask / 255.0) * 2.0 - 1.0

        t = torch.randint(0, config.TIMESTEPS, (x_train.shape[0],), device=device).long()
        
        noise = torch.randn_like(y_train)
        noisy_mask = get_noisy_sample(t, y_train, noise)
        
        pred_noise = model(noisy_mask, x_train, t)
        
        loss = criterion(pred_noise, noise)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        wandb.log({"batch_loss": loss.item()})

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} | Loss: {avg_epoch_loss:.6f}")
    wandb.log({"epoch": epoch, "loss": avg_epoch_loss})

    if epoch % config.VAL_INTERVAL == 0:
        print(f"Running validation sampling for Epoch {epoch}...")
        
        val_img = (fixed_val_batch['image'].to(device).float() / 255.0) * 2.0 - 1.0
        val_mask_gt = fixed_val_batch['mask'].to(device).float().unsqueeze(1) / 255.0 # Keep [0,1] for display
        
        mask_accumulator = torch.zeros((val_img.shape[0], 1, config.IMAGE_SIZE, config.IMAGE_SIZE), device=device)
        
        for r in range(config.NUM_GEN_RUNS):
            gen_out = denoising(model, val_img, config.TIMESTEPS, device)
            mask_accumulator += gen_out
            
        averaged_mask = mask_accumulator / config.NUM_GEN_RUNS
        
        pred_soft = (averaged_mask + 1) / 2
        pred_soft = torch.clamp(pred_soft, 0, 1)
        pred_binary = (pred_soft > 0.75).float()
        
        val_img_disp = (val_img + 1) / 2 
        
        wandb.log({
            "val_input": [wandb.Image(i) for i in val_img_disp],
            "val_gt": [wandb.Image(m) for m in val_mask_gt],
            "val_pred_soft": [wandb.Image(p) for p in pred_soft],
            "val_pred_binary": [wandb.Image(b) for b in pred_binary]
        })

    if epoch % config.SAVE_INTERVAL == 0:
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path) 

print("Training Complete.")
wandb.finish()