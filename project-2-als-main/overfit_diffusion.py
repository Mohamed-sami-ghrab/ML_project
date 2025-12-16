import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import src.dataset as dataset
import src.models.diffusion_unet as diffunet

TIMESTEPS = 250   
START_BETA = 0.0001
END_BETA = 0.02
LR = 1e-4
EPOCHS = 1000      
IMAGE_SIZE = 64   
BATCH_SIZE = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

sanity_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    ToTensorV2(),
])

sanity_ds = dataset.Dataset(root_dir='./data/training/', transform=sanity_transform)
loader = DataLoader(sanity_ds, batch_size=BATCH_SIZE, shuffle=True)

batch = next(iter(loader))

x_clean = (batch['image'].to(device).float() / 255.0) * 2.0 - 1.0
raw_mask = batch['mask'].to(device).float()

if len(raw_mask.shape) == 3:
    raw_mask = raw_mask.unsqueeze(1)

y_clean = (raw_mask / 255.0) * 2.0 - 1.0

encoder = models.resnet50(pretrained=True)
time_embedding_dim = 256

model = diffunet.DiffusionUNet(
    encoder=encoder, 
    width=IMAGE_SIZE, 
    height=IMAGE_SIZE, 
    initial_channels=4, 
    time_embedder=diffunet.SinusoidalPositionEmbeddings(dim=time_embedding_dim)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss() 

betas = torch.linspace(START_BETA, END_BETA, TIMESTEPS, device=device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

def get_noisy_sample(t, x0, noise):
    sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    return sqrt_alpha * x0 + sqrt_one_minus * noise 

print(f"Starting Diffusion Sanity Check on {device}...")

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    t = torch.randint(0, TIMESTEPS, (x_clean.shape[0],), device=device).long()
    t = epoch % TIMESTEPS
    t = torch.tensor([t], device=device).long()
    noise = torch.randn_like(y_clean)
    noisy_mask = get_noisy_sample(t, y_clean, noise)
    
    model_input = torch.cat((noisy_mask, x_clean), dim=1)
    
    pred_noise = model(noisy_mask, x_clean, t)
    
    loss = criterion(pred_noise, noise)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("Training finished. Starting Generation (Sampling)...")

def denoising(model, x, timesteps, device, start_beta, end_beta):
    model.eval()
    with torch.no_grad():
        betas = torch.linspace(start_beta, end_beta, timesteps, device=device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        one_minus_alpha = 1 - alphas
        sqrt_alphas = torch.sqrt(alphas)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        
        new_mask = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=device)
        x_next = torch.cat((new_mask, x), dim=1)

        def update_x(current_noisy_mask, t_index, predicted_noise):
            sqrt_alpha_t = sqrt_alphas[t_index].view(1, 1, 1, 1)
            one_minus_alpha_t = one_minus_alpha[t_index].view(1, 1, 1, 1)
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
            
            x_next = torch.cat((new_mask, x_next[:, 1:, :, :]), dim=1)

        return x_next



x_save = (x_clean + 1) / 2
y_save = (y_clean + 1) / 2
# gen_save = (generated_mask/3 + 1) / 2

# gen_save = torch.clamp(gen_save, 0, 1)

# binary_mask = (gen_save > 0.5).float()

# 1. Initialize an accumulator with zeros
# Shape matches the mask output: [Batch, 1, H, W]
mask_accumulator = torch.zeros((x_clean.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
NUM_RUNS = 3

print(f"Generating {NUM_RUNS} samples for averaging...")

for i in range(NUM_RUNS):
    # Run the denoising process
    generated_output = denoising(model, x_clean, TIMESTEPS, device, START_BETA, END_BETA)
    
    # Extract only the mask channel (Channel 0)
    current_mask = generated_output[:, 0:1, :, :]
    
    # Accumulate the result
    mask_accumulator += current_mask
    print(f"Sample {i+1}/{NUM_RUNS} done.")

# 2. Average the results
averaged_raw_mask = mask_accumulator / NUM_RUNS

# 3. Post-process (Un-normalize from [-1, 1] to [0, 1])
# Note: We average first, then normalize, which is mathematically safer
gen_save = (averaged_raw_mask + 1) / 2

# 4. Clamp and Threshold
gen_save = torch.clamp(gen_save, 0, 1)
binary_mask = (gen_save > 0.75).float()

# 5. Save
vutils.save_image(x_save, "diff_sanity_input.png")
vutils.save_image(y_save, "diff_sanity_ground_truth.png")
vutils.save_image(gen_save, "diff_sanity_generated_soft.png")
vutils.save_image(binary_mask, "diff_sanity_generated_binary.png")

print("Saved images with averaged predictions.")
