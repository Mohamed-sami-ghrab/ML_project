import torch
import models.diffusion_unet as diffunet
import src.dataset as ds
from torch.utils.data import DataLoader

def train_diffusion_UNet(
    start_beta, 
    end_beta,
    timesteps,
    encoder,
    width,
    height,
    time_embedding_dim,
    transforms,
    root_dir,
    batch_size,
    optimizer,
    criterion,
    n_epoch
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    betas = torch.linspace(start_beta, end_beta, timesteps, device=device)

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    time_embedder = diffunet.SinusoidalPositionEmbeddings(dim=time_embedding_dim)

    dataset = ds.Dataset(
    root_dir=root_dir,
    transform=transforms
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = diffunet.DiffusionUNet(encoder=encoder, 
                                    width=width,
                                    height=height,
                                    initial_channels=4,
                                    time_embedder=time_embedder)

    model.to(device)

    def get_noisy_sample(t, x0, noise):

        sqrt_alphas = sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        one_min_sqrt = sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        x_t =  sqrt_alphas * x0 + one_min_sqrt * noise 
        return x_t

    model.train()
    for epoch in range(n_epoch):

        epoch_loss = []

        for batch_idx, batch_data in enumerate(loader):
            x_train = (batch_data['image'].to(device).float() / 255.0) * 2.0 - 1.0
            y_train = (batch_data['mask'].to(device).float() / 255.0) * 2.0 - 1.0

            t = torch.randint(0, timesteps, (x_train.shape[0],), device=device)
            noise = torch.randn(y_train.shape, device=device)
            noisy_samples = get_noisy_sample(t, y_train, noise)
            
            noisy_train = torch.cat((noisy_samples, x_train), dim=1)

            preds = model(noisy_train, t)
            loss = criterion(preds, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch} : Loss = {torch.mean(torch.tensor(epoch_loss)).item():.6f}")

    return

def denoising(model, x, timesteps, device, start_beta, end_beta):

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
        t_batch = torch.full((x.shape[0],), time, device=device, dtype=torch.int)

        pred = model(x_next, t_batch)    
       
        new_mask = update_x(new_mask, time, pred)

        x_next = torch.cat((new_mask, x_next[:, 1:, :, :]), dim=1)

    return x_next

