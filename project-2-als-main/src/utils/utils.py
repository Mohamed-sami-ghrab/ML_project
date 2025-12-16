import cv2
import torch
import models.unet as unet
import os
import numpy as np

def load_unet(weights_path, encoder, device):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = unet.UNet(
        encoder=encoder,
        width=400,
        height=400,   
        initial_channels=3,
        dropout=0.4
    ).to(device)


    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def ensemble_with_tta(img_tensor, models, device):
    """
    Performs soft-voting ensemble prediction with Test-Time Augmentation (Rotations).
    
    Logic:
    1. For every model in the ensemble:
       2. Predict on Original Image (0°)
       3. Predict on Image Rotated 90°  -> Rotate result back -90°
       4. Predict on Image Rotated 180° -> Rotate result back -180°
       5. Predict on Image Rotated 270° -> Rotate result back -270°
    6. Average all probability maps together.
    """
    img_tensor = img_tensor.to(device)
    
    total_probs = 0.0
    rotations = [0, 1, 2, 3] 
    
    count = 0 
    with torch.no_grad():
        for model in models:
            model.eval()
            
            for k in rotations:
                img_rotated = torch.rot90(img_tensor, k=k, dims=[2, 3])
                
                logits = model(img_rotated)
                probs = torch.sigmoid(logits)
                
                probs_original_orientation = torch.rot90(probs, k=-k, dims=[2, 3])
                
                total_probs += probs_original_orientation
                count += 1

    avg_probs = total_probs / count
    
    mask = (avg_probs > 0.5).float()
    mask_np = mask.squeeze().cpu().numpy()
    return mask_np

import torch

def ensemble_with_tta_d4(img_tensor, models, device):
    img_tensor = img_tensor.to(device)
    total_probs = 0.0
    count = 0 
    
    flips = [False, True]
    rotations = [0, 1, 2, 3] 
    
    with torch.no_grad():
        for model in models:
            model.eval()
            
            for flip in flips:
                for k in rotations:
                    x = img_tensor
                    if flip:
                        x = torch.flip(x, dims=[3]) 
                    
                    x = torch.rot90(x, k=k, dims=[2, 3])
                    
                    logits = model(x)
                    probs = torch.sigmoid(logits)
                    
                    probs = torch.rot90(probs, k=-k, dims=[2, 3])
                    
                    if flip:
                        probs = torch.flip(probs, dims=[3])
                    
                    total_probs += probs
                    count += 1

    avg_probs = total_probs / count
    
    mask = (avg_probs > 0.5).float()
    mask_np = mask.squeeze().cpu().numpy()
    
    return mask_np

def ensembling_prediction(img_tensor, models, device):
    """
    Performs soft-voting ensemble prediction.
    """
    img_tensor = img_tensor.to(device)
    
    total_probs = 0.0
    
    with torch.no_grad():
        for model in models:
            model.eval()
            
            logits = model(img_tensor)
            total_probs += torch.sigmoid(logits)

    avg_probs = total_probs / len(models)
    mask = (avg_probs > 0.5).float()
    
    mask_np = mask.squeeze().cpu().numpy()
    return mask_np

def predict_image(img_tensor, model, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.sigmoid(output)
        mask = (probs > 0.5).float()
    
    mask_np = mask.squeeze().cpu().numpy()
    return mask_np

def prepare_image_for_inference(image_path, transform, device):
    """
    Loads an image, applies transforms, and prepares it for the model.
    
    Args:
        image_path (str): Path to the input image.
        transform (albumentations.Compose): The transform pipeline.
        device (str): 'cuda' or 'cpu'.
        
    Returns:
        torch.Tensor: The image tensor ready for the model [1, C, H, W].
        tuple: (original_height, original_width) for post-processing.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    if transform:
        augmented = transform(image=image)
        image_tensor = augmented['image']
    else:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)

    image_tensor = image_tensor.float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor.to(device), (original_h, original_w)

def save_prediction(mask, filename, output_dir="results"):
    """
    Saves a binary mask (0 or 1) as a visible PNG (0 or 255).
    
    Args:
        mask (numpy.ndarray): The binary mask output from your model.
        filename (str): The name of the file (e.g., 'prediction_01.png').
        output_dir (str): Folder to save the image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    mask_to_save = (mask * 255).astype(np.uint8)
    
    save_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(save_path, mask_to_save)
    print(f"Saved: {save_path}")
