import os
import re
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torchvision.models as models
from src.models.unet import UNet 

# --- CONFIG ---
CHECKPOINT_DIR = "checkpoints/" 
TEST_IMAGES_DIR = "./data/test_set_images/"
SUBMISSION_FILE = "submission_final.csv"
IMAGE_SIZE = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
# --------------

def load_models_ensemble():
    models_list = []
    # Load ResNet34 models
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if "resnet34" in f and ".pth" in f]
    checkpoints.sort()
    
    print(f"Loading {len(checkpoints)} models...")
    for ckpt in checkpoints:
        encoder = models.resnet34(weights=None)
        model = UNet(encoder=encoder, width=IMAGE_SIZE, height=IMAGE_SIZE, initial_channels=3, dropout=0.0)
        state_dict = torch.load(os.path.join(CHECKPOINT_DIR, ckpt), map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        models_list.append(model)
    return models_list

def predict_tta(models_list, x):
    # Test Time Augmentation
    preds_accum = None
    augments = [lambda img: img, lambda img: torch.flip(img, [3]), lambda img: torch.flip(img, [2])]
    de_augments = [lambda img: img, lambda img: torch.flip(img, [3]), lambda img: torch.flip(img, [2])]

    total = 0
    for model in models_list:
        for i, aug in enumerate(augments):
            with torch.no_grad():
                pred = torch.sigmoid(model(aug(x)))
            pred = de_augments[i](pred)
            if preds_accum is None: preds_accum = pred
            else: preds_accum += pred
            total += 1
    return preds_accum / total

def mask_to_submission_strings(image_filename, pred_mask):
    img_number = int(re.search(r"(\d+)", os.path.basename(image_filename)).group(1))
    im = pred_mask 
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = 1 if np.mean(patch) > 127 else 0
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

if __name__ == "__main__":
    models_list = load_models_ensemble()
    
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2(),
    ])

    # --- THE FIX FOR DUPLICATE ROWS ---
    test_files = []
    for root, dirs, files in os.walk(TEST_IMAGES_DIR):
        for file in files:
            # Ignore hidden files and checkpoints
            if file.endswith(".png") and not file.startswith(".") and "checkpoint" not in file:
                test_files.append(os.path.join(root, file))
    
    # Sort 1, 2, 3...
    test_files.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1)))
    
    print(f"Processing {len(test_files)} valid test images.") 
    if len(test_files) != 50:
        print("WARNING: You do not have exactly 50 images. Check your folder!")

    with open(SUBMISSION_FILE, 'w') as f:
        f.write('id,prediction\n')
        for image_path in tqdm(test_files):
            original_img = cv2.imread(image_path)
            if original_img is None: continue
            
            orig_h, orig_w = original_img.shape[:2]
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            x = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE).float()
            
            avg_pred = predict_tta(models_list, x).squeeze().cpu().numpy()
            avg_pred = cv2.resize(avg_pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Post-Processing
            mask = (avg_pred > THRESHOLD).astype(np.uint8) * 255
            # Morphological Cleanup (Crucial for score)
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image_path, mask))

    print(f"Done! {SUBMISSION_FILE} created.")