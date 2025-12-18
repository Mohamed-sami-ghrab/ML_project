# Project 2: Aerial Image Segmentation (ALS)

This project implements various deep learning architectures for semantic segmentation of aerial imagery. It includes implementations for custom U-Nets, SegFormer fine-tuning, and experimental diffusion models.

## 📂 Project Structure

```text
project-2-als-main/
├── data/                       # Data folder (not included in repo, see Setup)
├── src/
│   ├── dataset.py              # Custom PyTorch Dataset class
│   ├── helpers.py              # Helper functions
│   ├── loss.py                 # Custom loss functions (DiceBCELoss)
│   ├── trainer.py              # Generic training loop utilities
│   ├── models/
│   │   ├── unet.py             # Custom U-Net implementation with ResNet encoder
│   │   ├── deeplab.py          # DeepLab model definitions
│   │   ├── diffusion_unet.py   # U-Net modified for diffusion tasks
│   │   └── ...
│   └── utils/
│       ├── metrics.py          # F1 Score and other metrics
│       ├── plot_utils.py       # Visualization tools
│       ├── post_processing.py  # Morphological operations, etc.
│       └── dataset_utils/      # Scripts for cropping and cleaning data
├── train.py                    # Training script for the U-Net baseline
├── train_diffusion.py          # Training script for Diffusion model
├── finetune_segformer.py       # Script to fine-tune SegFormer (mit_b2)
├── tune_deeplab.py             # Hyperparameter tuning for DeepLab
├── overfit_diffusion.py        # Sanity check for diffusion models
├── README.md                   # Project documentation
└── .gitignore
🚀 Key Features
Models:

U-Net: Custom implementation using a ResNet18 encoder with skip connections.

SegFormer: Fine-tuning script for SegFormer-mit_b2 using segmentation_models_pytorch.

Diffusion: Experimental support for diffusion-based segmentation.

Data Augmentation: Extensive pipeline using albumentations (Flip, Rotate, GridDistortion, RandomBrightnessContrast, HueSaturationValue).

Logging: Integration with Weights & Biases (WandB) for experiment tracking.

Loss Functions: Combination of BCE (Binary Cross Entropy) and Dice Loss.

🛠️ Setup & Installation
1. Dependencies
Ensure you have Python installed. Install the required libraries:

Bash

pip install torch torchvision albumentations opencv-python segmentation-models-pytorch wandb numpy
2. Data Preparation
Important: The dataset is not included in this repository due to size constraints. You must create a data directory in the root of the project and structure it as follows:

Plaintext

data/
├── training/
│   ├── images/         # Input images
│   └── groundtruth/    # Binary segmentation masks
├── validation/
│   ├── images/
│   └── groundtruth/
└── chicago_crops/      # Additional training data (if using SegFormer script)
    ├── images/
    └── groundtruth/
Note: The Dataset class expects masks to be grayscale images where pixel values > 127 represent the positive class.

💻 Usage
Training U-Net (Sanity Check / Baseline)
To run the basic U-Net training loop which includes a sanity check (saving input/output images to disk):

Bash

python train.py
Config: Hyperparameters are currently hardcoded in train.py.

Output: Saves sanity_check_*.png images to the root directory.

Fine-Tuning SegFormer
To train the SegFormer model using the combined dataset (Chicago + Training):

Bash

python finetune_segformer.py
Config: You can modify the CONFIG dictionary at the top of finetune_segformer.py to change epochs, learning rate, or batch size.

Checkpoints: Models are saved to the checkpoints/ directory.

Logging: This script attempts to log metrics to WandB. Ensure you are logged in or set mode="disabled" in the wandb.init call if you don't use it.

Other Scripts
train_diffusion.py: Train the experimental diffusion model.

tune_deeplab.py: Run hyperparameter tuning for DeepLab.

src/utils/dataset_utils/: Contains scripts like generate_crops.py to preprocess large satellite images into smaller crops.

📊 Metrics
The project primarily uses F1 Score and Dice Loss to evaluate model performance.
