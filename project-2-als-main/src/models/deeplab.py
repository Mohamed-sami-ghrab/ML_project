import torch
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from typing import Optional, Dict, Any

import torch.nn as nn


class DeepLabWrapper(nn.Module):
    """
    Wrapper class for DeepLab segmentation model.
    Simplifies training and inference operations.
    """
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        aux_loss: bool = False
    ):
        """
        Initialize DeepLab model.
        
        Args:
            num_classes: Number of segmentation classes
            backbone: Backbone architecture ('resnet50' or 'resnet101')
            pretrained: Whether to use pretrained weights
            aux_loss: Whether to use auxiliary loss during training
        """
        super(DeepLabWrapper, self).__init__()
        
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        
        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=pretrained, aux_loss=aux_loss)
        elif backbone == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=pretrained, aux_loss=aux_loss)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier head for custom number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if aux_loss:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output segmentation map of shape (B, num_classes, H, W)
        """
        output = self.model(x)
        return output['out']
    
    def train_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (returns auxiliary outputs if enabled).
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing 'out' and optionally 'aux' outputs
        """
        return self.model(x)
    
    def predict(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Perform inference and return class predictions.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            threshold: Optional threshold for binary segmentation
            
        Returns:
            Predicted class indices of shape (B, H, W)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            if threshold is not None:
                probs = torch.softmax(output, dim=1)
                predictions = (probs[:, 1] > threshold).long()
            else:
                predictions = torch.argmax(output, dim=1)
                
        return predictions
    
    def get_parameters(self, lr: float, weight_decay: float = 0.0) -> list:
        """
        Get parameter groups for optimizer.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay factor
            
        Returns:
            List of parameter dictionaries
        """
        return [
            {
                'params': self.model.backbone.parameters(),
                'lr': lr * 0.1,
                'weight_decay': weight_decay
            },
            {
                'params': self.model.classifier.parameters(),
                'lr': lr,
                'weight_decay': weight_decay
            }
        ]
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True