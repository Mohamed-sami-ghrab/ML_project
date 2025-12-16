import torch

class DiceBCELoss(torch.nn.Module):
    def __init__(self, alpha=0.8, beta=0.2): # Changed defaults to favor F1
        super(DiceBCELoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs_sig = torch.sigmoid(inputs)       
        
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        bce_loss = self.bce(inputs, targets)
        
        return bce_loss * self.beta + dice_loss * self.alpha