import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from collections import OrderedDict
import importlib
from tqdm import tqdm
import random

#=============================================================================================================
class DiceCELoss(nn.Module):
    # Added 'device' here so the tensor knows where to go. 
    # Pass your DEVICE variable when you initialize the loss function!
    def __init__(self, num_classes, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        
        # --- THE NEW FIX ---
        # Weight Background (Class 0) at 0.1, and Calcium (Class 1) at 10.0
        # This forces the model to care 100x more about the calcium pixels.
        class_weights = torch.tensor([0.1, 10.0]).to(device)
        
        # Pass these weights into standard Cross-Entropy
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(self, score, target):
        # 1. Ensure target is an integer (required for one_hot)
        target = target.long() 
        
        # 2. Convert (B, D, H, W) -> (B, D, H, W, Classes) -> (B, Classes, D, H, W)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        # 3. Apply Softmax to model logits
        predict = F.softmax(score, dim=1)
        
        # 4. Calculate intersection and union over the spatial dimensions (D, H, W)
        smooth = 1e-5
        # intersect and union shape: (Batch, Classes)
        intersect = torch.sum(predict * target_one_hot, dim=(2, 3, 4))
        union = torch.sum(predict, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))
        
        dice = (2 * intersect + smooth) / (union + smooth)
        
        # --- THE PREVIOUS FIX ---
        # Ignore the background class (index 0). Only look at class 1 (and above).
        dice_foreground = dice[:, 1:] 
        
        return 1 - dice_foreground.mean()

    def forward(self, inputs, target):
        # inputs: (B, Classes, D, H, W)
        # target: (B, D, H, W)  <-- This is your stack of 2D target masks
        target = target.long()
        #return self.ce(inputs, target) + self.dice_loss(inputs, target)
        return self.dice_loss(inputs, target)

#=============================================================================================================
class DiceWeightedCELoss(nn.Module):
    def __init__(self, calcium_weight=100.0):
        super().__init__()
        # The weight multiplier for the calcium class
        self.calcium_weight = calcium_weight

    def forward(self, logits, targets):
        # 1. Format the targets properly
        # Make sure targets are integers and shape is (Batch, Depth, Height, Width)
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        targets_ce = targets.long()
        
        # 2. Weighted Cross Entropy Loss
        # Class 0 (Background) gets a weight of 1.0
        # Class 1 (Calcium) gets your massive multiplier
        weight_tensor = torch.tensor([1.0, self.calcium_weight]).to(logits.device)
        ce_loss = F.cross_entropy(logits, targets_ce, weight=weight_tensor)

        # 3. Dice Loss
        probs = F.softmax(logits, dim=1)
        p1 = probs[:, 1, ...] # Probability of calcium
        t1 = (targets_ce == 1).float() # Ground truth of calcium
        
        intersection = (p1 * t1).sum(dim=(1, 2, 3))
        union = p1.sum(dim=(1, 2, 3)) + t1.sum(dim=(1, 2, 3))
        
        # Add 1e-5 to prevent division by zero
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5) 
        dice_loss = 1.0 - dice_score.mean()

        # Combine them: CE forces it to find the pixels, Dice forces the shape
        return ce_loss + dice_loss

#=============================================================================================================
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        # alpha weights the classes (like your 1000:1, but scaled 0 to 1. 0.75 favors calcium)
        # gamma is the focusing parameter that punishes uncertainty (2.0 is standard)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        targets_long = targets.long()

        # 1. Focal Loss
        ce_loss = F.cross_entropy(logits, targets_long, reduction='none')
        pt = torch.exp(-ce_loss) # pt is the probability of the true class
        
        # Apply the alpha weight (calcium vs background)
        alpha_t = torch.where(targets_long == 1, self.alpha, 1 - self.alpha)
        
        # Calculate the focal loss
        focal_loss = (alpha_t * (1 - pt) ** self.gamma * ce_loss).mean()

        # 2. Dice Loss
        probs = F.softmax(logits, dim=1)
        p1 = probs[:, 1, ...] # Probability of calcium
        t1 = (targets_long == 1).float()
        
        intersection = (p1 * t1).sum(dim=(1, 2, 3))
        union = p1.sum(dim=(1, 2, 3)) + t1.sum(dim=(1, 2, 3))
        
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1.0 - dice_score.mean()

        return focal_loss + dice_loss

#=============================================================================================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        targets = targets.float()
        
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # --- THE FIX: Add a positive weight of 100 to force the model to care about Calcium ---
        pos_weight = torch.tensor([1000.0]).to(inputs.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='mean', pos_weight=pos_weight
        )
        
        inputs_prob = torch.sigmoid(inputs)       
        intersection = (inputs_prob * targets).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_prob.sum() + targets.sum() + smooth)  
        
        return bce_loss + dice_loss

#=============================================================================================================
class BinaryDiceFocalLoss(nn.Module):
    # NEW: Added smooth=1.0 to prevent NaN division errors
    def __init__(self, alpha=0.75, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth 

    def forward(self, logits, targets):
        # 1. FORCE FLOAT32: This prevents 16-bit precision underflow
        logits = logits.float()
        targets = targets.float()
        
        if targets.dim() == 4: 
            targets = targets.unsqueeze(1)

        # ---------------------------
        # 2. Binary Focal Loss
        # ---------------------------
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = (alpha_t * (1 - pt) ** self.gamma * bce_loss).mean()

        # ---------------------------
        # 3. Binary Dice Loss
        # ---------------------------
        probs = torch.sigmoid(logits) 
        
        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
        
        # THE FIX: Using 1.0 instead of 1e-5 completely stops the NaN crashes
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return focal_loss + dice_loss

#=============================================================================================================