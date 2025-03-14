import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp

# https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py

class FocalLossV1(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean') -> None:
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                                F.softplus(logits, -1, 50),
                                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                                  -logits + F.softplus(logits, -1, 50),
                                  -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class SoftDiceLossV1(nn.Module):
    '''
    Soft-dice loss, useful in binary segmentation.
    '''
    def __init__(self, p: float = 1, smooth: float = 1) -> None:
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            labels: tensor of shape (N, H, W, ...)
        output:
            loss: tensor of shape (1, )
        '''
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

class SegmentationBCELoss(nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Binary Cross Entropy loss for segmentation tasks with support for different reduction methods.
        
        Args:
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(SegmentationBCELoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction

    def forward(self, pred_masks: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            pred_masks (torch.Tensor): Predicted masks (logits) of shape [B, C, H, W] where C = 1.
            gt_mask (torch.Tensor): Ground truth masks of shape [B, C, H, W] where C = 1.
        
        Returns:
            torch.Tensor: Computed Binary Cross Entropy loss.
        """
        prd_mask_prob = torch.sigmoid(pred_masks)  # Ensure shape [B, 1, H, W]
        epsilon = 1e-6  # Small value to avoid log(0)
        seg_loss = (-gt_mask * torch.log(prd_mask_prob + epsilon) - 
                    (1 - gt_mask) * torch.log((1 - prd_mask_prob) + epsilon))
        if self.reduction == 'mean':
            return seg_loss.mean()
        elif self.reduction == 'sum':
            return seg_loss.sum()
        else:  # 'none'
            return seg_loss

class SegmentationMSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Mean Squared Error (MSE) loss for segmentation tasks with support for different reduction methods.
        
        Args:
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(SegmentationMSELoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction

    def forward(self, pred_masks: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            pred_masks (torch.Tensor): Predicted masks (logits) of shape [B, C, H, W] where C = 1.
            gt_mask (torch.Tensor): Ground truth masks of shape [B, C, H, W] where C = 1.
        
        Returns:
            torch.Tensor: Computed Mean Squared Error loss.
        """
        prd_mask_prob = torch.sigmoid(pred_masks)  # Ensure shape [B, 1, H, W]
        seg_loss = (prd_mask_prob - gt_mask) ** 2
        if self.reduction == 'mean':
            return seg_loss.mean()
        elif self.reduction == 'sum':
            return seg_loss.sum()
        else:  # 'none'
            return seg_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, gamma: float = 0.75, smooth: float = 1) -> None:
        """
        Initializes FocalTverskyLoss with Tversky index parameters.
        
        Args:
            alpha (float): Weight for false negatives.
            gamma (float): Focal parameter to adjust the loss.
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the Focal Tversky loss.
        
        Args:
            logits (torch.Tensor): Raw predictions of shape (N, H, W, ...).
            labels (torch.Tensor): Ground truth labels of shape (N, H, W, ...).
        
        Returns:
            torch.Tensor: The calculated Focal Tversky Loss.
        """
        probs = torch.sigmoid(logits)
        y_true = labels.reshape(-1)
        y_pred = probs.reshape(-1)
        true_pos = (y_true * y_pred).sum()
        false_neg = (y_true * (1 - y_pred)).sum()
        false_pos = ((1 - y_true) * y_pred).sum()
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)
        return focal_tversky_loss