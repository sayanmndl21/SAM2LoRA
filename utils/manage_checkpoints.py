import torch
from typing import Optional, Any, Dict

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    metrics: Optional[Dict] = None,
    checkpoint_path: str = "checkpoint.safetensors"
) -> None:
    if not model:
        raise ValueError('Model is required, nothing to save')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    checkpoint_path: str = "checkpoint.safetensors"
) -> Dict:
    # Load checkpoint using safetensors
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Load additional information
    metrics = checkpoint.get('metrics', {})
    return metrics
