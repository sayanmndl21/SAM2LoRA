import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class LoRALayer(nn.Module):
    def __init__(self, layer: nn.Linear, rank: int = 32, alpha: int = 64, dropout: float = 0.1, merge_weights: bool = True, device: str = 'cpu') -> None:
        super(LoRALayer, self).__init__()
        self.layer = layer
        std_dev = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float))
        in_dim = layer.in_features
        out_dim = layer.out_features
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev).to(device)
        self.B = nn.Parameter(torch.zeros(rank, out_dim)).to(device)
        self.rank = rank
        self.alpha = alpha
        self.layer.weight.requires_grad = False
        self.merged = False
        self.merge_weights = merge_weights
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.layer(x)
        
        # LoRA modification: x * A * B
        lora_output = torch.matmul(torch.matmul(x, self.A), self.B)
        
        # Apply LoRA scaling and dropout
        return base_output + ((self.alpha / self.rank) * self.dropout(lora_output))
    
class ImageEncoderAdapter(nn.Module):
    def __init__(self, atten_block: nn.Module, rank: int = 32, alpha: int = 64, dropout: float = 0.1, device: str = 'cpu') -> None:
        super(ImageEncoderAdapter, self).__init__()
        self.atten_block = atten_block

        modules_to_replace = []
        for name, module in self.atten_block.named_modules():
            if isinstance(module, nn.Linear) and 'attn.qkv' in name:
                modules_to_replace.append((name, module))
        
        for name, module in modules_to_replace:
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout, device=device)
            parent_module = self.get_parent_module(name)
            setattr(parent_module, name.split('.')[-1], lora_layer)
    
    def get_parent_module(self, name: str) -> nn.Module:
        # Helper function to get the parent module given a module name
        parts = name.split('.')[:-1]
        module = self.atten_block
        for part in parts:
            module = getattr(module, part)
        return module

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass through the adapted attention block
        x = self.atten_block(x)
        return x

    
class MaskDecoderAdapter(nn.Module):
    def __init__(self, transformer_layer: nn.Module, rank: int = 32, alpha: int = 64, dropout: float = 0.1, device: str = 'cpu') -> None:
        super(MaskDecoderAdapter, self).__init__()
        self.transformer_layer = transformer_layer

        # Collect modules to replace after iteration
        modules_to_replace = []
        for name, module in self.transformer_layer.named_modules():
            if isinstance(module, nn.Linear) and 'attn' in name:
                modules_to_replace.append((name, module))

        # Replace the modules after iteration
        for name, module in modules_to_replace:
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout, device=device)
            parent_module = self.get_parent_module(name)
            setattr(parent_module, name.split('.')[-1], lora_layer)

    def get_parent_module(self, name: str) -> nn.Module:
        # Helper function to get the parent module given a module name
        parts = name.split('.')[:-1]
        module = self.transformer_layer
        for part in parts:
            module = getattr(module, part)
        return module

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        queries, keys = self.transformer_layer(image_embedding, image_pe, point_embedding)
        return queries, keys
    

class ConvLoRALayer(nn.Module):
    def __init__(self, conv_module: nn.Module, rank: int = 32, alpha: int = 64, dropout: float = 0.1, merge_weights: bool = True, **kwargs) -> None:
        super(ConvLoRALayer, self).__init__()
        in_channels = conv_module.in_channels
        out_channels = conv_module.out_channels
        kernel_size = conv_module.kernel_size[0] if isinstance(conv_module.kernel_size, tuple) else conv_module.kernel_size
        self.conv = conv_module
        self.rank = rank
        self.alpha = alpha
        self.merged = False
        self.merge_weights = merge_weights
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((rank * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, rank * kernel_size))
            )
            self.scaling = self.alpha / self.rank
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self) -> None:
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True) -> nn.Module:
        super(ConvLoRALayer, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.rank > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.rank > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.rank > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)
    

class PromptEncoderAdapter(nn.Module):
    def __init__(self, mask_downscaling: nn.Module, rank: int = 32, alpha: int = 64, dropout: float = 0.1, device: str = 'cpu') -> None:
        super(PromptEncoderAdapter, self).__init__()
        self.mask_downscaling = mask_downscaling

        # Collect modules to replace after iteration
        modules_to_replace = []
        for name, module in self.mask_downscaling.named_modules():
            if isinstance(module, nn.Conv2d):
                modules_to_replace.append((name, module))

        # Replace the modules after iteration
        for name, module in modules_to_replace:
            lora_layer = ConvLoRALayer(module, rank=rank, alpha=alpha, dropout=dropout, device=device)
            parent_module = self.get_parent_module(name)
            setattr(parent_module, name.split('.')[-1], lora_layer)

    def get_parent_module(self, name: str) -> nn.Module:
        # Helper function to get the parent module given a module name
        parts = name.split('.')[:-1]
        module = self.mask_downscaling
        for part in parts:
            module = getattr(module, part)
        return module

    def forward(self, masks: Tensor) -> Tuple[Tensor, Tensor]:
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    
class CLProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super(CLProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
