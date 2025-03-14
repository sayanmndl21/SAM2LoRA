import os
import shutil
import argparse
from itertools import cycle
from typing import Any, Dict
import torch
import torch.nn.utils
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sam2.build_sam import build_sam2
from model.sam2lorabase import SAM2LoRABase
from model.sam2_image_predictor import SAM2ImagePredictor
from dataloader import get_dataset
from dataloader.retina_datasets import get_batched_dataset
from utils.loss_funcs import SoftDiceLossV1, FocalLossV1, SegmentationBCELoss, FocalTverskyLoss
from utils.manage_checkpoints import save_checkpoint
from utils.trainer_utils import training_step
from argparser import get_argparser
from set_logger import setup_logger

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

parser = get_argparser()
args: argparse.Namespace = parser.parse_args()

# Set basepath from args or default to current working directory.
basepath: str = args.basepath or os.getcwd()

region_type: str = args.eval_region
checkpoint_path: str = args.checkpoint_path
output_path: str = args.output_path
output_name: str = args.output_name
dataset_name: str = args.dataset_name
batch_size: int = args.batch_size
use_transform: Any = args.use_transform
seg_type: str = args.seg_type
lora_rank: int = args.lora_rank
lora_alpha: float = args.lora_alpha
model_size: str = args.model_size
high_res: bool = args.high_res
num_pos_points: int = int(args.num_pos_points)
num_neg_points: int = int(args.num_neg_points)
num_boxes: int = int(args.num_boxes)
optimizer_type: str = args.optimizer_type
scheduler_type: str = args.scheduler_type
accumulation_steps: int = int(args.accumulation_steps)
learning_rate: float = args.learning_rate
weight_decay: float = args.weight_decay
nesterov: bool = args.nesterov
step_size: int = args.step_size
momentum: float = args.momentum
number_steps: int = args.number_steps
tensorboard_path: str = args.tensorboard_path
log_path: str = args.log_path
save_frequency: int = int(args.save_frequency)

logger = setup_logger(log_path)

summary_path: str = os.path.join(tensorboard_path, output_name or dataset_name or '')
if os.path.exists(summary_path):
    shutil.rmtree(summary_path)
writer = SummaryWriter(log_dir=summary_path)

model_config: str = f"configs/sam2.1/sam2.1_hiera_{model_size}.yaml"
MODELSIZE: Dict[str, str] = {'l': 'large', 'b': 'base', 's': 'small'}
checkpoint_file: str = os.path.join(basepath, "segment-anything-2", "checkpoints", f"sam2.1_hiera_{MODELSIZE[model_size]}.pt")
model = build_sam2(model_config, checkpoint_file)
for param in model.parameters():
    param.requires_grad = False

sam_model = SAM2LoRABase(model, rank=lora_rank, alpha=lora_alpha, use_high_res_features_in_sam=high_res).to('cuda')
predictor = SAM2ImagePredictor(sam_model)

metrics: Dict[str, Any] = {}
if checkpoint_path.endswith('.ckpt'):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
else:
    logger.info("No checkpoints applied, falling back to OG model...")
    checkpoint = None
    model = build_sam2(model_config, checkpoint_file)
    for param in model.parameters():
        param.requires_grad = False
    predictor = SAM2ImagePredictor(model)
    output_name = output_name + '_vanilla' if output_name else 'vanilla'

if checkpoint is not None:
    predictor.model.load_state_dict(checkpoint['model_state_dict'])
    metrics = checkpoint.get('metrics', {})

pytorch_total_params: int = sum(p.numel() for p in predictor.model.parameters())
pytorch_train_params: int = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
logger.info(f"LoRA reduction in params {(pytorch_train_params / pytorch_total_params)}")

if optimizer_type == 'adamw':
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(params=predictor.model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if scheduler_type == 'cosinewarm':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=step_size, T_mult=1)
elif scheduler_type == 'steplr':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
else:
    scheduler = None

if checkpoint is not None and scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

scaler = torch.amp.GradScaler('cuda')
if checkpoint is not None:
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

if str(use_transform).lower() != 'false':
    logger.info("Using data transforms... ")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.1, p=0.6),
            transforms.RandomResizedCrop(size=(1000, 1000)),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=8, translate=(0.2, 0.2), scale=(0.2, 0.8)),
        ], p=0.5),
    ])
    color_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, hue=0.02),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.4)),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2),
        ], p=0.5),
    ])
else:
    transform, color_transform = None, None

if dataset_name in ['vessel']:
    batch_iter = get_batched_dataset(
        dataset_type=dataset_name,
        batch_size=batch_size,
        mode='train',
        seg_type=seg_type,
        transform=transform,
        color_transform=color_transform,
        num_pos_points=num_pos_points,
        num_neg_points=num_neg_points,
        num_boxes=num_boxes,
        region=region_type,
        random_state=0
    )
elif dataset_name in ['optic_disc']:
    batch_iter = get_batched_dataset(
        dataset_type=dataset_name,
        batch_size=batch_size,
        mode='train',
        seg_type=seg_type,
        transform=transform,
        color_transform=color_transform,
        num_pos_points=num_pos_points,
        num_neg_points=num_neg_points,
        num_boxes=num_boxes,
        region=region_type,
        random_state=0
    )
elif dataset_name in ['chasedb1', 'drive', 'fives', 'hrf', 'stare', 'drishtigs',
                      'g1020', 'grape', 'idrid', 'origa', 'papiladb', 'refuge2']:
    dataset = get_dataset(
        dataset_name=dataset_name,
        mode='train',
        seg_type=seg_type,
        transform=transform,
        color_transform=color_transform,
        num_pos_points=num_pos_points,
        num_neg_points=num_neg_points,
        num_boxes=num_boxes,
        region=region_type,
        random_state=0
    )
    batch_iter = dataset.get_batches(batch_size=batch_size, dataset_type='train')
else:
    raise ValueError(f'{dataset_name} does not exist!')

softdice = SoftDiceLossV1()
focalloss = FocalLossV1()
segloss = SegmentationBCELoss()
ftloss = FocalTverskyLoss()

def sam_loss_func(y_pred: Any, y_true: Any) -> Dict[str, Any]:
    sd_loss = softdice(y_pred, y_true)
    ft_loss = ftloss(y_pred, y_true)
    ce_loss = segloss(y_pred, y_true)
    tot_loss = sd_loss + ft_loss + ce_loss
    return {
        'total_loss': tot_loss,
        'soft_dice': sd_loss,
        'focal_tversky': ft_loss,
        'ce_loss': ce_loss,
    }

start_step: int = metrics.get('step', [0])[-1] + 0
end_step: int = start_step + number_steps
batch_iter = cycle(batch_iter)

os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
if log_path:
    os.makedirs(log_path, exist_ok=True)
os.makedirs(tensorboard_path, exist_ok=True)

if output_name and dataset_name.lower() != output_name.lower():
    output_name = output_name.removesuffix(".ckpt")
    output_name = f'fundus_{dataset_name.lower()}_{output_name.lower()}_sam2_{model_size}_r{lora_rank}_a{int(lora_alpha)}.ckpt'
else:
    output_name = f'fundus_{dataset_name.lower()}_sam2_{model_size}_r{lora_rank}_a{int(lora_alpha)}.ckpt'

logger.info(f"Using output: {os.path.join(output_path, output_name)}")
logger.info(f"Starting training for steps {start_step} - {end_step}: ...")
predictor.model.train()

acc_loss, best_loss = 0.0, float('inf')
for curr_step in tqdm(range(start_step + 1, end_step + 1), desc="Training Progress..", unit="step"):
    with torch.amp.autocast('cuda'):
        images, masks, pos_points, neg_points, boxes, num_masks = next(batch_iter)
        input_tuple = (images, masks, pos_points, neg_points, boxes, num_masks)
        metrics = training_step(curr_step, predictor, input_tuple, optimizer, scheduler, scaler,
                                sam_loss_func, writer=writer, accumulation_steps=accumulation_steps,
                                best_loss=best_loss, metrics=metrics, save_frequency=save_frequency,
                                output_path=os.path.join(output_path, output_name))
        acc_loss += metrics.get('loss', [0])[-1]
        best_loss = metrics.get('best_loss', float('inf'))
        if curr_step % accumulation_steps == 0:
            if acc_loss < best_loss:
                best_loss = acc_loss
                metrics['best_loss'] = best_loss
                if output_path is not None:
                    save_name = output_name.replace('.ckpt', f'_best.ckpt')
                    save_checkpoint(predictor.model, optimizer, scheduler, scaler, metrics,
                                    os.path.join(output_path, save_name))
            acc_loss = 0.0

save_checkpoint(predictor.model, optimizer, scheduler, scaler, metrics, os.path.join(output_path, output_name))
logger.info('Finishing up...')
writer.flush()
writer.close()