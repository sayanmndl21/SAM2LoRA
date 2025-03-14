import numpy as np
import torch
from collections import defaultdict
from typing import Any, Dict, Optional
from utils.manage_checkpoints import save_checkpoint
from utils.eval_metrics import calculate_batch_iou
from utils.helper_funcs import update_moving_average

def training_step(
    curr_step: int,
    predictor: Any,
    input_tuple: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    loss_func: Any,  # Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
    accumulation_steps: int = 2,
    best_loss: float = float('inf'),
    output_path: Optional[str] = None,
    save_frequency: int = 100,
    writer: Optional[Any] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    # Unpack inputs
    images, masks, pos_points, neg_points, boxes, num_masks = input_tuple

    if len(images) == 0 or len(masks) == 0:
        return metrics

    pos_pts = np.array(pos_points)
    neg_pts = np.array(neg_points)
    box_pts = np.array(boxes)

    input_label = np.array([[1] * pos_pts.shape[1] + [0] * neg_pts.shape[1]] * pos_pts.shape[0])
    input_points = np.concatenate((pos_pts, neg_pts), axis=1) if neg_pts.size > 0 else pos_pts
    if box_pts.shape[-1] != 0:
        input_boxes = box_pts
    else:
        input_boxes = None

    if pos_pts.size == 0 or input_label.size == 0:
        input_points = None
        input_label = None

    predictor.set_image_batch(images)
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        input_points, input_label, box=input_boxes, mask_logits=None, normalize_coords=True
    )

    concat_points = (unnorm_coords, labels) if unnorm_coords is not None else None

    if unnorm_box is not None:
        box_coords = unnorm_box.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device).repeat(unnorm_box.size(0), 1)
        concat_points = (
            torch.cat([box_coords, concat_points[0]], dim=1),
            torch.cat([box_labels, concat_points[1]], dim=1)
        ) if concat_points is not None else (box_coords, box_labels)

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=concat_points, boxes=None, masks=None,
    )

    batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=batched_mode,
        high_res_features=high_res_features,
    )
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

    gt_mask = torch.tensor(np.array(masks).astype(np.float32)).cuda()
    num_classes = gt_mask.shape[1]
    prd_mask = prd_masks[:, :num_classes, :, :]

    loss_dicts = []
    for n in range(1):
        loss_dicts.append(loss_func(prd_mask[:, n, :, :].unsqueeze(1), gt_mask[:, n, :, :].unsqueeze(1)))
    
    pred_mask_prob = torch.sigmoid(prd_mask)
    iou = calculate_batch_iou(gt_mask, pred_mask_prob)
    score_loss = torch.abs(prd_scores[:, :num_classes] - iou).mean()

    tot_loss = sum(l_dict['total_loss'] for l_dict in loss_dicts) / len(loss_dicts)
    loss = tot_loss + score_loss * 0.1

    loss = loss / accumulation_steps
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

    if (curr_step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    if metrics is None:
        metrics = {'loss': [], 'curr_iou': [], 'mean_iou': [], 'step': [], 'best_loss': best_loss, 'total_samples': 0}

    current_batch_mean_iou = iou.mean().item()
    batch_size = iou.size(0)

    mean_iou = metrics['mean_iou'][-1] if 'mean_iou' in metrics and len(metrics['mean_iou']) > 0 else 0
    total_samples = metrics['total_samples'] if 'total_samples' in metrics else 0

    mean_iou = update_moving_average(mean_iou, current_batch_mean_iou, total_samples, batch_size)
    total_samples += batch_size

    if 'loss' not in metrics:
        metrics['loss'] = []
    metrics['loss'].append(loss.item())

    accumulated_loss = defaultdict(float)
    for l_dict in loss_dicts:
        for k, v in l_dict.items():
            accumulated_loss[k] += v.item()

    accumulated_loss = {key: value / len(loss_dicts) for key, value in accumulated_loss.items()}

    # Update metrics with the mean of each accumulated loss
    for k, v in accumulated_loss.items():
        if k not in metrics:
            metrics[k] = []
        metrics[k].append(v)

    for k in ['curr_iou', 'mean_iou', 'step', 'total_samples']:
        if k not in metrics:
            metrics['curr_iou'] = []
            metrics['mean_iou'] = []
            metrics['step'] = []
    metrics['curr_iou'].append(current_batch_mean_iou)
    metrics['mean_iou'].append(mean_iou)
    metrics['step'].append(curr_step)
    metrics['total_samples'] = total_samples
    best_loss = metrics.get('best_loss', best_loss)

    if curr_step % save_frequency == 0 and output_path is not None:
        save_path = output_path.replace('.ckpt', f'_{curr_step}.ckpt')
        save_checkpoint(predictor.model, optimizer, scheduler, scaler, metrics, save_path)

    if writer:
        writer.add_scalar("Loss/train", loss.item(), curr_step)
        writer.add_scalar("TotLoss/train", tot_loss.item(), curr_step)
        writer.add_scalar("BatchIOU/train", current_batch_mean_iou, curr_step)
        writer.add_scalar("MeanIOU/train", mean_iou, curr_step)

    if curr_step % save_frequency == 0:
        print(f"Step {curr_step}:\tAccuracy (IoU) = {mean_iou},\tLoss = {loss.item()}")

    return metrics
