import math
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Tuple, Dict, Optional

from utils.eval_metrics import SegmentationEVAL, calculate_mean_metrics

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

numpy_sigmoid = np.vectorize(sigmoid)
segeval = SegmentationEVAL()

def eval_step(step: int, predictor: Any, input_tuple: Tuple[Any, ...], prompt_type: str = 'none') -> Dict[str, Any]:
    metrics: Optional[Dict[str, Any]] = None
    if prompt_type.lower() not in ['point', 'box', 'all', 'none']:
        raise ValueError(f'Prompt type should be point, box, all or none. Received {prompt_type}!')

    images, gt_masks, pos_points, neg_points, boxes, num_masks = input_tuple
    num_classes: int = min(gt_masks.shape[0], 3)

    if len(images) == 0 or len(gt_masks) == 0:
        return metrics

    pos_pts = np.array(pos_points)
    neg_pts = np.array(neg_points)
    box_pts = np.array(boxes)

    if prompt_type.lower() == 'none':
        pos_pts = None
        neg_pts = None
        box_pts = None
    elif prompt_type.lower() == 'point':
        box_pts = None
    elif prompt_type.lower() == 'box':
        pos_pts = None
        neg_pts = None

    if pos_pts is not None:
        input_label = np.array([1] * pos_pts.shape[0] + [0] * neg_pts.shape[0])
        input_points = np.concatenate((pos_pts, neg_pts), axis=0) if neg_pts.size > 0 else pos_pts
    else:
        input_points, input_label = None, None

    input_boxes = box_pts
    if input_points is None and input_boxes is None and prompt_type.lower() != 'none':
        raise ValueError(f"Did not find input prompt for prompt type: {prompt_type}")

    if metrics is None:
        metrics = {}

    num_pos_pts: int = pos_pts.shape[0] if pos_pts is not None else 0
    num_neg_pts: int = neg_pts.shape[0] if neg_pts is not None else 0
    num_boxes: int = 1 if box_pts is not None else 0

    with torch.no_grad():
        predictor.set_image(images)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_label,
            box=input_boxes,
            multimask_output=True,
            return_logits=True,
        )

    np_masks = np.array(masks[:num_classes])
    np_scores = scores[:num_classes]

    eval_scores = {}
    for i in range(num_classes):
        curr_y_pred_proba = numpy_sigmoid(np_masks[i]).astype(np.float32())
        curr_y_true = gt_masks[i]
        iou_pred_score = np_scores[i]

        eval_scores[f'class_{i+1}'] = segeval.get_eval(curr_y_pred_proba, curr_y_true)
        eval_scores[f'class_{i+1}']['iou_pred'] = float(iou_pred_score)

    if num_classes == 1:
        eval_scores['main'] = eval_scores['class_1']
    else:
        eval_scores['main'] = calculate_mean_metrics(list(eval_scores.values()))

    metrics['metrics'] = eval_scores
    metrics['num_pos_pts'] = num_pos_pts
    metrics['num_neg_pts'] = num_neg_pts
    metrics['num_boxes'] = num_boxes
    metrics['idx'] = step
    metrics['prompt'] = prompt_type
    metrics['num_class'] = num_classes

    return metrics
