import os
import json
import random
import argparse
from collections import defaultdict
from typing import Any, Dict, DefaultDict
import torch
import torch.nn.utils
import numpy as np
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from model.sam2lorabase import SAM2LoRABase
from dataloader import get_dataset
from utils.eval_utils import eval_step
from argparser import get_argparser
from set_logger import setup_logger
from dictionaries import VESSEL_DATASET_DICT, DISC_DATASET_DICT
from utils.eval_metrics import calculate_mean_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main() -> None:
    seed_everything(0)
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

    parser = get_argparser()
    args: argparse.Namespace = parser.parse_args()

    checkpoint_path: str = args.checkpoint_path
    eval_mode: int = args.eval_mode
    region_type: str = args.eval_region
    output_path: str = args.output_path
    output_name: str = args.output_name
    dataset_name: str = args.dataset_name
    seg_type: str = args.seg_type
    lora_rank: int = args.lora_rank
    lora_alpha: float = args.lora_alpha
    model_size: str = args.model_size
    high_res: bool = args.high_res
    num_pos_points: int = args.num_pos_points
    num_neg_points: int = args.num_neg_points
    log_path: str = args.log_path
    basepath: str = args.basepath or os.getcwd()

    logger = setup_logger(log_path)

    model_config: str = f"configs/sam2.1/sam2.1_hiera_{model_size}.yaml"
    MODELSIZE: Dict[str, str] = {'l': 'large', 'b': 'base', 's': 'small'}
    basepath: str = os.getcwd()
    checkpoint_file: str = os.path.join(basepath, "segment-anything-2", "checkpoints", f"sam2.1_hiera_{MODELSIZE[model_size]}.pt")
    model = build_sam2(model_config, checkpoint_file)
    for param in model.parameters():
        param.requires_grad = False

    sam_model = SAM2LoRABase(model, rank=lora_rank, alpha=lora_alpha, use_high_res_features_in_sam=high_res).to('cuda')
    predictor = SAM2ImagePredictor(sam_model)

    metrics: Dict[str, Any] = {}
    if checkpoint_path and checkpoint_path.endswith('.ckpt'):
        logger.info("Loading from checkpoint...")
        checkpoint_dict: Dict[str, Any] = torch.load(checkpoint_path)
        predictor.model.load_state_dict(checkpoint_dict['model_state_dict'])
    else:
        logger.warning("Please provide valid checkpoint path, fall back to vanilla weights")
        model = build_sam2(model_config, checkpoint_file)
        for param in model.parameters():
            param.requires_grad = False
        predictor = SAM2ImagePredictor(model)
        if not output_name:
            output_name = 'vanilla'
        else:
            output_name = output_name + '_vanilla'

    predictor.model.eval()

    if eval_mode == 0:
        prompt_type = 'none'
        num_pos_points = 0
        num_neg_points = 0
        num_boxes = 0
    elif eval_mode == 1:
        prompt_type = 'point'
        num_pos_points = min(1, num_pos_points)
        num_neg_points = 0
    elif eval_mode == 2:
        prompt_type = 'point'
        num_pos_points = min(2, num_pos_points)
        num_neg_points = 0
    elif eval_mode == 3:
        prompt_type = 'point'
        num_pos_points = min(5, num_pos_points)
        num_neg_points = 0
    elif eval_mode == 4:
        prompt_type = 'point'
        num_pos_points = min(5, num_pos_points)
        num_neg_points = min(1, num_neg_points)
    elif eval_mode == 5:
        prompt_type = 'box'
    elif eval_mode == 6:
        prompt_type = 'all'
        num_pos_points = min(5, num_pos_points)
        num_neg_points = min(1, num_neg_points)

    region_type = 'density'

    dataset_dict: Dict[str, Any] = {}
    if dataset_name in ['vessel']:
        for name in VESSEL_DATASET_DICT.keys():
            dataset_dict[name.lower()] = get_dataset(
                dataset_name=name.lower(),
                mode='test',
                seg_type=seg_type,
                transform=None,
                color_transform=None,
                num_pos_points=num_pos_points,
                num_neg_points=num_neg_points,
                num_boxes=1,
                region=region_type,
                random_state=0
            )
    elif dataset_name in ['optic_disc']:
        for name in DISC_DATASET_DICT.keys():
            if name.lower() != 'idrid':
                dataset_dict[name.lower()] = get_dataset(
                    dataset_name=name.lower(),
                    mode='test',
                    seg_type=seg_type,
                    transform=None,
                    color_transform=None,
                    num_pos_points=num_pos_points,
                    num_neg_points=num_neg_points,
                    num_boxes=1,
                    region=region_type,
                    random_state=0
                )
    elif dataset_name in ['chasedb1', 'drive', 'fives', 'hrf', 'stare', 'drishtigs', 'g1020', 'grape', 'idrid', 'origa', 'papiladb', 'refuge2']:
        dataset_dict[dataset_name] = get_dataset(
            dataset_name=dataset_name,
            mode='test',
            seg_type=seg_type,
            transform=None,
            color_transform=None,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=1,
            region=region_type,
            random_state=0
        )

    os.makedirs(output_path, exist_ok=True)
    if log_path:
        os.makedirs(log_path, exist_ok=True)

    if output_name:
        output_name = output_name.removesuffix(".json")
        if 'vanilla' in output_name:
            output_name = f'fundus_{dataset_name.lower()}_{output_name.lower()}_sam2_{model_size}_e{eval_mode}.json'
        elif dataset_name.lower() != output_name.lower():
            output_name = f'fundus_{dataset_name.lower()}_{output_name.lower()}_sam2_{model_size}_r{lora_rank}_a{int(lora_alpha)}_e{eval_mode}.json'
    else:
        output_name = f'fundus_{dataset_name.lower()}_sam2_{model_size}_r{lora_rank}_a{int(lora_alpha)}_e{eval_mode}.json'

    logger.info(f"Using output path: {os.path.join(output_path, output_name)}")
    logger.info("Starting Evaluation: ...")

    eval_metrics: Dict[str, Any] = {}
    if eval_mode in [0, 1, 2, 3, 4, 5, 6]:
        for dataset_name_key, dataset_dataset in tqdm(dataset_dict.items(), desc="Dataset Progress.."):
            eval_metrics[dataset_name_key] = {}
            eval_metrics[dataset_name_key]['data'] = []
            for idx in tqdm(range(len(dataset_dataset)), desc=f"{dataset_name_key.title()} Eval Progress..", unit="data"):
                try:
                    images, masks, pos_points, neg_points, boxes, num_masks = dataset_dataset[idx]
                except Exception as e:
                    logger.error(f"Data {idx} failed for {dataset_name_key} due to {e}. Skipping...")
                    continue
                input_tuple = (images, masks, pos_points, neg_points, boxes, num_masks)
                metrics = eval_step(idx, predictor, input_tuple, prompt_type=prompt_type)
                if metrics:
                    eval_metrics[dataset_name_key]['data'].append(metrics)

            class_score_dict: DefaultDict[str, list] = defaultdict(list)
            for metric in eval_metrics[dataset_name_key]['data']:
                eval_scores = metric['metrics']
                num_pos_pts = metric['num_pos_pts']
                num_neg_pts = metric['num_neg_pts']
                num_boxes = metric['num_boxes']
                prompt_type = metric['prompt']
                num_classes = metric['num_class']
                for k, v in eval_scores.items():
                    class_score_dict[k].append(v)
            for k in class_score_dict.keys():
                eval_metrics[dataset_name_key][k] = calculate_mean_metrics(class_score_dict[k])
            eval_metrics[dataset_name_key]['num_pos_pts'] = num_pos_pts
            eval_metrics[dataset_name_key]['num_neg_pts'] = num_neg_pts
            eval_metrics[dataset_name_key]['num_boxes'] = num_boxes
            eval_metrics[dataset_name_key]['prompt'] = prompt_type
            eval_metrics[dataset_name_key]['num_class'] = num_classes

    logger.info('Finishing up...')
    json.dump(eval_metrics, open(os.path.join(output_path, output_name), 'w'), indent=4)

if __name__ == "__main__":
    main()
