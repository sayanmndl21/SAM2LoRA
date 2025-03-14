import os
import glob
import random
from pathlib import Path
from typing import Any, List, Tuple, Dict
import cv2
import numpy as np

from dataloader.datautils import BaseSegmentationDataset


class DRISHTIGSDataset(BaseSegmentationDataset):
    def __init__(
        self,
        data_dir: str,
        dataset_type: str = 'train',
        seg_type: str = 'od',
        unified: bool = True,
        transform: Any = None,
        color_transform: Any = None,
        num_pos_points: int = 10,
        num_neg_points: int = 0,
        num_boxes: int = 1,
        region: str = 'general',
        random_state: int = 0,
    ) -> None:
        super().__init__(
            data_dir,
            image_folder=None,
            mask_folder=None,
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
        )
        if seg_type not in ['od', 'cup', 'cup_only', 'rim']:
            raise ValueError('seg_type can be one of od: Optic Disc, cup: Cup, cup_only or rim: Neuroretinal Rim')
        self.seg_type: str = seg_type
        self.data: List[Dict[str, Any]] = self.load_data(random_state=random_state)

    def load_data(self, random_state: int = 0) -> List[Dict[str, Any]]:
        base_dir: str = os.path.join(self.data_dir, 'Training') if self.dataset_type == 'train' else os.path.join(self.data_dir, 'Test')
        data: List[Dict[str, Any]] = []
        all_images: List[str] = glob.glob(os.path.join(base_dir, "Images", "**", "*.png"), recursive=True)
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4]
            mask_base_path: str = os.path.join(base_dir, "GT")
            cup_mask: str = os.path.join(mask_base_path, f"{mask_name}/SoftMap/{mask_name}_cupsegSoftmap.png")
            od_mask: str = os.path.join(mask_base_path, f"{mask_name}/SoftMap/{mask_name}_ODsegSoftmap.png")
            if os.path.exists(cup_mask) and os.path.exists(od_mask):
                im_dict: Dict[str, Any] = {
                    "image": image_path,
                    "annotation": (cup_mask, od_mask)
                }
                if im_dict not in data:
                    data.append(im_dict)
        return data

    def read_image(self, image_path: str, mask_path: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB

        mask_od: np.ndarray = cv2.imread(mask_path[1], cv2.IMREAD_GRAYSCALE)
        mask_od = np.expand_dims(np.array(mask_od), axis=0)
        mask_od = self.binarize_mask(mask_od)

        if self.seg_type in ['cup', 'cup_only', 'rim']:
            mask_oc: np.ndarray = cv2.imread(mask_path[0], cv2.IMREAD_GRAYSCALE)
            mask_oc = np.expand_dims(np.array(mask_oc), axis=0)
            mask_oc = self.binarize_mask(mask_oc)

        if self.seg_type == 'od':
            mask = mask_od
        elif self.seg_type == 'cup':
            mask = np.concatenate((mask_od, mask_oc), axis=0)
        elif self.seg_type == 'cup_only':
            mask = mask_oc
        elif self.seg_type == 'rim':
            mask_rim: np.ndarray = np.clip(mask_od - mask_oc, 0, 1).astype(np.uint8)
            mask = np.concatenate((mask_od, mask_rim), axis=0)
        else:
            raise ValueError("Invalid segmentation type. Choose from 'od', 'cup', 'cup_only', or 'rim'.")

        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(np.transpose(mask, (1, 2, 0)), (1024, 1024), interpolation=cv2.INTER_NEAREST)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        else:
            mask = np.transpose(mask, (2, 0, 1))
        return img, mask


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "DRISHTI-GS")
    dataset = DRISHTIGSDataset(data_dir, dataset_type='train', unified=False, transform=None)