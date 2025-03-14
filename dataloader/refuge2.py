import os
import cv2
import glob
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataloader.datautils import BaseSegmentationDataset


class REFUGE2Dataset(BaseSegmentationDataset):
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
            None,
            None,
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
            raise ValueError(
                'seg_type can be one of "od" (Optic Disc), "cup" (Cup), "cup_only", or "rim" (Neuroretinal Rim)'
            )
        self.seg_type: str = seg_type
        self.data: List[Dict[str, Any]] = self.load_data(random_state=random_state)

    def load_data(self, random_state: int = 0) -> List[Dict[str, Any]]:
        if self.dataset_type == 'train':
            base_dir: str = os.path.join(self.data_dir, 'train')
        elif self.dataset_type == 'val':
            base_dir = os.path.join(self.data_dir, 'val')
        else:
            base_dir = os.path.join(self.data_dir, 'test')
        data: List[Dict[str, Any]] = []
        all_images: List[str] = glob.glob(os.path.join(base_dir, "images", "*.jpg"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4]
            mask_base_path: str = os.path.join(base_dir, "mask")
            annot: str = ""
            if os.path.exists(os.path.join(mask_base_path, mask_name + '.png')):
                annot = os.path.join(mask_base_path, mask_name + '.png')
            elif os.path.exists(os.path.join(mask_base_path, mask_name + '.bmp')):
                annot = os.path.join(mask_base_path, mask_name + '.bmp')
            im_dict: Dict[str, Any] = {"image": image_path, "annotation": annot}
            if im_dict not in data:
                data.append(im_dict)
        return data

    def read_image(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask_rim: np.ndarray = (mask == 128).astype(np.uint8)
        mask_cup: np.ndarray = (mask == 0).astype(np.uint8)
        mask_od: np.ndarray = np.logical_or(mask_cup, mask_rim).astype(np.uint8)

        mask_cup = np.expand_dims(mask_cup, axis=0)
        mask_rim = np.expand_dims(mask_rim, axis=0)
        mask_od = np.expand_dims(mask_od, axis=0)

        if self.seg_type == 'od':
            final_mask = mask_od
        elif self.seg_type == 'cup':
            final_mask = np.concatenate((mask_od, mask_cup), axis=0)
        elif self.seg_type == 'cup_only':
            final_mask = mask_cup
        elif self.seg_type == 'rim':
            final_mask = np.concatenate((mask_od, mask_rim), axis=0)
        else:
            raise ValueError("Invalid segmentation type. Choose from 'od', 'cup', 'cup_only', or 'rim'.")
        final_mask = self.binarize_mask(final_mask)
        img_resized: np.ndarray = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized: np.ndarray = cv2.resize(np.transpose(final_mask, (1, 2, 0)), (1024, 1024), interpolation=cv2.INTER_NEAREST)
        if len(mask_resized.shape) == 2:
            mask_resized = np.expand_dims(mask_resized, axis=0)
        else:
            mask_resized = np.transpose(mask_resized, (2, 0, 1))
        return img_resized, mask_resized


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "REFUGE2")
    dataset = REFUGE2Dataset(data_dir, dataset_type='train', unified=False, transform=None)