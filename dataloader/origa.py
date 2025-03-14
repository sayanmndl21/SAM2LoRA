import os
import glob
import scipy.io
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import train_test_split
from dataloader.datautils import BaseSegmentationDataset


class ORIGADataset(BaseSegmentationDataset):
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
                'seg_type can be one of: "od" (Optic Disc), "cup" (Cup), "cup_only", or "rim" (Neuroretinal Rim)'
            )
        self.seg_type: str = seg_type
        self.data: List[Dict[str, Any]] = self.load_data(random_state=random_state)

    def load_data(self, random_state: int = 0) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        all_images: List[str] = glob.glob(os.path.join(self.data_dir, "Images", "*.jpg"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4] + '.mat'
            mask_base_path: str = os.path.join(self.data_dir, "Semi-automatic-annotations")
            full_mask_path: str = os.path.join(mask_base_path, mask_name)
            if os.path.exists(full_mask_path):
                im_dict: Dict[str, Any] = {"image": image_path, "annotation": full_mask_path}
                if im_dict not in data:
                    data.append(im_dict)
        trainset, testset = train_test_split(data, test_size=0.25, random_state=random_state)
        return trainset if self.dataset_type == 'train' else testset

    def read_image(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        height1, width1, _ = img.shape

        mat: Dict[str, Any] = scipy.io.loadmat(mask_path)
        mask: np.ndarray = mat['mask']
        height2, width2 = mask.shape

        # Adjust mask width by cutting equal parts from both sides
        width_diff: int = width2 - width1
        cut_amount: int = width_diff // 2
        mask = mask[:, cut_amount: width2 - cut_amount]
        if mask.shape[0] > height1:
            mask = mask[:height1, :]

        mask_rim: np.ndarray = (mask == 1).astype(np.uint8)
        mask_cup: np.ndarray = (mask == 2).astype(np.uint8)
        mask_od: np.ndarray = np.logical_or(mask_rim, mask_cup).astype(np.uint8)
        mask_cup = np.expand_dims(mask_cup, axis=0)
        mask_rim = np.expand_dims(mask_rim, axis=0)
        mask_od = np.expand_dims(mask_od, axis=0)

        if self.seg_type == 'od':
            final_mask: np.ndarray = mask_od
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
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "ORIGA")
    dataset = ORIGADataset(data_dir, dataset_type='train', unified=False, transform=None)