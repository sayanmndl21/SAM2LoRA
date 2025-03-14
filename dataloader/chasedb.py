import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader.datautils import BaseSegmentationDataset

"""Load the CHASEDB Dataset"""
class CHASEDBDataset(BaseSegmentationDataset):
    def __init__(
        self,
        data_dir: str,
        dataset_type: str = 'train',
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
        self.data: List[Dict[str, Any]] = self.load_data(random_state=random_state)

    def load_data(self, random_state: int = 0) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        all_images: List[str] = glob.glob(os.path.join(self.data_dir, "*.jpg"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask1_name: str = image_name[:-4] + '_1stHO.png'
            mask2_name: str = image_name[:-4] + '_2ndHO.png'
            if os.path.exists(os.path.join(self.data_dir, mask1_name)) and os.path.exists(os.path.join(self.data_dir, mask2_name)):
                data.append({
                    "image": os.path.join(self.data_dir, image_name),
                    "annotation": [
                        os.path.join(self.data_dir, mask1_name),
                        os.path.join(self.data_dir, mask2_name)
                    ]
                })

        trainset, testset = train_test_split(data, test_size=0.27, random_state=random_state)
        return trainset if self.dataset_type == 'train' else testset

    def read_image(self, image_path: str, mask_path: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]
        mask1: np.ndarray = cv2.imread(mask_path[0], cv2.IMREAD_GRAYSCALE)
        mask1 = self.binarize_mask(mask1)
        mask2: np.ndarray = cv2.imread(mask_path[1], cv2.IMREAD_GRAYSCALE)
        mask2 = self.binarize_mask(mask2)
        assert mask1.shape == mask2.shape, print("Mask shapes inconsistent")

        mask: np.ndarray = np.logical_or(mask1, mask2).astype(np.uint8)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)
        return img, mask


if __name__ == "__main__":
    basepath = os.getcwd()
    data_dir = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "CHASEDB1", "CHASEDB1")
    dataset = CHASEDBDataset(data_dir, dataset_type='train', unified=False, transform=None)