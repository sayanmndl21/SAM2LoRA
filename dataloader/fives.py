import os
import glob
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataloader.datautils import BaseSegmentationDataset


class FIVESDataset(BaseSegmentationDataset):
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
            region=region
        )
        self.data: List[Dict[str, Any]] = self.load_data(random_state=random_state)

    def load_data(self, random_state: int = 0) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        if self.dataset_type == 'train':
            base_path: str = os.path.join(self.data_dir, 'train')
        else:
            base_path: str = os.path.join(self.data_dir, 'test')
        all_images: List[str] = glob.glob(os.path.join(base_path, 'Original', "*.png"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask1_name: str = image_name
            mask_path: str = os.path.join(base_path, "Ground truth", mask1_name)
            if os.path.exists(mask_path):
                im_dict: Dict[str, Any] = {
                    "image": os.path.join(base_path, 'Original', image_name),
                    "annotation": mask_path
                }
                data.append(im_dict)
        return data

    def read_image(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.binarize_mask(mask.astype(np.float32))
        img = cv2.resize(img, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)
        return img, mask


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "FIVES")
    dataset = FIVESDataset(data_dir, dataset_type='train', unified=False, transform=None)
