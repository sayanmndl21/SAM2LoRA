import os
import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

from dataloader.datautils import BaseSegmentationDataset


class DRIVEDataset(BaseSegmentationDataset):
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
        if self.dataset_type == 'train':
            base_path: str = os.path.join(self.data_dir, 'training')
        else:
            base_path: str = os.path.join(self.data_dir, 'test')  # contains 2 manual segmentations from independent graders
        all_images: List[str] = glob.glob(os.path.join(base_path, 'images', "*.tif"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask1_name: str = image_name[:2] + '_manual1.gif'
            if self.dataset_type == 'test':
                mask2_name: str = image_name[:2] + '_manual2.gif'
            if os.path.exists(os.path.join(base_path, '1st_manual', mask1_name)):
                im_dict: Dict[str, Any] = {
                    "image": os.path.join(base_path, 'images', image_name),
                    "annotation": [os.path.join(base_path, '1st_manual', mask1_name)]
                }
                if self.dataset_type == 'test' and os.path.exists(os.path.join(base_path, '2nd_manual', mask2_name)):
                    im_dict["annotation"].append(os.path.join(base_path, '2nd_manual', mask2_name))
                data.append(im_dict)
        return data

    def read_image(self, image_path: str, mask_path: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        # Read TIFF image using PIL
        img: Image.Image = Image.open(image_path)
        img_np: np.ndarray = np.array(img)[..., :3]

        # Read GIF masks using PIL
        mask1: Image.Image = Image.open(mask_path[0])
        mask1_np: np.ndarray = np.array(mask1.convert('L'))
        mask1_np = self.binarize_mask(mask1_np.astype(np.float32))
        if len(mask_path) == 2:
            mask2: Image.Image = Image.open(mask_path[1])
            mask2_np: np.ndarray = np.array(mask2.convert('L'))
            mask2_np = self.binarize_mask(mask2_np.astype(np.float32))
            assert mask1_np.shape == mask2_np.shape, "Mask shapes inconsistent"
            mask_np: np.ndarray = np.logical_or(mask1_np, mask2_np).astype(np.uint8)
        else:
            mask_np = mask1_np

        mask_np = self.binarize_mask(mask_np)
        img_resized: np.ndarray = cv2.resize(img_np, (1024, 1024))
        mask_resized: np.ndarray = cv2.resize(mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.expand_dims(mask_resized, axis=0)
        return img_resized, mask_resized


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "DRIVE")
    dataset = DRIVEDataset(data_dir, dataset_type='train', unified=False, transform=None)
