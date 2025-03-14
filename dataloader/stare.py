import os
import cv2
import glob
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
import imageio
from sklearn.model_selection import train_test_split
from dataloader.datautils import BaseSegmentationDataset


class STAREDataset(BaseSegmentationDataset):
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
        all_images: List[str] = glob.glob(os.path.join(self.data_dir, 'stare-images', "*.ppm"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4]
            mask_base_dir_exp1: str = os.path.join(self.data_dir, 'labels-ah')
            mask_base_dir_exp2: str = os.path.join(self.data_dir, 'labels-vk')
            if (os.path.exists(os.path.join(mask_base_dir_exp1, mask_name + '.ah.ppm')) and
                os.path.exists(os.path.join(mask_base_dir_exp2, mask_name + '.vk.ppm'))):
                im_dict: Dict[str, Any] = {
                    "image": os.path.join(self.data_dir, 'stare-images', image_name),
                    "annotation": (
                        os.path.join(mask_base_dir_exp1, mask_name + '.ah.ppm'),
                        os.path.join(mask_base_dir_exp2, mask_name + '.vk.ppm')
                    )
                }
                if im_dict not in data:
                    data.append(im_dict)
        trainset, testset = train_test_split(data, test_size=0.25, random_state=random_state)
        return trainset if self.dataset_type == 'train' else testset

    def read_image(self, image_path: str, mask_path: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = imageio.imread(image_path)
        mask1: np.ndarray = imageio.imread(mask_path[0])
        mask1 = self.binarize_mask(mask1)
        mask2: np.ndarray = imageio.imread(mask_path[1])
        mask2 = self.binarize_mask(mask2)
        mask: np.ndarray = np.logical_or(mask1, mask2).astype(np.uint8)
        img_resized: np.ndarray = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized: np.ndarray = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.expand_dims(mask_resized, axis=0)
        return img_resized, mask_resized


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "STARE")
    dataset = STAREDataset(data_dir, dataset_type='train', unified=False, transform=None)