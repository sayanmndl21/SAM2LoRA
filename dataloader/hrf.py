import os
import cv2
import glob
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
from sklearn.model_selection import train_test_split
from dataloader.datautils import BaseSegmentationDataset


class HRFDataset(BaseSegmentationDataset):
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
        # Load images from the primary directory
        all_images: List[str] = glob.glob(os.path.join(self.data_dir, 'images', "*.jpg"))
        # Optionally add images from an alternative directory (if needed)
        all_images += glob.glob(os.path.join("your_directory_path", "images", "*.JPG"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4] + '.tif'
            mask_path: str = os.path.join(self.data_dir, "manual1", mask_name)
            if os.path.exists(mask_path):
                im_dict: Dict[str, Any] = {
                    "image": os.path.join(self.data_dir, 'images', image_name),
                    "annotation": mask_path
                }
                if im_dict not in data:
                    data.append(im_dict)
        trainset, testset = train_test_split(data, test_size=0.27, random_state=random_state)
        return trainset if self.dataset_type == 'train' else testset

    def read_image(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        # Read TIFF mask using PIL
        mask_img: Image.Image = Image.open(mask_path)
        mask_np: np.ndarray = np.array(mask_img.convert('L'))
        mask_np = self.binarize_mask(mask_np)
        img_resized: np.ndarray = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized: np.ndarray = cv2.resize(mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.expand_dims(mask_resized, axis=0)
        return img_resized, mask_resized


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "HRF")
    dataset = HRFDataset(data_dir, dataset_type='train', unified=False, transform=None)
