import os
import glob
import random
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from dataloader.datautils import BaseSegmentationDataset


class GRAPEDataset(BaseSegmentationDataset):
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
            raise ValueError('seg_type can be one of: "od" (Optic Disc), "cup" (Cup), "cup_only", or "rim" (Neuroretinal Rim)')
        self.seg_type: str = seg_type
        self.data: List[Dict[str, Any]] = self.load_data(random_state=random_state)

    def load_data(self, random_state: int = 0) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        all_images: List[str] = glob.glob(os.path.join(self.data_dir, "ROI images", "*.jpg"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4] + '.json'
            mask_base_path: str = os.path.join(self.data_dir, "json")
            if os.path.exists(os.path.join(mask_base_path, mask_name)):
                im_dict: Dict[str, Any] = {
                    "image": image_path,
                    "annotation": os.path.join(mask_base_path, mask_name)
                }
                if im_dict not in data:
                    data.append(im_dict)
        trainset, testset = train_test_split(data, test_size=0.25, random_state=random_state)
        return trainset if self.dataset_type == 'train' else testset

    def read_image(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        height, width, _ = img.shape

        polygons: Dict[str, Any] = self.read_poly_coords(mask_path)

        # Create mask for OD
        img_temp_od: Image.Image = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img_temp_od).polygon(polygons['od'], outline=1, fill=1)
        mask_od: np.ndarray = np.expand_dims(np.array(img_temp_od), axis=0)

        # Create mask for cup if needed
        if self.seg_type in ['cup', 'rim', 'cup_only']:
            img_temp_cup: Image.Image = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img_temp_cup).polygon(polygons['cup'], outline=1, fill=1)
            mask_cup: np.ndarray = np.expand_dims(np.array(img_temp_cup), axis=0)

        if self.seg_type == 'od':
            mask: np.ndarray = mask_od
        elif self.seg_type == 'cup':
            mask = np.concatenate((mask_od, mask_cup), axis=0)
        elif self.seg_type == 'cup_only':
            mask = mask_cup
        elif self.seg_type == 'rim':
            mask_rim: np.ndarray = np.clip(mask_od - mask_cup, 0, 1).astype(np.uint8)
            mask = np.concatenate((mask_od, mask_rim), axis=0)
        else:
            raise ValueError("Invalid segmentation type. Choose from 'od', 'cup', 'cup_only', or 'rim'.")

        mask = self.binarize_mask(mask)
        img_resized: np.ndarray = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized: np.ndarray = cv2.resize(np.transpose(mask, (1, 2, 0)), (1024, 1024), interpolation=cv2.INTER_NEAREST)
        if len(mask_resized.shape) == 2:
            mask_resized = np.expand_dims(mask_resized, axis=0)
        else:
            mask_resized = np.transpose(mask_resized, (2, 0, 1))
        return img_resized, mask_resized

    def read_poly_coords(self, fname: str) -> Dict[str, List[Tuple[float, float]]]:
        polygon: Dict[str, List[Tuple[float, float]]] = {}
        if fname.endswith('json'):
            lbl_dict = json.load(open(fname, 'r'))
            for lbl in lbl_dict['shapes']:
                if lbl['label'] == 'OD':
                    polygon['od'] = [tuple(pt) for pt in lbl['points']]
                elif lbl['label'] == 'OC':
                    polygon['cup'] = [tuple(pt) for pt in lbl['points']]
                else:
                    polygon['discloc'] = [tuple(pt) for pt in lbl['points']]
        return polygon


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "GRAPE")
    dataset = GRAPEDataset(data_dir, dataset_type='train', unified=False, transform=None)