import os
import glob
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import train_test_split
import skimage.draw as drw
from dataloader.datautils import BaseSegmentationDataset


def contour_to_mask(contour_file: str, img_shape: Tuple[int, int, int]) -> np.ndarray:
    c: np.ndarray = np.loadtxt(contour_file)
    mask: np.ndarray = np.zeros(img_shape[:-1], dtype=np.uint8)
    rr, cc = drw.polygon(c[:, 1], c[:, 0])
    mask[rr, cc] = 1
    return mask


class PAPILADBDataset(BaseSegmentationDataset):
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
        data: List[Dict[str, Any]] = []
        all_images: List[str] = glob.glob(os.path.join(self.data_dir, "FundusImages", "*.jpg"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4]
            mask_base_path: str = os.path.join(self.data_dir, "ExpertsSegmentations", "Contours")
            cup_exp1: str = os.path.join(mask_base_path, f"{mask_name}_cup_exp1.txt")
            cup_exp2: str = os.path.join(mask_base_path, f"{mask_name}_cup_exp2.txt")
            disc_exp1: str = os.path.join(mask_base_path, f"{mask_name}_disc_exp1.txt")
            disc_exp2: str = os.path.join(mask_base_path, f"{mask_name}_disc_exp2.txt")
            if os.path.exists(cup_exp1) and os.path.exists(cup_exp2) and os.path.exists(disc_exp1) and os.path.exists(disc_exp2):
                im_dict: Dict[str, Any] = {
                    "image": image_path,
                    "annotation": (cup_exp1, cup_exp2, disc_exp1, disc_exp2)
                }
                if im_dict not in data:
                    data.append(im_dict)
        trainset, testset = train_test_split(data, test_size=0.25, random_state=random_state)
        return trainset if self.dataset_type == 'train' else testset

    def read_image(self, image_path: str, mask_paths: Tuple[str, str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]
        width, height, _ = img.shape

        mask1: np.ndarray = contour_to_mask(mask_paths[0], img.shape)
        mask2: np.ndarray = contour_to_mask(mask_paths[1], img.shape)
        mask_oc: np.ndarray = np.logical_or(mask1, mask2).astype(np.uint8)
        mask_oc = np.expand_dims(mask_oc, axis=0)

        mask3: np.ndarray = contour_to_mask(mask_paths[2], img.shape)
        mask4: np.ndarray = contour_to_mask(mask_paths[3], img.shape)
        mask_od: np.ndarray = np.logical_or(mask3, mask4).astype(np.uint8)
        mask_od = np.expand_dims(mask_od, axis=0)

        if self.seg_type == 'od':
            mask: np.ndarray = mask_od
        elif self.seg_type == 'cup':
            mask = np.concatenate((mask_od, mask_oc), axis=0)
        elif self.seg_type == 'cup_only':
            mask = mask_oc
        elif self.seg_type == 'rim':
            mask_rim: np.ndarray = np.clip(mask_oc - mask_od, 0, 1).astype(np.uint8)
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


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "PapilaDB")
    dataset = PAPILADBDataset(data_dir, dataset_type='train', unified=False, transform=None)