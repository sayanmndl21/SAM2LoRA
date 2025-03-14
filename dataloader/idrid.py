import os
import glob
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
from dataloader.datautils import BaseSegmentationDataset


class IDRIDDataset(BaseSegmentationDataset):
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
        if seg_type not in ['od', 'se', 'he', 'hm', 'ma']:
            raise ValueError(
                'seg_type can be one of: "od" (Optic Disc), "se" (Soft Exudates), '
                '"he" (Hard Exudates), "hm" (Haemorrhages), "ma" (Microaneurysms)'
            )
        self.seg_type: str = seg_type
        self.data: List[Dict[str, Any]] = self.load_data(random_state=random_state)

    def load_data(self, random_state: int = 0) -> List[Dict[str, Any]]:
        img_base_dir: str = os.path.join(self.data_dir, 'Images')
        mask_base_dir: str = os.path.join(self.data_dir, 'GT')
        if self.dataset_type == 'train':
            img_base_dir = os.path.join(img_base_dir, 'a. Training Set')
            mask_base_dir = os.path.join(mask_base_dir, 'a. Training Set')
        else:
            img_base_dir = os.path.join(img_base_dir, 'b. Testing Set')
            mask_base_dir = os.path.join(mask_base_dir, 'b. Testing Set')
        
        data: List[Dict[str, Any]] = []
        all_images: List[str] = glob.glob(os.path.join(img_base_dir, "*.jpg"))
        for image_path in all_images:
            image_name: str = Path(image_path).name
            mask_name: str = image_name[:-4]
            annot: Dict[str, str] = {}
            if os.path.exists(os.path.join(mask_base_dir, '1. Microaneurysms', mask_name + '_MA.tif')):
                annot['ma'] = os.path.join(mask_base_dir, '1. Microaneurysms', mask_name + '_MA.tif')
            if os.path.exists(os.path.join(mask_base_dir, '2. Haemorrhages', mask_name + '_HE.tif')):
                annot['hm'] = os.path.join(mask_base_dir, '2. Haemorrhages', mask_name + '_HE.tif')
            if os.path.exists(os.path.join(mask_base_dir, '3. Hard Exudates', mask_name + '_EX.tif')):
                annot['he'] = os.path.join(mask_base_dir, '3. Hard Exudates', mask_name + '_EX.tif')
            if os.path.exists(os.path.join(mask_base_dir, '4. Soft Exudates', mask_name + '_SE.tif')):
                annot['se'] = os.path.join(mask_base_dir, '4. Soft Exudates', mask_name + '_SE.tif')
            if os.path.exists(os.path.join(mask_base_dir, '5. Optic Disc', mask_name + '_OD.tif')):
                annot['od'] = os.path.join(mask_base_dir, '5. Optic Disc', mask_name + '_OD.tif')
            if self.seg_type in annot:
                im_dict: Dict[str, Any] = {"image": image_path, "annotation": annot[self.seg_type]}
                if im_dict not in data:
                    data.append(im_dict)
        return data

    def read_image(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        mask: np.ndarray = np.array(Image.open(mask_path).convert('L'))
        mask = self.binarize_mask(mask)
        img_resized: np.ndarray = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized: np.ndarray = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.expand_dims(mask_resized, axis=0)
        return img_resized, mask_resized


if __name__ == "__main__":
    basepath: str = os.getcwd()
    data_dir: str = os.path.join(basepath, "datasets", "eye_dataset", "Fundus", "IDRID", "Segmentation")
    dataset = IDRIDDataset(data_dir, dataset_type='train', unified=False, transform=None)
