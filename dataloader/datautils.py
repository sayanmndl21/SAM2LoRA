import os
import json
from typing import Any, Dict, List, Tuple, Iterator
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.spatial import distance


class BaseSegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_folder: str,
        mask_folder: str,
        dataset_type: str = 'train',
        unified: bool = True,
        transform: Any = None,
        color_transform: Any = None,
        num_pos_points: int = 10,
        num_neg_points: int = 0,
        num_boxes: int = 1,
        seg_type: str = None,
        region: str = 'general',
        erode_kernel_size: int = 2,
    ) -> None:
        self.data_dir: str = data_dir
        self.image_folder: str = image_folder
        self.mask_folder: str = mask_folder
        self.dataset_type: str = dataset_type
        self.unified: bool = unified
        self.transform: Any = transform
        self.color_transform: Any = color_transform
        self.num_pos_points: int = num_pos_points
        self.num_neg_points: int = num_neg_points
        self.num_boxes: int = num_boxes
        self.region: str = region
        self.seg_type: str = seg_type or ''
        self.erode_kernel_size: int = erode_kernel_size
        self.density_blur_size: int = 15
        self.density_threshold: float = 0.9
        # Load data in the respective dataloaders; uncomment the following line if desired.
        # self.data: List[Dict[str, Any]] = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        if self.unified:
            image_folder: str = self.image_folder or "images"
            mask_folder: str = self.mask_folder or "masks"
        else:
            image_folder = self.image_folder or f"images_{self.dataset_type}"
            mask_folder = self.mask_folder or f"masks_{self.dataset_type}"

        folder_path: str = os.path.join(self.data_dir, image_folder)
        for name in os.listdir(folder_path):
            mask_path: str = os.path.join(self.data_dir, mask_folder, name[:-4] + ".png")
            if os.path.exists(mask_path):
                data.append({
                    "image": os.path.join(self.data_dir, image_folder, name),
                    "annotation": mask_path,
                })
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        entry: Dict[str, Any] = self.data[idx]
        img, mask = self.read_image(entry["image"], entry["annotation"])
        mask = self.binarize_mask(mask)

        if self.transform and self.dataset_type == 'train':
            img, mask = self.apply_transforms(img, mask)
        pos_points = self.get_points(mask, num_points=self.num_pos_points, region=self.region)
        if self.seg_type in ['rim', 'cup', 'cup_only']:
            neg_points = self.get_points_negative_local(mask, num_points=self.num_neg_points, region='center')
        else:
            neg_points = self.get_points_negative(mask, num_points=self.num_neg_points)
        boxes = self.get_boxes(mask, num_boxes=self.num_boxes)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        return img, mask, pos_points, neg_points, boxes, (self.num_pos_points + self.num_boxes)

    def read_image(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img: np.ndarray = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img.shape != mask.shape:
            print("Warning, image and mask shapes are not consistent, reshaping...")
            r: float = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
            img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
            mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.float32)
        thresh: float = (np.max(mask) + np.min(mask)) / 2
        mask = (mask > thresh).astype(np.uint8)
        return mask

    def apply_transforms(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert image and mask to torch tensors
        if self.transform:
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to CHW format
            if len(mask.shape) == 2:
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            else:
                mask_tensor = torch.tensor(mask, dtype=torch.float32)
            combined = torch.cat([img_tensor, mask_tensor], dim=0)
            transformed = self.transform(combined)
            img_transformed = transformed[:3, :, :]
            mask_transformed = transformed[3:, :, :]
            if self.color_transform:
                img_transformed = self.color_transform(img_transformed.type(torch.uint8))
            img_np = img_transformed.permute(1, 2, 0).numpy().astype(np.uint8)
            img_np = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            mask_np = cv2.resize(mask_transformed.permute(1, 2, 0).numpy(), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            if len(mask_np.shape) == 2:
                mask_np = np.expand_dims(mask_np, axis=0)
            else:
                mask_np = np.transpose(mask_np, (2, 0, 1))
            return img_np, self.binarize_mask(mask_np)
        return img, self.binarize_mask(mask)

    def get_batches(self, batch_size: int, dataset_type: str = 'train') -> Iterator[Tuple[List[Any], List[Any], List[Any], List[Any], List[Any], List[int]]]:
        for i in range(0, len(self), batch_size):
            batch_data = self.data[i:i + batch_size]
            images: List[Any] = []
            masks: List[Any] = []
            pos_points: List[Any] = []
            neg_points: List[Any] = []
            boxes: List[Any] = []
            for entry in batch_data:
                try:
                    img, mask = self.read_image(entry["image"], entry["annotation"])
                    mask = self.binarize_mask(mask)
                    if dataset_type == 'train' and self.transform:
                        img, mask = self.apply_transforms(img, mask)
                    pos = self.get_points(mask, self.num_pos_points, region=self.region)
                    neg = self.get_points_negative(mask, self.num_neg_points)
                    box = self.get_boxes(mask, self.num_boxes)
                    if len(mask.shape) == 2:
                        mask = np.expand_dims(mask, axis=0)
                except Exception:
                    continue
                else:
                    images.append(img)
                    masks.append(mask)
                    pos_points.append(pos)
                    neg_points.append(neg)
                    boxes.append(box)
            yield images, masks, pos_points, neg_points, boxes, [self.num_pos_points + self.num_boxes] * batch_size

    def get_points(self, mask: np.ndarray, num_points: int, region: str = 'general') -> np.ndarray:
        points: List[Any] = []
        if int(num_points) < 1:
            return np.array(points)
        target_mask: np.ndarray = mask[-1] if len(mask.shape) == 3 else mask
        h: int = target_mask.shape[-1]
        eroded_mask: np.ndarray = cv2.erode(target_mask, np.ones((self.erode_kernel_size, self.erode_kernel_size), np.uint8), iterations=1)
        coords: np.ndarray = np.argwhere(eroded_mask > 0)

        if region == 'center':
            centroid = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - centroid, axis=1)
            sorted_indices = np.argsort(distances)
            sorted_coords = coords[sorted_indices]
            threshold_distance = (1 / num_points) * max(eroded_mask.shape)
            selected_coords = []
            for coord in sorted_coords:
                if len(selected_coords) >= num_points:
                    break
                if all(np.linalg.norm(coord - np.array(prev)) > threshold_distance for prev in selected_coords):
                    selected_coords.append([coord[1], coord[0]])
            points = selected_coords

        elif region == 'periphery':
            centroid = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - centroid, axis=1)
            sorted_indices = np.argsort(distances)[::-1]
            sorted_coords = coords[sorted_indices]
            threshold_distance = (5 / num_points) * max(eroded_mask.shape)
            selected_coords = []
            for coord in sorted_coords:
                if len(selected_coords) >= num_points:
                    break
                if all(np.linalg.norm(coord - np.array(prev)) > threshold_distance for prev in selected_coords):
                    selected_coords.append([coord[1], coord[0]])
            points = selected_coords

        elif region == 'density':
            initial_erode_size = max(1, self.density_blur_size)
            success = False
            while not success:
                try:
                    kernel = np.ones((initial_erode_size, initial_erode_size), np.uint8)
                    neroded_mask = cv2.erode(eroded_mask, kernel, iterations=1)
                    threshold_value = np.max(neroded_mask) * self.density_threshold
                    high_density_region = np.where(neroded_mask >= threshold_value, 1, 0).astype(np.uint8)
                    dense_coords = np.argwhere(high_density_region > 0)
                    selected_coords = dense_coords[np.random.choice(dense_coords.shape[0], num_points, replace=False)]
                    points = [[int(coord[1]), int(coord[0])] for coord in selected_coords]
                    success = True
                except Exception:
                    initial_erode_size -= 1
                    if initial_erode_size <= 0:
                        points = [[int(coord[1]), int(coord[0])] for coord in np.random.choice(coords, num_points)]
                        break

        else:
            for _ in range(num_points):
                yx = np.array(coords[np.random.randint(len(coords))])
                points.append([yx[1], yx[0]])

        return np.array(points)

    def get_points_negative(self, mask: np.ndarray, num_points: int) -> np.ndarray:
        points: List[Any] = []
        if num_points < 1:
            return np.array(points)
        target_mask: np.ndarray = mask[-1] if len(mask.shape) == 3 else mask
        coords: np.ndarray = np.argwhere(target_mask == 0)
        for _ in range(num_points):
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])
        return np.array(points)

    def get_points_negative_local(self, mask: np.ndarray, num_points: int, region: str = "general") -> np.ndarray:
        points: List[Any] = []
        if num_points < 1:
            return np.array(points)
        target_mask: np.ndarray = mask[-1] if len(mask.shape) == 3 else mask
        coords: np.ndarray = np.argwhere(target_mask == 0)
        if region == "general":
            for _ in range(num_points):
                yx = np.array(coords[np.random.randint(len(coords))])
                points.append([yx[1], yx[0]])
        elif region in ["center", "periphery"]:
            positive_coords = np.argwhere(target_mask == 1)
            centroid = np.mean(positive_coords, axis=0).astype(int)
            dists = distance.cdist([centroid], coords).flatten()
            thresh_factor = 0.05 if self.seg_type in ['rim', 'cup'] else 1
            success = False
            while not success:
                try:
                    if region == "center":
                        threshold_distance = (1 / num_points) * min(target_mask.shape) * thresh_factor
                        close_points = coords[dists <= threshold_distance]
                        chosen_indices = np.random.choice(len(close_points), num_points, replace=False)
                        points = [close_points[i][::-1].tolist() for i in chosen_indices]
                        success = True
                    elif region == "periphery":
                        threshold_distance = (5 / num_points) * min(target_mask.shape) * thresh_factor
                        far_points = coords[dists >= threshold_distance]
                        chosen_indices = np.random.choice(len(far_points), num_points, replace=False)
                        points = [far_points[i][::-1].tolist() for i in chosen_indices]
                        success = True
                except Exception:
                    thresh_factor *= 2
        return np.array(points)

    def get_boxes(self, mask: np.ndarray, num_boxes: int = None) -> np.ndarray:
        if num_boxes is not None and num_boxes == 0:
            return np.array([])
        target_mask: np.ndarray = mask[-1] if len(mask.shape) == 3 else mask
        eroded_mask: np.ndarray = cv2.erode(target_mask, np.ones((self.erode_kernel_size, self.erode_kernel_size), np.uint8), iterations=1)
        eroded_mask = eroded_mask.squeeze()
        row, col = np.argwhere(eroded_mask).T
        y0, x0 = row.min(), col.min()
        y1, x1 = row.max(), col.max()
        box: List[float] = [x0, y0, x1, y1]
        if self.dataset_type == 'train':
            threshold = np.random.uniform(0, 0.02 * max(eroded_mask.shape))
            box = [
                max(0, x0 - threshold),
                max(0, y0 - threshold),
                min(eroded_mask.shape[1], x1 + threshold),
                min(eroded_mask.shape[0], y1 + threshold)
            ]
        return np.array(box)

    def visualize_data(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        pos_points: np.ndarray,
        neg_points: np.ndarray,
        boxes: np.ndarray,
        visualize_data: bool = True,
    ) -> None:
        if visualize_data:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(img)
            plt.axis('off')
            color_maps = ['jet', 'viridis']
            plt.subplot(1, 3, 2)
            plt.title('Binarized Mask')
            plt.imshow(img)
            for i in range(mask.shape[0]):
                cmap = color_maps[i % len(color_maps)]
                plt.imshow(mask[i], cmap=cmap, alpha=0.5)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.title('Binarized Mask with Points')
            for i in range(mask.shape[0]):
                cmap = color_maps[i % len(color_maps)]
                plt.imshow(mask[i], cmap=cmap, alpha=0.5)
            if pos_points is not None:
                for point in pos_points:
                    plt.scatter(point[0], point[1], c='blue', s=100, label='Positive Point')
            if neg_points is not None:
                for point in neg_points:
                    plt.scatter(point[0], point[1], c='red', s=100, label='Negative Point')
            if boxes is not None and len(boxes) > 0:
                x0, y0, x1, y1 = boxes
                plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c='green', linewidth=2, label='Bounding Box')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def read_poly_coords(self, fname: str) -> Any:
        if fname.endswith('txt'):
            lines = open(fname, 'r').readlines()
            polygon: List[Tuple[float, float]] = []
            for line in lines:
                coord_str = line.strip().split(' ')
                polygon.append((float(coord_str[-1]), float(coord_str[0])))
        elif fname.endswith('json'):
            lbl_dict = json.load(open(fname, 'r'))
            polygon = {}
            for lbl in lbl_dict['shapes']:
                if lbl['label'] == 'disc':
                    polygon['od'] = [tuple(pt) for pt in lbl['points']]
                elif lbl['label'] == 'cup':
                    polygon['cup'] = [tuple(pt) for pt in lbl['points']]
                else:
                    polygon['discloc'] = [tuple(pt) for pt in lbl['points']]
        return polygon


if __name__ == "__main__":
    basepath = os.getcwd()
    data_dir = os.path.join(basepath, "datasets", "seg", "data", "thin_object_detection", "ThinObject5K")
    dataset = BaseSegmentationDataset(data_dir, image_folder="", mask_folder="", dataset_type='train', unified=False, transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, masks, pos_points, neg_points, boxes in dataloader:
        print(f"Images batch: {np.array(images).shape}, Masks batch: {np.array(masks).shape}, "
              f"Pos Points batch: {np.array(pos_points).shape}, Neg Points batch: {np.array(neg_points).shape}, "
              f"Boxes batch: {np.array(boxes).shape}")
