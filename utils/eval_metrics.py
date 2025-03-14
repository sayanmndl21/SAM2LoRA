import math
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, matthews_corrcoef
from skimage.morphology import binary_erosion
from medpy.metric.binary import hd95
from typing import Any, Dict, List

class SegmentationEVAL:
    def __init__(self) -> None:
        """
        Initialize the SegmentationEVAL class.
        """
        pass

    def get_iou(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the Intersection over Union (IoU) score.
        """
        intersection = np.logical_and(y_pred, y_truth).sum()
        union = np.logical_or(y_pred, y_truth).sum()
        return intersection.item() / union.item() if union != 0 else 0

    def get_auc(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the Area Under the ROC Curve (AUC) score.
        """
        return roc_auc_score(y_truth.flatten(), y_pred.flatten()).item()

    def get_dice(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the Dice coefficient.
        """
        intersection = 2 * np.logical_and(y_pred, y_truth).sum()
        total_sum = y_pred.sum() + y_truth.sum()
        return intersection.item() / total_sum.item() if total_sum != 0 else 0

    def get_hd95(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the 95th percentile of the Hausdorff Distance.
        """
        try:
            hd = hd95(y_pred.astype(np.uint8()), y_truth.astype(np.uint8()))
        except Exception:
            h, w = y_pred.shape
            hd = math.sqrt(h**2 + w**2)
        return hd

    def get_f1(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the F1 score.
        """
        return f1_score(y_truth.flatten(), y_pred.flatten()).item()

    def get_accuracy(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the accuracy.
        """
        return (y_pred == y_truth).sum().item() / np.prod(y_truth.shape).item()

    def get_sensitivity(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the sensitivity (recall).
        """
        tp = np.logical_and(y_pred, y_truth).sum().item()
        fn = np.logical_and(np.logical_not(y_pred), y_truth).sum().item()
        return tp / (tp + fn) if (tp + fn) != 0 else 0

    def get_specificity(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the specificity.
        """
        tn = np.logical_and(np.logical_not(y_pred), np.logical_not(y_truth)).sum().item()
        fp = np.logical_and(y_pred, np.logical_not(y_truth)).sum().item()
        return tn / (tn + fp) if (tn + fp) != 0 else 0

    def get_precision(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the precision.
        """
        tp = np.logical_and(y_pred, y_truth).sum().item()
        fp = np.logical_and(y_pred, np.logical_not(y_truth)).sum().item()
        return tp / (tp + fp) if (tp + fp) != 0 else 0

    def get_mcc(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the Matthews Correlation Coefficient (MCC).
        """
        return matthews_corrcoef(y_truth.flatten(), y_pred.flatten())

    def get_biou(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the boundary IoU using binary erosion.
        """
        y_pred_boundary = y_pred - binary_erosion(y_pred)
        y_truth_boundary = y_truth - binary_erosion(y_truth)
        intersection = np.logical_and(y_pred_boundary, y_truth_boundary).sum()
        union = np.logical_or(y_pred_boundary, y_truth_boundary).sum()
        return intersection.item() / union.item() if union != 0 else 0

    def get_assd(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Calculate the Average Symmetric Surface Distance (ASSD).
        """
        dist1 = directed_hausdorff(y_pred, y_truth)[0]
        dist2 = directed_hausdorff(y_truth, y_pred)[0]
        return (dist1 + dist2) / 2

    def get_eval(self, y_pred_prob: np.ndarray, y_truth: np.ndarray) -> Dict[str, float]:
        """
        Evaluate segmentation metrics based on prediction probabilities and ground truth.
        """
        if np.max(y_pred_prob) > 1:
            raise ValueError('Probability values greater than 1, please recheck.')
        y_pred = (y_pred_prob > 0.5).astype(np.uint8())
        y_truth = y_truth.astype(np.uint8)
        return {
            'iou': float(self.get_iou(y_pred, y_truth)),
            'auc': float(self.get_auc(y_pred_prob, y_truth)),
            'dice': float(self.get_dice(y_pred, y_truth)),
            'hd95': float(self.get_hd95(y_pred, y_truth)),
            'f1': float(self.get_f1(y_pred, y_truth)),
            'accuracy': float(self.get_accuracy(y_pred, y_truth)),
            'sensitivity': float(self.get_sensitivity(y_pred, y_truth)),
            'specificity': float(self.get_specificity(y_pred, y_truth)),
            'precision': float(self.get_precision(y_pred, y_truth)),
            'mcc': float(self.get_mcc(y_pred, y_truth)),
            'boundary_iou': float(self.get_biou(y_pred, y_truth)),
            'assd': float(self.get_assd(y_pred, y_truth))
        }

def calculate_batch_iou(gt_mask: Any, pred_mask: Any, epsilon: float = 1e-6) -> Any:
    """
    Calculate the batch IoU for a set of ground truth and predicted masks.
    Assumes that the masks are torch.Tensor objects.
    """
    # Convert to binary masks (assuming they are logits or probabilities, so we threshold at 0.5)
    gt_mask = (gt_mask > 0.5).float()
    pred_mask = (pred_mask > 0.5).float()

    # Flatten the masks to simplify intersection and union calculations
    B, N, H, W = gt_mask.shape 
    gt_mask = gt_mask.view(B, N, -1)  # Flatten per batch
    pred_mask = pred_mask.view(B, N, -1)

    # Intersection and union
    intersection = (gt_mask * pred_mask).sum(dim=2)  # Element-wise multiplication and sum over pixels
    union = gt_mask.sum(dim=2) + pred_mask.sum(dim=2) - intersection  # Union = gt + pred - intersection

    # Compute IoU with epsilon for numerical stability
    iou = intersection / (union + epsilon)
    
    return iou

def compute_iou(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the mean Intersection over Union (IoU) from the confusion matrix.
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Compute mean IoU
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

def calculate_dice(y_pred: np.ndarray, y_truth: np.ndarray) -> float:
    """
    Calculate the Dice loss (1 - Dice coefficient) between the prediction and ground truth.
    """
    return 1 - distance.dice(y_truth.flatten(), y_pred.flatten())

def calculate_mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate the mean of a list of metric dictionaries.
    """
    # Initialize a dictionary to accumulate sums of each metric
    metrics_sum: Dict[str, float] = {}
    num_data_points = len(metrics_list)
    
    # Sum up each metric across all data points
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0
            metrics_sum[key] += value
    
    # Calculate the mean for each metric
    mean_metrics = {key: float(metrics_sum[key] / num_data_points) for key in metrics_sum}
    
    return mean_metrics
