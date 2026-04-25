import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              jaccard_score, confusion_matrix)


def evaluate(y_test, y_pred, gt_masks, pred_masks):
    """
    Compute all required evaluation metrics.

    Calculates accuracy, precision, recall, IoU (Jaccard score
    on segmentation masks), and matching accuracy.

    Args:
        y_test: True classification labels.
        y_pred: Predicted classification labels.
        gt_masks: List of ground truth segmentation masks (flattened).
        pred_masks: List of predicted segmentation masks (flattened).

    Returns:
        Dictionary with keys: accuracy, precision, recall, iou,
        matching_accuracy, confusion_matrix.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # IoU (Jaccard) for segmentation mask evaluation
    # Flatten and binarize all masks, then compute average IoU
    iou_scores = []
    for gt, pred in zip(gt_masks, pred_masks):
        gt_flat = (gt.flatten() > 0).astype(int)
        pred_flat = (pred.flatten() > 0).astype(int)
        iou_scores.append(jaccard_score(gt_flat, pred_flat, average='binary'))
    iou = np.mean(iou_scores) if iou_scores else 0.0

    # Matching accuracy (same as accuracy in binary classification)
    matching_accuracy = accuracy

    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'matching_accuracy': matching_accuracy,
        'confusion_matrix': cm
    }
