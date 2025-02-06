import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

# Some functions taken / built from:
# Sensitivity of Slot-Based Object-Centric Models to their Number of Slots
# https://arxiv.org/pdf/2305.18890
# the functions: get_ar_metrics, get_ari, get_arp, get_arr, _check_ar_quantities, _compute_quantities

def compute_pixelwise_accuracy(pred_mask, true_mask):
    """
    Compute pixel-wise accuracy for each element in the batch without explicit looping.
    
    Args:
    pred_mask, true_mask: (B, H, W) with integer labels in [0..n_classes-1]
    
    Returns:
    A tensor of pixel-wise accuracies for each item in the batch.
    """
    # Ensure that predictions and ground truth have the same shape
    assert pred_mask.shape == true_mask.shape, "Shape mismatch between prediction and ground truth"

    # Calculate element-wise correctness
    correct_pixels = (pred_mask == true_mask).float()  # (B, H, W)

    # Calculate total pixels per batch element
    total_pixels_per_batch = true_mask.size(1) * true_mask.size(2)  # H * W

    # Sum correct predictions over each (H, W) for each batch
    correct_per_batch = correct_pixels.view(pred_mask.size(0), -1).sum(dim=1)  # (B,)

    # Compute pixel-wise accuracy per batch element
    pixel_wise_accuracies = correct_per_batch / total_pixels_per_batch  # (B,)

    return pixel_wise_accuracies


def compute_accuracy_per_class(pred_mask, true_mask, n_classes=11):
    """
    pred_mask, true_mask: (B, H, W) with integer labels in [0..n_classes-1]
    Returns: correct_per_class, total_per_class (each of length n_classes)
    """
    correct_per_class = [0]*n_classes
    total_per_class = [0]*n_classes
    
    # Flatten
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    for c in range(n_classes):
        # Boolean mask of ground-truth pixels for class c
        gt_c = (true_mask == c)
        total_per_class[c] = gt_c.sum().item()
        
        # Among those, how many predicted c?
        correct_per_class[c] = ((pred_mask == c) & gt_c).sum().item()

    return correct_per_class, total_per_class


def compute_iou(masks1, masks2):
    """
    Calculate IoU for batches of segmentation masks, ignoring specific values.
    
    Parameters:
        masks1 (torch.Tensor): A batch of ground truth masks with shape (batch_size, height, width).
        masks2 (torch.Tensor): A batch of predicted masks with shape (batch_size, height, width).
        ignore_value (int): Specifies the value in masks1 to be ignored.
    
    Returns:
        torch.Tensor: A tensor containing IoU scores for each item in the batch.
    """
    batch_size = masks1.shape[0]
    
    # Flatten the height and width dimensions for vectorized operations
    masks1 = masks1.view(batch_size, -1)
    masks2 = masks2.view(batch_size, -1)
    
    # Compute intersection and union where the valid mask is True
    intersection = (masks1 & masks2).sum(dim=1).float()
    union = (masks1 | masks2).sum(dim=1).float()

    # Compute IoU, ignoring invalid areas
    iou = torch.where(union > 0, intersection / union, torch.tensor(float('nan')))
    
    return iou

def get_ar_metrics(x, y):
    m_squared, m, the_P, the_Q, expected_m_squared = _compute_quantities(y, x)
    #ari = get_ari(x, y, m_squared, m, the_P, the_Q, expected_m_squared)
    arp = get_arp(x, y, m_squared, m, the_P, the_Q, expected_m_squared)
    arr = get_arr(x, y, m_squared, m, the_P, the_Q, expected_m_squared)
    ari = 2 / ((1 /  arp) + (1 / arr))
    return ari, arp, arr
    
def get_ari(x, y, m_squared=None, m=None, the_P=None, the_Q=None, expected_m_squared=None):
    if _check_ar_quantities(m_squared, m, the_P, the_Q, expected_m_squared):
        m_squared, m, the_P, the_Q, expected_m_squared = _compute_quantities(y, x)
    return (m_squared - expected_m_squared) / (the_P + the_Q + 2 * m - expected_m_squared)

def get_arp(x, y, m_squared=None, m=None, the_P=None, the_Q=None, expected_m_squared=None):
    if _check_ar_quantities(m_squared, m, the_P, the_Q, expected_m_squared):
        m_squared, m, the_P, the_Q, expected_m_squared = _compute_quantities(y, x)
    return (m_squared - expected_m_squared) / (the_Q + m - expected_m_squared)

def get_arr(x, y, m_squared=None, m=None, the_P=None, the_Q=None, expected_m_squared=None):
    if _check_ar_quantities(m_squared, m, the_P, the_Q, expected_m_squared):
        m_squared, m, the_P, the_Q, expected_m_squared = _compute_quantities(y, x)
    return (m_squared - expected_m_squared) / (the_P + m - expected_m_squared)

def _check_ar_quantities(m_squared, m, P, Q, expected_m_squared):
    return m_squared is None or m is None or P is None or Q is None \
        or expected_m_squared is None

def _compute_quantities(segmentation_gt: torch.Tensor, segmentation_pred: torch.Tensor):
    """ Compute the (Adjusted) Rand Precision/Recall.
    Args:
    segmentation_gt: Int tensor with shape (batch_size, height, width) containing the
        ground-truth segmentations.
    segmentation_pred: Int tensor with shape (batch_size, height, width) containing the
        predicted segmentations.
    mode: Either "precision" or "recall" depending on which metric shall be computed.
    adjusted: Return values for adjusted or non-adjusted metric.
    Returns:
    Float tensor with shape (batch_size), containing the (Adjusted) Rand
        Precision/Recall per sample.
    """
    
    # MASK OUT THE VOID INDEX IN PASCAL
    valid_mask = (segmentation_gt != 255) & (segmentation_pred != 255)
    segmentation_gt = segmentation_gt[valid_mask]
    segmentation_pred = segmentation_pred[valid_mask]

    # Proceed with only the valid classes
    if segmentation_gt.numel() == 0 or segmentation_pred.numel() == 0:
        # If masked out all, return zeros or handle it as special case
        return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

    max_classes = max(segmentation_gt.max(), segmentation_pred.max()) + 1
    oh_segmentation_gt = F.one_hot(segmentation_gt, max_classes)
    oh_segmentation_pred = F.one_hot(segmentation_pred, max_classes)
    coincidence = torch.einsum("ij,ik->jk", oh_segmentation_gt.float(), oh_segmentation_pred.float())
    coincidence_gt = coincidence.sum(-1)
    coincidence_pred = coincidence.sum(-2)

    m_squared = torch.sum(coincidence ** 2)
    m = torch.sum(coincidence)

    P = torch.sum(coincidence_gt * (coincidence_gt - 1))
    Q = torch.sum(coincidence_pred * (coincidence_pred - 1))

    expected_m_squared = (P + m) * (Q + m) / (m * (m - 2)) + (m ** 2 - Q - P - 2 * m) / (m - 1)

    return m_squared, m, P, Q, expected_m_squared

    max_classes = max(segmentation_gt.max(), segmentation_pred.max()) + 1
    oh_segmentation_gt = F.one_hot(segmentation_gt, max_classes) # (batch, height, width, j)
    oh_segmentation_pred = F.one_hot(segmentation_pred, max_classes) # (batch, height, width, i)
    coincidence = torch.einsum("bhwj,bhwi->bji",
                               oh_segmentation_gt.type(torch.FloatTensor),
                               oh_segmentation_pred.type(torch.FloatTensor)) # (batch, j, i)
    coincidence_gt = coincidence.sum(-1) # for each gt class, how many pred ones match? (batch, j)
    coincidence_pred = coincidence.sum(-2) # for each pred class, how many gt ones match? (batch, i)
    
    # (batch, )
    m_squared = torch.sum(coincidence**2, (1, 2))
    m = torch.sum(coincidence, (1, 2))

    # How many pairs of pixels have the same label assigned in ground-truth segmentation.
    P = torch.sum(coincidence_gt * (coincidence_gt - 1), -1) # (batch, )

    # How many pairs of pixels have the same label assigned in predicted segmentation.
    Q = torch.sum(coincidence_pred * (coincidence_pred - 1), -1) # (batch, )

    # (batch, )
    expected_m_squared = (P + m) * (Q + m) / (m * (m - 2)) + (m**2 - Q - P -2 * m) / (m - 1)

    return m_squared, m, P, Q, expected_m_squared