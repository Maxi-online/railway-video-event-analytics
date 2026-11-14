from typing import Tuple
import numpy as np
import cv2


def sigmoid(x):
    """Apply the logistic sigmoid elementwise: 1 / (1 + exp(-x)).

    Args:
        x: A scalar or NumPy array of logits.

    Returns:
        A scalar or array of sigmoid values with the same shape as x.
    """
    return 1 / (1 + np.exp(-x))  # Return the sigmoid function result


def create_binary_mask(
        prob_mask: np.ndarray,
        desired_size: Tuple[int, int],
        threshold: float = 0.4,
) -> np.ndarray:
    """
    Function to Binarize the mask according to the threshold and change its size to desired_size.

    Input parameters:
        prob_mask (np.ndarray): Probability mask.
        desired_size (Tuple[int, int]): The desired size of the mask.
        threshold (float): The threshold for the binarization of the mask.

    Output parameters:
        (np.ndarray: Binarized mask after size change.
    """
    mask = (prob_mask > threshold).astype(np.uint8)
    mask = cv2.resize(mask, desired_size, interpolation=cv2.INTER_NEAREST)

    return mask
