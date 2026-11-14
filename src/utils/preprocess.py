import cv2
import numpy as np

# Reuse one CLAHE object to all images
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_frame(img: np.ndarray) -> np.ndarray:
    """It increases the contrast with the CLAHE-algorithm.
    Works for BGR only (H×W×3)). If image
    black and white or in another format, returned unchanged.
    Without learning, improves the visibility of small details and improves Recall.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img 