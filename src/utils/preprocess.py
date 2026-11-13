import cv2
import numpy as np

# Переиспользуем один объект CLAHE на все кадры
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_frame(img: np.ndarray) -> np.ndarray:
    """Усиливает контраст кадра CLAHE-алгоритмом.
    Работает только для BGR-кадров (H×W×3). Если картинка
    чёрно-белая или имеет другой формат, возвращается без изменений.
    Без обучения повышает различимость мелких деталей и улучшает Recall.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img 