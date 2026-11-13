import numpy as np

__all__ = ["soft_nms"]

def _iou(box, boxes):
    """IoU между box (4,) и boxes (N,4) (xyxy)"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def soft_nms(boxes: np.ndarray,
             scores: np.ndarray,
             classes: np.ndarray,
             iou_thr: float = 0.5,
             sigma: float = 0.5,
             score_thr: float = 0.001) -> np.ndarray:
    """Soft-NMS в варианте Gaussian. Возвращает индексы оставшихся боксов.

    boxes  – (N,4) xyxy,  scores – (N,), classes – (N,)
    """
    idxs = np.arange(boxes.shape[0])
    keep = []

    while scores.size:
        max_idx = np.argmax(scores)
        cur_box = boxes[max_idx]
        cur_cls = classes[max_idx]
        cur_id = idxs[max_idx]
        keep.append(cur_id)

        # убираем выбранный бокс из списка
        boxes = np.delete(boxes, max_idx, 0)
        classes = np.delete(classes, max_idx, 0)
        scores = np.delete(scores, max_idx, 0)
        idxs = np.delete(idxs, max_idx, 0)

        if not scores.size:
            break

        ious = _iou(cur_box, boxes)
        same_cls = classes == cur_cls
        decay = np.exp(-(ious ** 2) / sigma)
        scores[same_cls] = scores[same_cls] * decay[same_cls]

        # отбрасываем слабые
        keep_mask = scores > score_thr
        boxes = boxes[keep_mask]
        classes = classes[keep_mask]
        scores = scores[keep_mask]
        idxs = idxs[keep_mask]
    return np.array(keep, dtype=int) 