"""Pure danger-zone predicate shared by the live task and offline replay."""
import numpy as np
from .geometry_utils import intersection_rect_area, intersection_polygon_rect_area, rect_area

def feet_in_zone(feet_box, zone, threshold=0.5):
    if not 0 <= threshold <= 1: raise ValueError("threshold must be between 0 and 1")
    area=rect_area(feet_box)
    if area <= 0: return False
    coords=np.asarray(zone)
    if coords.ndim==1 and coords.shape==(4,): overlap=intersection_rect_area(feet_box,coords)
    elif coords.ndim==2 and coords.shape[0]>=3 and coords.shape[1]==2:
        overlap=intersection_polygon_rect_area(coords,feet_box)
    else: raise ValueError("Expected an xyxy rectangle or polygon vertices")
    return bool(overlap/area > threshold)
