from typing import List, Tuple, Literal
import numpy as np
from shapely.geometry import box, Polygon
import cv2


def euclidean_distance(
    point1: Tuple[float],
    point2: Tuple[float],
) -> float:
    """
    The function calculates Euclido's distance between two points in two-dimensional space.

    Args:
        point1 (Tuple[float]): Coordinates of the first point (x1, y1).
        point2 (Tuple[float]): Coordinates of the second point (x2, y2).

    Returns:
        (float): Euclido is the distance between two points.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point2 - point1)


def intersection_rect_area(
    boxA: List[float],
    boxB: List[float],
) -> float:
    """The function calculates the intersection area of the two rectangles in the image.

    Args:
        boxA (List[float]): The coordinates of Bbox 1st object. [x1, y1, x2, y2]].
        boxB (List[float]): The coordinates of the Bbox 2nd object. [a1, b1, a2, b2]].

    Returns:
        interArea (float): Area of intersection of two rectangles.
    """
    # Definition (x, y) of the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculation of the area of the intersection rectangle
    interArea = max((xB - xA), 0) * max((yB - yA), 0)

    return interArea


def intersection_over_union(
    boxA: List[float],
    boxB: List[float],
) -> float:
    """The function calculates IoU: the degree of intersection of two Bboxes.

    Args:
        boxA (List[float]): The coordinates of Bbox 1st object. [x1, y1, x2, y2]].
        boxB (List[float]): The coordinates of the Bbox 2nd object. [a1, b1, a2, b2]].

    Returns:
        IoU (float): The extent of intersection of two Bboxes.
    """
    # Calculation of the area of the intersection rectangle
    interArea = intersection_rect_area(boxA, boxB)
    if interArea == 0:
        return 0
    # Calculation of the area of 1st Bbox
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    # Calculation of 2nd Bbox area
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # Calculation of IoU
    IoU = interArea / float(boxAArea + boxBArea - interArea)

    return IoU


def rect_area(box: List[float]) -> float:
    """The function calculates the Bbox area.

    Args:
        box (List[float]): The coordinates of the Bbox object. [x1, y1, x2, y2]].

    Returns:
        rectArea (float): Bbox area.
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_polygon_rect_area(
    poly_cords: np.ndarray,
    rect_cords: List[float],
) -> float:
    """The function calculates the area of intersection of the rectangle with the zone.

    Args:
        poly_cords (np.ndarray): The coordinates of the range.
        rect_cords (List[float]): The coordinates of the rectangle.

    Returns:
        (float): Area of intersection of rectangle with range.
    """
    return Polygon(poly_cords).intersection(box(rect_cords[0], rect_cords[1], rect_cords[2], rect_cords[3])).area


def intersection_2polygons_area(
    poly1_cords: np.ndarray,
    poly2_cords: np.ndarray,
) -> Tuple[float, Tuple[float, float]]:
    """The function calculates the intersection area of the 2 polygons.

    Args:
        poly1_cords (np.ndarray): The coordinates of the range.
        poly2_cords (np.ndarray): The coordinates of the range.

    Returns:
        (float): Area of intersection of the two zones.
        (tuple): Coordinates of the intersection centre.
    """
    if not (len(poly1_cords) >= 3 and len(poly2_cords) >= 3):
        raise ValueError("Each polygon must have at least three vertices")

    poly1 = Polygon(poly1_cords)
    poly2 = Polygon(poly2_cords)
    if not (poly1.is_valid and poly2.is_valid):
        return 0, (None, None)

    intesect = poly1.intersection(poly2)

    if intesect.is_empty:
        return 0, (None, None)

    centroid = intesect.centroid

    return intesect.area, (centroid.x, centroid.y)


def intersection_over_area_bbox(
    boxA: List[float],
    boxB: List[float],
    format: Literal['xyxy', 'xywh'] = 'xyxy',
) -> float:
    """Calculates the ratio of the intersection of two Bboxes to the area of the first Bbox.

    Args:
        boxA (List[float]): Coordinates of the first Bbox
            (format 'xyxy' - [x1,y1, x2,y2] or 'xywh' - [xc,yc,w,h]]).
        boxB (List[float]): Coordinates of the second Bbox (similar format as boxA)).
        format (str): Formation of input bounding boxes ('xyxy' or 'xywh')').

    Returns:
        float: Ratio of intersection area to first Bbox area.
    """
    def convert_to_xywh_if_needed(box: List[float]) -> List[float]:
        if format == 'xywh':
            return [
                box[0] - box[2] / 2,
                box[1] - box[3] / 2,
                box[0] + box[2] / 2,
                box[1] + box[3] / 2
            ]
        return box

    # We'll convert both sides of the box, if necessary.
    boxA_xyxy = convert_to_xywh_if_needed(boxA)
    boxB_xyxy = convert_to_xywh_if_needed(boxB)

    # Calculation of the area of the intersection rectangle
    interArea = intersection_rect_area(boxA_xyxy, boxB_xyxy)
    if interArea == 0:
        return 0

    # Calculation of the area of 1st Bbox
    boxAArea = rect_area(boxA_xyxy)

    return interArea / boxAArea


def intersection_over_area_mask(
    boxA: List[float],
    polygon: np.ndarray,
) -> float:
    """Calculates the ratio of Bbox crossing area to Bbox area.

    Args:
        boxA (List[float]): Coordinates of the first Bbox [x1, y1, x2, y2].
        polygon (np.ndarray): The coordinates of the range.

    Returns:
        (float): Ratio of intersection area to Bbox area.
    """
    # Calculation of the area of the intersection rectangle
    interArea = intersection_polygon_rect_area(polygon, boxA)

    # Calculation of the area of 1st Bbox
    boxAArea = rect_area(boxA)

    # If the area of the intersection is equal to the area of Bbox, that means the full intersection
    if interArea == boxAArea:
        return 1.0

    # Calculation of the ratio of intersection to area of 1st Bbox
    return interArea / boxAArea


def intersection_rect_mask_area(
    box: List[int],
    mask: np.ndarray,
) -> float:
    """
    The function calculates the intersection area of the rectangle with the mask.

    Args:
        box (List[int]): Bbox coordinates [x1, y1, x2, y2]].
        mask (np.ndarray): Binary mask of image, where objects are marked as non-zero pixels.

    Returns:
        interArea (float): Area of intersection of rectangle with mask areas.
    """
    # Create a rectangle mask with the same size as the original mask
    rect_mask = np.zeros_like(mask)
    cv2.rectangle(rect_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, thickness=cv2.FILLED)

    # Calculation of the intersection of the rectangle with the mask
    intersection = cv2.bitwise_and(rect_mask, mask)

    # Calculation of the intersection area
    interArea = np.count_nonzero(intersection)

    return interArea


def transform_mask_2_max_polygon(mask: np.ndarray) -> np.ndarray:
    """
    The function converts the mask into a range.

    Args:
        mask (np.ndarray): Binary image mask.

    Returns:
        mask_contour (np.ndarray): Mask as a contour.
    """
    # We find all the contours on the mask.
    mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # We find the contour with the largest area.
    contours_areas = [cv2.contourArea(contour) for contour in mask_contours]
    mask_contour = mask_contours[np.argmax(contours_areas)].reshape((-1, 2))

    return mask_contour


def filter_contour_points(
    contour: np.ndarray,
    center: Tuple[float],
    radius: int,
) -> np.ndarray:
    """
    Filter the contour points leaving only those within a specified radius from the centre point.

    Args:
        contour (np.ndarray): Contour points.
        center (tuple): Coordinates (x, y) of the central point.
        radius (int): Radio within which points should be left.

    Returns:
        (np.ndarray): Filtered contour points.
    """
    # Calculation of the distance from the centre to each point of the contour
    distances = np.sqrt((contour[:, 0] - center[0])**2 + (contour[:, 1] - center[1])**2)
    # Filtering points where distance is less than radius
    filtered_contour = contour[distances < radius]
    return filtered_contour


def fit_line(points: np.ndarray,) -> Tuple[float, np.ndarray]:
    """
    Function for building a straight line by points. Also returns the directional angle relative to the x axis

    Input parameters:
        points (np.ndarray): List of figure points in terms of x, y coordinates.

    Output parameters:
        angle_2_x (float): The angle of the line direction relative to the x axis.
        poly_coeffs (np.ndarray): Polynomial coefficients.
    """
    # Splitting of coordinates x and y
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Approximation of straight track points
    poly_coeffs = np.polyfit(x_coords, y_coords, 1)
    slope = poly_coeffs[0]

    # Optimized calculation of directional angle relative to x axis
    angle_2_x = np.arctan(slope) * 180 / np.pi
    return angle_2_x, poly_coeffs


def closest_point_on_contour(
    contour: List[Tuple[float, float]],
    target_point: Tuple[float, float],
) -> Tuple[float, float]:
    """Finds a point on the contour nearest to the target point.

    Arguments:
        (List[Tuple[float, float]]): List of points representing the outline.
        target_point (Tuple[float, float]): Target point for which the nearest point of the contour needs to be found.

    Returns:
        Tuple[float, float]: boundary point nearest to the target point.
    """
    min_distance = float('inf')
    closest_point = None

    for point in contour:
        distance = euclidean_distance(point, target_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point
