from typing import List, Tuple, Literal
import numpy as np
from shapely.geometry import box, Polygon
import cv2


def euclidean_distance(
    point1: Tuple[float],
    point2: Tuple[float],
) -> float:
    """
    Функция рассчитывает Евклидово расстояние между двумя точками в двумерном пространстве.

    Args:
        point1 (Tuple[float]): Координаты первой точки (x1, y1).
        point2 (Tuple[float]): Координаты второй точки (x2, y2).

    Returns:
        (float): Евклидово расстояние между двумя точками.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point2 - point1)


def intersection_rect_area(
    boxA: List[float],
    boxB: List[float],
) -> float:
    """Функция рассчитывает площадь пересечения двух прямоугольников на изображении.

    Args:
        boxA (List[float]): Координаты Bbox 1-го объекта. [x1, y1, x2, y2].
        boxB (List[float]): Координаты Bbox 2-го объекта. [a1, b1, a2, b2].

    Returns:
        interArea (float): Площадь пересечения двух прямоугольников.
    """
    # Определение (x, y) координат прямоугольника пересечения
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Расчет площади прямоугольника пересечения
    interArea = max((xB - xA), 0) * max((yB - yA), 0)

    return interArea


def intersection_over_union(
    boxA: List[float],
    boxB: List[float],
) -> float:
    """Функция рассчитывает IoU: степень пересечения двух Bboxes.

    Args:
        boxA (List[float]): Координаты Bbox 1-го объекта. [x1, y1, x2, y2].
        boxB (List[float]): Координаты Bbox 2-го объекта. [a1, b1, a2, b2].

    Returns:
        IoU (float): Степень пересечения двух Bboxes.
    """
    # Расчет площади прямоугольника пересечения
    interArea = intersection_rect_area(boxA, boxB)
    if interArea == 0:
        return 0
    # Расчет площади 1-го Bbox
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    # Расчет площади 2-го Bbox
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # Расчет IoU
    IoU = interArea / float(boxAArea + boxBArea - interArea)

    return IoU


def rect_area(box: List[float]) -> float:
    """Функция рассчитывает площадь Bbox.

    Args:
        box (List[float]): Координаты Bbox объекта. [x1, y1, x2, y2].

    Returns:
        rectArea (float): площадь Bbox.
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_polygon_rect_area(
    poly_cords: np.ndarray,
    rect_cords: List[float],
) -> float:
    """Функция рассчитывает площадь пересечения прямоугольника с полигоном.

    Args:
        poly_cords (np.ndarray): Координаты полигона.
        rect_cords (List[float]): Координаты прямоугольника.

    Returns:
        (float): Площадь пересечения прямоугольника с полигоном.
    """
    return Polygon(poly_cords).intersection(box(rect_cords[0], rect_cords[1], rect_cords[2], rect_cords[3])).area


def intersection_2polygons_area(
    poly1_cords: np.ndarray,
    poly2_cords: np.ndarray,
) -> Tuple[float, Tuple[float, float]]:
    """Функция рассчитывает площадь пересечения 2-х полигонов.

    Args:
        poly1_cords (np.ndarray): Координаты полигона.
        poly2_cords (np.ndarray): Координаты полигона.

    Returns:
        (float): Площадь пересечения двух полигонов.
        (tuple): Координаты центра пересечения.
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
    """Вычисляет отношение площади пересечения двух Bboxes к площади первого Bbox.

    Args:
        boxA (List[float]): Координаты первого Bbox
            (формат 'xyxy' - [x1,y1,x2,y2] или 'xywh' - [xc,yc,w,h]).
        boxB (List[float]): Координаты второго Bbox (аналогичный формат как boxA).
        format (str): Формат входных bounding boxes ('xyxy' или 'xywh').

    Returns:
        float: Отношение площади пересечения к площади первого Bbox.
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

    # Конвертируем оба bounding box при необходимости
    boxA_xyxy = convert_to_xywh_if_needed(boxA)
    boxB_xyxy = convert_to_xywh_if_needed(boxB)

    # Расчет площади прямоугольника пересечения
    interArea = intersection_rect_area(boxA_xyxy, boxB_xyxy)
    if interArea == 0:
        return 0

    # Расчет площади 1-го Bbox
    boxAArea = rect_area(boxA_xyxy)

    return interArea / boxAArea


def intersection_over_area_mask(
    boxA: List[float],
    polygon: np.ndarray,
) -> float:
    """Вычисляет отношение площади пересечения Bbox и полигона к площади Bbox.

    Args:
        boxA (List[float]): Координаты первого Bbox [x1, y1, x2, y2].
        polygon (np.ndarray): Координаты полигона.

    Returns:
        (float): Отношение площади пересечения к площади Bbox.
    """
    # Расчет площади прямоугольника пересечения
    interArea = intersection_polygon_rect_area(polygon, boxA)

    # Расчет площади 1-го Bbox
    boxAArea = rect_area(boxA)

    # Если площадь пересечения равна площади Bbox, значит полное пересечение
    if interArea == boxAArea:
        return 1.0

    # Расчет отношения площади пересечения к площади 1-го Bbox
    return interArea / boxAArea


def intersection_rect_mask_area(
    box: List[int],
    mask: np.ndarray,
) -> float:
    """
    Функция рассчитывает площадь пересечения прямоугольника с маской.

    Args:
        box (List[int]): Координаты Bbox объекта [x1, y1, x2, y2].
        mask (np.ndarray): Бинарная маска изображения, где объекты обозначены как ненулевые пиксели.

    Returns:
        interArea (float): Площадь пересечения прямоугольника с областями маски.
    """
    # Создаем маску прямоугольника с теми же размерами, что и исходная маска
    rect_mask = np.zeros_like(mask)
    cv2.rectangle(rect_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, thickness=cv2.FILLED)

    # Вычисление пересечения прямоугольника с маской
    intersection = cv2.bitwise_and(rect_mask, mask)

    # Расчет площади пересечения
    interArea = np.count_nonzero(intersection)

    return interArea


def transform_mask_2_max_polygon(mask: np.ndarray) -> np.ndarray:
    """
    Функция преобразует маску в полигон.

    Args:
        mask (np.ndarray): Бинарная маска изображения.

    Returns:
        mask_contour (np.ndarray): Маска в виде контура.
    """
    # Находим все контуры на маске
    mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Находим контур с наибольшей площадью
    contours_areas = [cv2.contourArea(contour) for contour in mask_contours]
    mask_contour = mask_contours[np.argmax(contours_areas)].reshape((-1, 2))

    return mask_contour


def filter_contour_points(
    contour: np.ndarray,
    center: Tuple[float],
    radius: int,
) -> np.ndarray:
    """
    Фильтрует точки контура, оставляя только те, которые находятся в заданном радиусе от центральной точки.

    Args:
        contour (np.ndarray): Точки контура.
        center (tuple): Координаты (x, y) центральной точки.
        radius (int): Радиус, в пределах которого нужно оставить точки.

    Returns:
        (np.ndarray): Отфильтрованные точки контура.
    """
    # Расчет расстояний от центра до каждой точки контура
    distances = np.sqrt((contour[:, 0] - center[0])**2 + (contour[:, 1] - center[1])**2)
    # Фильтрация точек, где расстояние меньше радиуса
    filtered_contour = contour[distances < radius]
    return filtered_contour


def fit_line(points: np.ndarray,) -> Tuple[float, np.ndarray]:
    """
    Функция для построения прямой линии по точкам. Также возвращает угол направления относительно оси x

    Входные параметры:
        points (np.ndarray): Список точек фигуры в виде координат x, y.

    Выходные параметры:
        angle_2_x (float): Угол направления линии относительно оси x.
        poly_coeffs (np.ndarray): Коэффициенты полинома.
    """
    # Разделение координат x и y
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Аппроксимация прямой по точкам трека
    poly_coeffs = np.polyfit(x_coords, y_coords, 1)
    slope = poly_coeffs[0]

    # Оптимизированный расчет угла направления относительно оси x
    angle_2_x = np.arctan(slope) * 180 / np.pi
    return angle_2_x, poly_coeffs


def closest_point_on_contour(
    contour: List[Tuple[float, float]],
    target_point: Tuple[float, float],
) -> Tuple[float, float]:
    """Находит точку на контуре, ближайшую к целевой точке.

    Аргументы:
        contour (List[Tuple[float, float]]): Список точек, представляющих контур.
        target_point (Tuple[float, float]): Целевая точка, для которой нужно найти ближайшую точку контура.

    Возвращает:
        Tuple[float, float]: Точка контура, ближайшая к целевой точке.
    """
    min_distance = float('inf')
    closest_point = None

    for point in contour:
        distance = euclidean_distance(point, target_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point
