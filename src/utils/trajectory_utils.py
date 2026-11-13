from typing import List, Tuple
import numpy as np
import cv2


def calculate_person_trajectory(
    img: np.ndarray,
    track_p_c_xy_list: List[Tuple]
) -> Tuple:
    """Функция вычисляет траекторию пути человека.

    Args:
        img (np.ndarray): Исходное изображение.
        track_p_c_xy_list (List[Tuple]): список координат центра Bbox людей.

    Returns:
        pt1, pt2 (Tuple): Координаты траектории пути человека. Пример: (10, 20), (30, 40).
        k (float): Коэффициент наклона траектории пути человека.

    Raises:
        ValueError: Если количество точек в track_p_c_xy_list меньше 2.
    """
    if len(track_p_c_xy_list) < 2:
        raise ValueError("track_p_c_xy_list must contain at least two points")

    # Определение координат (x1, y1), (x2, y2)
    x1, y1 = track_p_c_xy_list[-2]
    x2, y2 = track_p_c_xy_list[-1]

    # Расчет коэффициентов наклона прямой и пересечения с прямой Oy
    k, b = np.polyfit([x1, x2], [y1, y2], 1)

    # Расчет конечных точек на основе размеров изображения
    if x2 > x1:
        x_end = img.shape[1]  # Правый край изображения
    else:
        x_end = 0  # Левый край изображения

    y_end = k * x_end + b  # Соответствующее значение y на выбранной границе x

    # Собираем координаты в точки для удобного использования
    pt1, pt2 = (int(x1), int(y1)), (int(x_end), int(y_end))

    return pt1, pt2, k


def calculate_object_trajectory(
    img: np.ndarray,
    track_c_xy_list: List[Tuple],
    radius: float,
) -> Tuple:
    """Функция вычисляет траекторию в направлении движения объекта.

    Args:
        img (np.ndarray): Исходное изображение.
        track_c_xy_list (List[Tuple]): Cписок координат центра Bbox объекта.
        radius (float): Радиус окружности для определения конечной точки траектории движения объекта. (в пикселях).

    Returns:
        pt1, pt2 (Tuple): Точки траектории движения объекта. Пример: (10, 20), (30, 40).
        k (float): Коэффициент наклона траектории движения объекта.

    Raises:
        ValueError: Если количество точек в track_c_xy_list меньше 5.
    """
    if len(track_c_xy_list) < 5:
        raise ValueError("The track_p_c_xy_list must contain at least two points")

    # Определение координат (x1, y1), (x2, y2)
    x1, y1 = track_c_xy_list[-5]
    x2, y2 = track_c_xy_list[-1]

    # Вычисление вектора направления от (x1, y1) до (x2, y2)
    direction_vector = np.array([x2 - x1, y2 - y1])
    direction_norm = np.linalg.norm(direction_vector)

    # Нормализация вектора направления
    if direction_norm != 0:
        direction_vector = direction_vector / direction_norm

    # Расчет конечной точки в направлении движения и радиуса
    x_end = x2 + direction_vector[0] * radius
    y_end = y2 + direction_vector[1] * radius

    # Ограничение конечной точки в пределах изображения
    x_end = np.clip(x_end, 0, img.shape[1] - 1)
    y_end = np.clip(y_end, 0, img.shape[0] - 1)

    # Собираем координаты в точки для удобного использования
    pt1, pt2 = (int(x2), int(y2)), (int(x_end), int(y_end))

    # Вычисление коэффициента наклона траектории перемещения объекта
    k = (y_end - y2) / (x_end - x2)

    return pt1, pt2, k


def calculate_wagon_trajectory(
    mask: np.ndarray,
    box: List[float],
) -> Tuple:
    """Функция вычисляет траекторию движения вагона по рельсе.

    Args:
        mask (np.ndarray): Маска (//полигон) рельсы, по которой движется вагон.
        box (List[float]): Bbox рельсы.

    Returns:
        pt1, pt2 (Tuple): Координаты траектории движения вагона по рельсе. Пример: (10, 20), (30, 40).
        k (float): Коэффициент наклона траектории движения вагона.
    """
    # Определение точек аппроксимированной прямой полигона (маски)
    line_rail = cv2.fitLine(mask, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    x1, y1 = int(line_rail[0] * 100 + line_rail[2]), int(line_rail[1] * 100 + line_rail[3])
    x2, y2 = int(-line_rail[0] * 100 + line_rail[2]), int(-line_rail[1] * 100 + line_rail[3])
    x_min = box[0]
    x_max = box[2]

    # Подбор коэффициентов прямой для построения прямой не только между 1-ой точкой и 2-ой, а более длинной
    coefs = np.polyfit([x1, x2], [y1, y2], 1)

    # Модель прямой
    model = np.poly1d(coefs)

    # Генерация данных для построения функции прямой
    x = np.linspace(x_min, x_max, 2)

    # Расчет значений функции
    y = model(x)

    # Собираем координаты в точки для удобного использования
    pt1, pt2 = (int(x[0]), int(y[0])), (int(x[-1]), int(y[-1]))

    # Коэффициент наклона траектории движения вагона по рельсе
    k = coefs[0]

    return pt1, pt2, k
