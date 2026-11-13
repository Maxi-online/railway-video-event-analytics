from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from src.services.tracker.tracker import TrackerLWT
from src.utils.draw_utils import plot_contour
from src.utils.geometry_utils import (
    fit_line, intersection_2polygons_area,
    euclidean_distance, closest_point_on_contour,
    intersection_over_area_mask
)


def approximate_rail_line(
    polygon: np.ndarray,
    img_width: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Аппроксимирует линию рельса по полигону и возвращает угловой коэффициент (k) и точки линии.

    Args:
        polygon (np.ndarray): Список точек полигона [(x1, y1), (x2, y2), ...].
        img_width (int): Ширина изображения (для определения диапазона x).

    Returns:
        angle_2_x (float): Угол направления линии относительно оси x.
        x_range (np.ndarray): Массив x-координат аппроксимированной линии.
        y_range (np.ndarray): Массив y-координат аппроксимированной линии.
    """
    if len(polygon) > 2:
        points = np.array(polygon, dtype=np.float32)
        angle_2_x, poly_coeffs = fit_line(points)

        poly_func = np.poly1d(poly_coeffs)
        x_range = np.linspace(0, img_width - 1, img_width)
        y_range = poly_func(x_range)

        # Ограничиваем точки в пределах изображения
        valid_idx = np.where((y_range >= 0) & (y_range < img_width))
        x_range = x_range[valid_idx]
        y_range = y_range[valid_idx]

        return angle_2_x, x_range, y_range


#==========================================================#
#       Вспомогательные функции для логики task_4_2        #
#==========================================================#

def get_moving_wagons_rails(
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    speed_threshold_lwt: float,
    thrld_dist_w_r: float,
    show_debug: bool,
    img: np.ndarray,
) -> Dict[int, int]:
    """
    Функция определяет движущиеся вагоны и пересечение их с рельсами.
    
    Args:
        tracker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        results_r (Dict[str, np.ndarray]): Результаты сегментации рельсов.
        speed_threshold_lwt (float): Пороговое значение скорости движения вагона.
        thrld_dist_w_r (float): Доля от высоты вагона для определения пересечения с рельсами.
        show_debug (bool): Флаг для отображения отладочной информации.
        img (np.ndarray): Изображение для отображения отладочной информации.

    Returns:
        wagons_rails_link (Dict[int, int]): Словарь, связывающий индексы рельсов и индексы вагонов.
    """
    # Словарь для хранения связей рельс-вагон
    wagons_rails_link = {}
    
    # Перебираем все результаты трекинга вагонов
    for index_lwt, (_, _, mask_lwt, cls_lwt, _, track_id_lwt) in enumerate(zip(*tracker_LWT.results.values())):
        # Проверяем, движется ли вагон
        if not is_wagon_moving(tracker_LWT, cls_lwt, track_id_lwt, speed_threshold_lwt):
            continue

        # Получение ориентированного bbox и высоты вагона
        o_bbox, height_lwt_or = get_obbox_and_wagon_height(mask_lwt)
        # Вычисление вектора смещения вагона
        displacement_vector = get_displacement_vector(tracker_LWT, cls_lwt, track_id_lwt)
        
        # Определение пересекающих рельсов и угловых точек вагона
        indexs_r_list, corner_points_list = get_intersecting_rails(mask_lwt, results_r, displacement_vector, o_bbox, height_lwt_or, thrld_dist_w_r)
        
        # Если флаг show_debug установлен, отображаем отладочную информацию
        if show_debug:
            draw_debug_info(img, corner_points_list, indexs_r_list, results_r)

        # Сопоставляем индексы рельсов и вагонов
        for index_r in indexs_r_list:
            wagons_rails_link[index_r] = index_lwt

    return wagons_rails_link


def get_wagons_rails(
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    thrld_dist_w_r: float,
    show_debug: bool,
    img: np.ndarray,
) -> Tuple[Dict, Dict]:
    """
    Функция определяет вагоны и пересечение их с рельсами.

    Args:
        tracker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        results_r (Dict[str, np.ndarray]): Результаты сегментации рельсов.
        thrld_dist_w_r (float): Доля от высоты вагона для определения пересечения с рельсами.
        show_debug (bool): Флаг для отображения отладочной информации.
        img (np.ndarray): Изображение для отображения отладочной информации.

    Returns:
        wagons_rails_link (Dict[int, int]): Словарь, связывающий индексы рельсов и индексы вагонов.
    """
    # Словарь для хранения связей рельс-вагон
    wagons_rails_link = {}

    # Словарь для хранения угловых точек вагона для каждой рельсы по ее индексу
    indexR_cornerPoints = {}

    # Перебираем все результаты трекинга вагонов
    for index_lwt, (_, _, mask_lwt, cls_lwt, _, track_id_lwt) in enumerate(zip(*tracker_LWT.results.values())):

        # Получение ориентированного bbox и высоты вагона
        o_bbox, height_lwt_or = get_obbox_and_wagon_height(mask_lwt)
        # Вычисление вектора смещения вагона
        displacement_vector = get_displacement_vector(tracker_LWT, cls_lwt, track_id_lwt)

        # Определение пересекающих рельсов и угловых точек вагона
        indexs_r_list, corner_points_list = get_intersecting_rails(mask_lwt, results_r, displacement_vector, o_bbox,
                                                                   height_lwt_or, thrld_dist_w_r)

        # Если флаг show_debug установлен, отображаем отладочную информацию
        if show_debug:
            draw_debug_info(img, corner_points_list, indexs_r_list, results_r)

        # Сопоставляем индексы рельсов и вагонов
        for index_r in indexs_r_list:
            wagons_rails_link[index_r] = index_lwt

        # Сопоставляем индексы рельсов и угловых точек вагона
        for i in range(len(indexs_r_list)):
            # Индекс рельсы
            index_r = indexs_r_list[i]
            # Координаты угловых точек вагона, находящегося на рельсе
            corner_points = corner_points_list[i]
            indexR_cornerPoints[index_r] = corner_points

    return wagons_rails_link, indexR_cornerPoints


def is_wagon_moving(
    tracker_LWT: TrackerLWT,
    cls_lwt: int,
    track_id_lwt: int,
    move_velocity_lwt: float,
) -> bool:
    """
    Функция определяет, движется ли вагон на основе его текущей скорости и заданного порогового значения.

    Args:
        tracker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        cls_lwt (int): Класс вагона/локомотива/цистерны.
        track_id_lwt (int): Идентификатор трека вагона.
        move_velocity_lwt (float): Пороговое значение скорости движения вагона.

    Returns:
        bool: Возвращает True, если вагон движется, и False в противном случае.
    """
    # Формируем ключ для доступа к данным о скорости вагона в истории трекинга
    track_key_lwt_vel_ema = f"{tracker_LWT.class_names[cls_lwt]}_{track_id_lwt}_ema"
    
    # Извлекаем список скоростей вагона из истории трекинга
    track_lwt_ema_list = list(tracker_LWT.track_history.get(track_key_lwt_vel_ema, []))
    
    # Проверяем, есть ли в списке хотя бы одна запись о скорости
    if len(track_lwt_ema_list) > 0:
        # Берем последнюю записанную скорость
        current_lwt_vel = track_lwt_ema_list[-1]
        # Если скорость существует и больше порогового значения, возвращаем True
        return current_lwt_vel and current_lwt_vel > move_velocity_lwt
    else:
        # Если скорость не существует или меньше порогового значения, возвращаем False
        return False


def get_obbox_and_wagon_height(mask_lwt: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Функция находит ориентированный ограничивающий прямоугольник (OBB) и высоту вагона по маске.

    Args:
        mask_lwt (np.ndarray): Маска вагона в виде бинарного массива.

    Returns:
        box_oriented_lwt (np.ndarray): Координаты углов ориентированного прямоугольника, содержащего вагон.
        min(w_o, h_o) (float): Минимальная сторона ориентированного прямоугольника, представляющая высоту вагона.
    """
    # Создание ориентированного ограничивающего прямоугольника вокруг маски вагона
    rect = cv2.minAreaRect(mask_lwt)
    # Извлечение параметров прямоугольника
    (_, _), (w_o, h_o), _ = rect
    # Преобразование прямоугольника в координаты его углов
    box_oriented_lwt = cv2.boxPoints(rect)
    # Преобразование координат углов к целочисленному типу
    box_oriented_lwt = np.int8(box_oriented_lwt)
    # Возвращаем координаты углов ориентированного прямоугольника и минимальную сторону как высоту вагона
    return box_oriented_lwt, min(w_o, h_o)


def get_displacement_vector(
    tracker_LWT: TrackerLWT,
    cls_lwt: int,
    track_id_lwt: int,
) -> np.ndarray:
    """
    Функция для вычисления вектора смещения вагона на основе истории его трека.

    Args:
        tracker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов.
        cls_lwt (int): Класс вагона, локомотива или цистерны.
        track_id_lwt (int): Идентификатор трека вагона.

    Returns:
        np.ndarray: Вектор смещения вагона.
    """
    # Формируем ключ для доступа к историческим данным о центре вагона
    track_key_lwt_center_xy = f"{tracker_LWT.class_names[cls_lwt]}_{track_id_lwt}_c_xy"
    
    # Получаем список координат центра вагона из истории трекинга
    track_lwt_c_xy_list = np.array(list(tracker_LWT.track_history.get(track_key_lwt_center_xy, [])))
    
    # Если длинна истории трекинга меньше заранее установленного значения track_history_len
    if len(track_lwt_c_xy_list) < tracker_LWT.track_history_len:
        # Берём первую точку в качестве начальной
        start_point = track_lwt_c_xy_list[0]
    else:
        # Иначе, берем точку, удалённую на track_history_len кадров назад
        start_point = track_lwt_c_xy_list[-tracker_LWT.track_history_len]
    
    # Возвращаем вектор смещения как разницу между начальной и последней точкой
    return start_point - track_lwt_c_xy_list[-1]


def get_intersecting_rails(
    mask_lwt: np.ndarray,
    results_r: Dict[str, np.ndarray],
    displacement_vector: np.ndarray,
    o_bbox: np.ndarray,
    height_lwt_or: float,
    thrld_dist_w_r: float,
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Функция для определения пересекающихся рельсов с вагоном на основе масок и положения вагона.

    Args:
        mask_lwt (np.ndarray): Маска вагона в виде бинарного массива, представляющего вагон.
        results_r (Dict[str, np.ndarray]): Словарь с результатами сегментации рельсов, содержащий маски рельсов.
        displacement_vector (np.ndarray): Вектор смещения вагона, основанный на его движении.
        o_bbox (np.ndarray): Координаты углов ориентированного ограничивающего прямоугольника (OBB) вагона.
        height_lwt_or (float): Высота вагона, определяемая минимальной стороной ориентированного прямоугольника.
        thrld_dist_w_r (float): Пороговое значение для определения пересечения вагона и рельсов в зависимости от их расстояния.

    Returns:
        Tuple[List[int], List[np.ndarray]]: Список индексов масок рельсов, пересекающихся с вагоном, и список угловых точек вагона.
    """
    indexs_r_list = []  # Список для хранения индексов рельсов, пересекающихся с вагоном
    corner_points_list = []  # Список для хранения угловых точек вагона

    # Перебор всех масок рельсов и их индексов
    for index_r, r_mask in enumerate(results_r['masks']):
        # Вычисление площади пересечения и центра пересечения рельсов и вагона
        intesect_area, intesect_centroid = intersection_2polygons_area(r_mask, mask_lwt)
        # Если площадь пересечения больше нуля, добавляем рельсы в список пересекающихся
        if intesect_area > 0:
            # Получение угловых точек вагона на основе маски и вектора смещения
            corner_points = get_corner_points(mask_lwt, o_bbox, displacement_vector)
            corner_points_list.append(corner_points)
            # Проверка, находятся ли угловые точки вагона в пределах расстояния thrld_dist_w_r от центра пересечения
            if any(euclidean_distance(intesect_centroid, point) < height_lwt_or * thrld_dist_w_r for point in corner_points):
                indexs_r_list.append(index_r)  # Добавление индекса рельсов в список пересекающихся

    return indexs_r_list, corner_points_list


def get_corner_points(
    mask_lwt: np.ndarray,
    o_bbox: np.ndarray,
    displacement_vector: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Функция для определения угловых точек вагона рядом с рельсами на основе его
    текущего смещения и ориентированного bbox.

    Args:
        mask_lwt (np.ndarray): Маска вагона в виде бинарного массива.
        o_bbox (np.ndarray): Ориентированный ограничивающий прямоугольник (OBB) вагона.
        displacement_vector (np.ndarray): Вектор смещения вагона.

    Returns:
        List[Tuple[int, int]]: Список координат угловых точек вагона рядом с рельсами.
    """
    # Если вагон движется влево вниз или вправо вверх
    if (displacement_vector[1] < 0 < displacement_vector[0]) or (displacement_vector[0] < 0 < displacement_vector[1]):
        # Определение нижней левой и верхней правой угловых точек вагона
        lower_left_box = max(o_bbox, key=lambda point: (point[1], -point[0]))
        upper_right_box = max(o_bbox, key=lambda point: (point[0], point[1]))
        return [
            # Для повышения точности определяются ближайшие точки на контуре вагона к выбранным угловым точкам
            closest_point_on_contour(mask_lwt, lower_left_box),
            closest_point_on_contour(mask_lwt, upper_right_box),
        ]
    # Если вагон движется вправо вниз или влево вверх
    elif (displacement_vector[0] > 0 and displacement_vector[1] > 0) or (displacement_vector[0] < 0 and displacement_vector[1] < 0):
        # Определение нижней правой и верхней левой угловых точек вагона
        lower_right_box = max(o_bbox, key=lambda point: (point[1], point[0]))
        upper_left_corner = min(mask_lwt, key=lambda point: (point[0], -point[1]))
        return [
            # Для повышения точности определяются ближайшие точки на контуре вагона к выбранным угловым точкам
            closest_point_on_contour(mask_lwt, lower_right_box),
            closest_point_on_contour(mask_lwt, upper_left_corner),
        ]
    # Если вагон движется влево или вправо
    elif displacement_vector[1] == 0:
        return [
            # Определение крайней правой и крайней левой точки вагона
            max(mask_lwt, key=lambda point: (point[0], point[1])),
            min(mask_lwt, key=lambda point: (point[0], -point[1])),
        ]
    # Если вагон движется вверх или вниз
    else:
        return [
            # Определение верхней и нижней точки вагона
            max(mask_lwt, key=lambda point: point[1]),
            min(mask_lwt, key=lambda point: point[1]),
        ]


def draw_debug_info(
    img: np.ndarray,
    corner_points_list: List[np.ndarray],
    indexs_r_list: List[int],
    results_r: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Функция отображает на кадре отладочную информацию, связанную с движением вагонов и их пересечением с рельсами.

    Args:
        img (np.ndarray): Изображение, на которое будет нанёсена отладочная информация.
        corner_points_list (List[np.ndarray]): Список угловых точек всех вагонов, пересекающих рельсы.
        indexs_r_list (List[int]): Список индексов рельсов, пересекающихся с вагонами.
        results_r (Dict[str, np.ndarray]): Результаты сегментации рельсов, содержащие маски рельсов.

    Returns:
        img (np.ndarray): Изображение с нанесённой отладочной информацией.
    """
    # Отображение угловых точек вагонов на изображении
    for points_pair in corner_points_list:
        for point in points_pair:
            # Рисуем круг в каждой угловой точке
            cv2.circle(img, tuple(point), 10, (255, 255, 0), 5)

    # Отображение пересекающихся рельсов на изображении
    for index_r in indexs_r_list:
        # Отрисовываем контур рельсов на изображении
        img = plot_contour(img, results_r['masks'][index_r], color=[0, 255, 255], thickness=3)
    
    # Возвращаем изображение с отладочной информацией
    return img


def get_persons_feet(
    results_tracking_p: Dict[str, np.ndarray],
    feet_ratio: float,
) -> np.ndarray:
    """
    Функция для получения координат ног людей на основе результатов трекинга.

    Args:
        results_tracking_p (Dict[str, np.ndarray]): Результаты трекинга, содержащие координаты ограничивающих рамок (bbox) людей.
        feet_ratio (float): Коэффициент, используемый для вычисления позиции ног относительно нижней границы bbox.

    Returns:
        np.ndarray: Массив координат ног людей. Если результаты трекинга отсутствуют, возвращается пустой массив.
    """
    # Проверяем, есть ли результаты трекинга
    if len(results_tracking_p["boxes_xyxy"]) > 0:
        # Копируем массив координат bbox людей
        persons_feet = np.array(results_tracking_p['boxes_xyxy']).copy()
        # Изменяем верхнюю границу bbox, чтобы получить координаты ног
        persons_feet[:, 1] = persons_feet[:, 3] * feet_ratio
        return persons_feet
    # Если результаты трекинга отсутствуют, возвращаем пустой массив
    return np.array([])


def is_person_crossing_rails(
    box_feet_p: np.ndarray,
    wagons_rails_link: Dict[int, int],
    results_r: Dict[str, np.ndarray],
    threshold_ioa: float,
) -> Tuple[Any, bool]:
    """
    Функция для определения, пересекают ли ноги человека рельсы, которые находятся под движущимися вагонами.

    Args:
        box_feet_p (np.ndarray): Координаты ног человека в виде [x1, y1, x2, y2].
        wagons_rails_link (Dict[int, int]): Словарь, связывающий индексы рельсов и индексы вагонов.
        results_r (Dict[str, np.ndarray]): Результаты сегментации рельсов, содержащие маски рельсов.
        threshold_ioa (float): Пороговое значение IoA (Intersection over Area) для определения пересечения ног человека и рельсов.

    Returns:
        bool: Возвращает True, если ноги человека пересекают рельсы, и False в противном случае.
    """
    # Перебираем все индексы рельсов, связанных с движущимися вагонами
    for index_r in wagons_rails_link.keys():
        # Получаем маску рельсов по индексу
        mask_r = results_r["masks"][index_r]
        # Проверяем, пересекаются ли ноги человека с рельсами, используя IoA
        if intersection_over_area_mask(box_feet_p, mask_r) > threshold_ioa:
            # Если пересечение есть, возвращаем True
            return index_r, True
    # Если пересечений нет, возвращаем False
    return None, False
