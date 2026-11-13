from src.utils.zone_events import feet_in_zone
import logging
from collections import deque
from datetime import datetime
from time import time
from typing import Tuple, Dict

import cv2
import numpy as np
from src.services.database.database import DataBase
from src.services.inferencer.inferencer import InferencerWorld
from src.services.tracker.tracker import TrackerPJ, TrackerLWT
from numpy.linalg import norm

from src.utils.draw_utils import (
    plot_bbox, plot_contour,
    plot_results, plot_danger_zone,
    plot_line_from_point, plot_approximated_rail_line
)
from src.utils.geometry_utils import (
    euclidean_distance, intersection_over_union, intersection_over_area_mask,
    intersection_rect_area, rect_area, intersection_polygon_rect_area,
    intersection_over_area_bbox, filter_contour_points,
)
from src.utils.tasks_funcs import (
    get_wagons_rails, approximate_rail_line,
    get_moving_wagons_rails, get_persons_feet,
    is_person_crossing_rails,
)

# Определяем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================#
#          Task_1_3  Определение сотрудника без СИЗ.            #
# ==============================================================#
def task_1_3(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    database: DataBase,
    font: int,
    threshold_pj: float,
    color_ok: Tuple[int, int, int],
    color_no: Tuple[int, int, int],
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """Функция решает задачу 1.3 - Определение сотрудника без СИЗ.

    Args:
        img (np.ndarray): Изображение с нанесенным текстом предупреждений.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        database (DataBase): База данных для хранения информации о срабатывании логики.
        font (int): Шрифт текста.
        threshold_pj (float): Порог отсечения для определения, что жилет надет на человеке, по значению IoU.
        color_ok (Tuple[int, int, int]): Цвет для маркировки людей в жилетах в формате BGR.
        color_no (Tuple[int, int, int]): Цвет для маркировки людей без жилетов в формате BGR.
        fontscale (float): Масштаб текста.
        thickness (int): Толщина линий для bbox и текста.

    Returns:
        img (np.ndarray): Кадр с отображением человека в жилете и без.
    """
    # Срабатывание логики
    database.triggered_1_3 = False

    # ID человека, на котором сработает логика
    persons_ids = tracker_PJ.results_persons["track_ids"]

    # Визуализация людей и жилетов с учетом логики
    for bbox_p, id in zip(tracker_PJ.results_persons['boxes_xyxy'], persons_ids):
        IoU = float('-inf')  # Инициализация переменной для хранения максимального IoU
        # Поиск максимального значения IoU между bboxes людей и жилетов
        for bbox_j in tracker_PJ.results_jackets['boxes_xyxy']:
            IoU = max(IoU, intersection_over_union(bbox_p, bbox_j))

        # Проверка, надет ли жилет на человеке на основе порога IoU threshold_pj
        if IoU > threshold_pj:
            # Визуализация, когда жилет надет
            img = plot_bbox(img, bbox_p, color_ok, thickness)
            cv2.putText(img, "В жилете", (int(bbox_p[0]), int(bbox_p[1])), font, fontscale, color_ok, 2)
        else:
            # Визуализация, когда жилет не надет
            img = plot_bbox(img, bbox_p, color_no, thickness)
            cv2.putText(img, "Нет жилета", (int(bbox_p[0]), int(bbox_p[1])), font, fontscale, color_no, 2)
            # Указание на то, что логика отработала
            database.triggered_1_3 = True
            database.triggered_time_1_3 = datetime.now()
            database.people_ids["1_3"] = int(id)

        # cv2.putText(img, f"ID: {id}", (int(bbox_p[0]), int(bbox_p[1] - 30)), font, fontscale, (240, 211, 105), 2)

    return img


# ====================================================================#
#          Task_1_2 Определение сотрудника в опасной зоне             #
# ====================================================================#
def task_1_2(
    img: np.ndarray,
    results_p: dict,
    database: DataBase,
    zone_id: int,
    font: int,
    danger_zone_cords: list,
    danger_zone_thrsd: float,
    feet_ratio: float,
    color_danger: Tuple[int, int, int],
    color_safe: Tuple[int, int, int],
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """
    Функция решает задачу 1.2 - Определение сотрудника в опасной зоне.

    Args:
        img (np.ndarray): Изображение с нанесенным текстом предупреждений.
        results_p (dict): Результаты предсказания модели.
        database (DataBase): База данных для хранения информации о срабатывании логики.
        zone_id (int): ID зоны.
        font (int): Шрифт текста (cv2.FONT_HERSHEY_*).
        danger_zone_cords (list): Координаты опасной зоны. Может быть либо список координат прямоугольника, либо массив координат полигона.
        danger_zone_thrsd (float): Порог отсечения, что человек находится в опасной зоне. Если доля пересечения bbox ног с опасной зоной превышает этот порог, человек считается в опасной зоне.
        feet_ratio (float): Порог для определения положения ног человека.
        color_danger (Tuple[int, int, int]): Цвет для маркировки опасной зоны в формате BGR.
        color_safe (Tuple[int, int, int]): Цвет для маркировки безопасной зоны в формате BGR.
        fontscale (float): Масштаб текста.
        thickness (int): Толщина линий для bbox и текста.

    Returns:
        img (np.ndarray): Кадр с отображением человека в жилете и без.
    """
    if not isinstance(danger_zone_cords, (list, dict, np.ndarray)) or len(danger_zone_cords) < 3:
        logger.warning(f"Invalid danger zone coordinates for zone {zone_id}")
        return img
    # Срабатывание логики
    database.triggered_1_2 = False

    # Координаты опасной зоны
    danger_zone_cords = np.array(danger_zone_cords)
    # # Находим все Bboxes ног людей
    persons_feet = get_persons_feet(results_p, feet_ratio)

    # ID человека, на котором сработает логика
    persons_ids = results_p["track_ids"]

    # Проходим по всем ногам людей и проверяем что все они не наступают на запрещенную зону
    count = 0
    for bbox, id in zip(persons_feet, persons_ids):
        if feet_in_zone(bbox, danger_zone_cords, danger_zone_thrsd):
            count += 1
            database.people_ids[f"1_2_{zone_id}"] = int(id)

    # Визуализация
    if count != 0:
        img = plot_danger_zone(img, danger_zone_cords, color_danger, thickness)
        if danger_zone_cords.ndim == 1:  # Если опасная зона - прямоугольник
            cv2.putText(img, "Человек в опасной зоне", (danger_zone_cords[0], danger_zone_cords[1]), font, fontscale,
                        color_danger, thickness)
        else:  # Если опасная зона - полигоны
            cv2.putText(img, "Человек в опасной зоне", (danger_zone_cords[:, 0].max(), danger_zone_cords[:, 1].min()),
                        font, fontscale, color_danger, thickness)
        # Указание на то, что логика отработала
        database.triggered_1_2 = True
        database.triggered_time_1_2 = datetime.now()
    else:
        img = plot_danger_zone(img, danger_zone_cords, color_safe, thickness)
        if danger_zone_cords.ndim == 1:  # Если опасная зона - прямоугольник
            cv2.putText(img, "Зона чиста", (danger_zone_cords[0], danger_zone_cords[1]), font, fontscale, color_safe,
                        thickness)
        else:  # Если опасная зона - полигоны
            cv2.putText(img, "Зона чиста", (danger_zone_cords[:, 0].max(), danger_zone_cords[:, 1].min()), font,
                        fontscale, color_safe, thickness)

    return img


# =================================================================================================================#
#                           Task_3_1: Спрыгивание с подвижного состава, платформы                                  #
# =================================================================================================================#
def task_3_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    font: int,
    skip_interval: int,
    move_velocity_lwt: float,
    feet_wagon_intersections_history_len: int,
    feet_ratio: float,
    threshold_ioa: float,
    show_task: bool,
    alert_text: str,
    color_alert: Tuple[int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
) -> np.ndarray:
    """
    Функция для определения, спрыгнул ли сотрудник с движущегося вагона.

    Args:
        img (np.ndarray): Исходное изображение.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        traker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        database (Database): Экземаляр Класса Базы данных для хранения информации о срабатывании логики.
        font (int): Шрифт.
        skip_interval (int): Интервал пропуска кадров.
        move_velocity_lwt (float): Пороговое значение скорости движения вагона, при котором начинается движение.
        feet_wagon_intersections_history_len (int): Длина истории хранимых пересечений человека по id с движущимся вагоном.
        feet_ratio (float): Порог для определения положения ног человека.
        threshold_ioa (float): Пороговое значение IoA, когда срабатывает логика.
        show_task (bool): Показывать предупреждающее сообщение или нет.
        alert_text (str): Текст предупреждающего сообщения.
        color_alert (Tuple[int]): Цвет предупреждающего сообщения логики.
        txt_up_alert (int): Насколько высоко предупреждающее сообщение отступает от верхней границы рамки объекта в пикселях.
        fontscale_alert (float): Масштаб текста предупреждающего сообщения.
        thick_alert (int): Толщина текста предупреждающего сообщения.

    Returns:
        img (np.ndarray): Изображение с нанесенным текстом предупреждения.
    """
    # Срабатывание логики
    database.triggered_3_1 = False

    # Получение результатов трекинга людей и вагонов
    results_tracking_p = tracker_PJ.results_persons
    # Получаем координаты ног людей
    persons_feet = get_persons_feet(tracker_PJ.results_persons, feet_ratio)

    results_tracking_lwt = traker_LWT.results
    track_history_LWT = traker_LWT.track_history
    class_names_LWT = traker_LWT.class_names

    # Перебор результатов трекинга людей
    for box_feet_p, box_xyxyx_p, track_id_p in zip(persons_feet, results_tracking_p['boxes_xyxy'],
                                                   results_tracking_p['track_ids']):
        # Перебор результатов трекинга поездов
        for box_lwt, _, mask_lwt, cls_lwt, _, track_id_lwt in zip(*results_tracking_lwt.values()):
            # Определение ключа для нахождения истории пересечений человека с движущимся вагоном
            track_key_p_feet_wagon_intersection = f"person{track_id_p}_{track_id_lwt}"
            # Извлечение текущей скорости вагона для трека
            track_key_lwt_vel_ema = f"{class_names_LWT[cls_lwt]}_{track_id_lwt}_ema"
            # Получение списка скоростей вагона из истории трекинга
            track_lwt_ema_list = list(track_history_LWT.get(track_key_lwt_vel_ema, []))
            # Использование последнего элемента в списке скоростей как текущей скорости
            current_lwt_vel = track_lwt_ema_list[-1]
            # Проверка, что поезд движется
            if current_lwt_vel:
                # Проверка, что текущая скорость вагона больше заданного порога для движения вагона
                if current_lwt_vel > move_velocity_lwt:
                    # Получаем значения пересечений человека с движущимся вагоном
                    p_feet_wagon_intersections_vals = tracker_PJ.feet_wagon_intersections.get(
                        track_key_p_feet_wagon_intersection,
                        deque(maxlen = int(feet_wagon_intersections_history_len) + 1))
                    feet_wagon_IoA = intersection_polygon_rect_area(mask_lwt, box_feet_p) / rect_area(box_feet_p)

                    # Для тестов
                    # print(f"person{track_id_p} = {feet_wagon_IoA}")

                    # Проверка что пересечение ног человека и движущегося вагона больше threshold_ioa
                    # Записываем '1' если ноги человека пересекаются с движущимся вагоном, иначе '0'
                    if feet_wagon_IoA > threshold_ioa:
                        p_feet_wagon_intersections_vals.append(1)
                    else:
                        p_feet_wagon_intersections_vals.append(0)

                    # Записываем обновленные значения пересечений для человека
                    tracker_PJ.feet_wagon_intersections[
                        track_key_p_feet_wagon_intersection] = p_feet_wagon_intersections_vals

                    # Проверка на выполнение бизнес-логики
                    if 1 in p_feet_wagon_intersections_vals and 0 in list(p_feet_wagon_intersections_vals)[-int(feet_wagon_intersections_history_len/skip_interval):]:
                        if show_task:
                            # Отображение на изображении текста предупреждения
                            cv2.putText(img, alert_text, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - txt_up_alert),
                                        font, fontscale_alert, color_alert, thick_alert)
                        # Указание на то, что логика отработала
                        database.triggered_3_1 = True
                        database.triggered_time_3_1 = datetime.now()
                        database.people_ids["3_1"] = int(track_id_p)

    # Для тестов
    # print(tracker_PJ.feet_wagon_intersections)

    return img


# =================================================================================================================#
#          Task_3_3: Проезд сотрудника на подножке, наружной лестнице цистерны, совмещенной с подножкой            #
# =================================================================================================================#
def task_3_3(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    font: int,
    move_velocity_lwt: float,
    move_velocity_p: float,
    atol_velocity: float,
    threshold_ioa: float,
    feet_ratio: float,
    show_task: bool,
    alert_text: str,
    color_alert: Tuple[int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
) -> np.ndarray:
    """
    Функция для определения, зацепился ли сотрудник за движущийся вагон, локомотив, цистерну.
    
    Args:
        img (np.ndarray): Исходное изображение.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        traker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        database (Database): Экземаляр Класса Базы данных для хранения информации о срабатывании логики.
        font (int): Шрифт.
        move_velocity_lwt (float): Пороговое значение скорости движения вагона, при котором начинается движение.
        move_velocity_p (float): Пороговое значение скорости движения человека, при котором начинается движение.
        atol_velocity (float): Absolute tolerance velocity - разница между скоростями вагона и человека, меньше которой считается, что их скорости равны.
        threshold_ioa (float): Пороговое значение IoA, когда срабатывает логика.
        feet_ratio (float): Порог для определения положения ног человека.
        show_task (bool): Показывать предупреждающее сообщение или нет.
        alert_text (str): Текст предупреждающего сообщения.
        color_alert (Tuple[int]): Цвет предупреждающего сообщения логики.
        txt_up_alert (int): Насколько высоко предупреждающее сообщение отступает от верхней границы рамки объекта в пикселях.
        fontscale_alert (float): Масштаб текста предупреждающего сообщения.
        thick_alert (int): Толщина текста предупреждающего сообщения.

    Returns:
        img (np.ndarray): Изображение с нанесенным текстом предупреждения, если человек зацепился за вагон.
    """
    # Срабатывание логики
    database.triggered_3_3 = False

    # Получение результатов трекинга людей и вагонов
    results_tracking_p = tracker_PJ.results_persons
    track_history_P = tracker_PJ.track_history
    results_tracking_lwt = traker_LWT.results
    track_history_LWT = traker_LWT.track_history
    class_names_LWT = traker_LWT.class_names

    # Находим все Bboxes ног людей
    persons_feet = get_persons_feet(results_tracking_p, feet_ratio)

    # Перебор результатов трекинга людей
    for box_xyxyx_p, box_feet_p, track_id_p in zip(results_tracking_p['boxes_xyxy'], persons_feet,
                                                   results_tracking_p['track_ids']):
        # Извлечение текущей скорости человека для трека
        track_key_p_vel_ema = f"person_{track_id_p}_ema"
        # Получение списка скоростей человека из истории трекинга
        track_p_ema_list = list(track_history_P.get(track_key_p_vel_ema, []))
        # Использование последнего элемента в списке скоростей как текущей скорости
        current_p_vel = track_p_ema_list[-1]

        # Находим самую нижнюю точку ног человека
        box_feet_p_y_lowest = box_feet_p[1::2].max()

        # Перебор результатов трекинга поездов
        for box_lwt, _, mask_lwt, cls_lwt, _, track_id_lwt in zip(*results_tracking_lwt.values()):
            # Ноги человека находятся ли на вагоне
            # Находим самую нижнюю точку вагона
            mask_lwt_y_lowest = mask_lwt[:, 1].max()
            if box_feet_p_y_lowest > mask_lwt_y_lowest:
                continue

            # Извлечение текущей скорости вагона для трека
            track_key_lwt_vel_ema = f"{class_names_LWT[cls_lwt]}_{track_id_lwt}_ema"
            # Получение списка скоростей вагона из истории трекинга
            track_lwt_ema_list = list(track_history_LWT.get(track_key_lwt_vel_ema, []))
            # Использование последнего элемента в списке скоростей как текущей скорости
            current_lwt_vel = track_lwt_ema_list[-1]
            # Проверка, что поезд движется
            if current_lwt_vel:
                # Проверка, что текущая скорость вагона больше заданного порога для движения вагона
                if current_lwt_vel > move_velocity_lwt and current_p_vel > move_velocity_p:
                    # Проверка условий схожести скоростей и пересечения областей
                    if (current_lwt_vel - current_p_vel) < atol_velocity:
                        # Вычисление коэффициента пересечения областей (Intersection over Area - IoA)
                        IoA = intersection_polygon_rect_area(mask_lwt, box_xyxyx_p) / rect_area(box_xyxyx_p)
                        # Проверка, что коэффициент пересечения больше порогового значения
                        if IoA > threshold_ioa:
                            if show_task:
                                # Отображение на изображении текста предупреждения
                                cv2.putText(img, alert_text, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - txt_up_alert),
                                            font, fontscale_alert, color_alert, thick_alert)
                            # Указание на то, что логика отработала
                            database.triggered_3_3 = True
                            database.triggered_time_3_3 = datetime.now()
                            database.people_ids["3_3"] = int(track_id_p)

    return img


# ====================================================================================================#
#          Task_4_2  Переход через железнодорожный путь перед движущимся подвижным составом           #
# ====================================================================================================#

def task_4_2(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    database: DataBase,
    font: int,
    move_velocity_lwt: float,
    threshold_ioa: float,
    thrld_dist_w_r: float,
    feet_ratio: float,
    alert_text: str,
    color_alert: Tuple[int, int, int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
    show_task: bool,
    show_debug: bool,
) -> np.ndarray:
    """
    Функция для определения, переходит ли сотрудник через рельсы перед движущимся подвижным составом.
    
    Args:
        img (np.ndarray): Исходное изображение.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        tracker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        database (DataBase): Экземпляр класса базы данных для хранения информации о срабатывании логики.
        font (int): Шрифт.
        move_velocity_lwt (float): Пороговое значение скорости движения вагона, при котором начинается логика.
        threshold_ioa (float): Пороговое значение для определения пересечения рельсов.
        thrld_dist_w_r (float): Доля от высоты вагона, на которой должен находиться центр пересечения рельсов и вагона.
        feet_ratio (float): Порог для определения положения ног человека.
        alert_text (str): Текст предупреждающего сообщения.
        color_alert (Tuple[int]): Цвет предупреждающего сообщения.
        txt_up_alert (int): Отступ текста предупреждающего сообщения от верхнего левого угла Bbox объекта.
        fontscale_alert (float): Масштаб текста предупреждающего сообщения.
        thick_alert (int): Толщина текста предупреждающего сообщения.
        show_task (bool): Флаг отображения текста предупреждающего сообщения на кадре.
        show_debug (bool): Флаг отображения отладочной информации на кадре.
    
    Returns:
        img (np.ndarray): Изображение с нанесенным текстом предупреждения.
    """
    # Срабатывание логики
    database.triggered_4_2 = False

    # Получаем список движущихся вагонов и пересекаемых рельсов
    wagons_rails_link = get_moving_wagons_rails(tracker_LWT, results_r, move_velocity_lwt,
                                                thrld_dist_w_r, show_debug, img)

    # Если нет движущихся вагонов, ничего не делаем
    if not wagons_rails_link:
        return img

    # Получаем координаты ног людей
    persons_feet = get_persons_feet(tracker_PJ.results_persons, feet_ratio)

    # Проверяем, пересекает ли кто-то из людей рельсы перед движущимся вагоном
    for box_feet_p, box_xyxyx_p, track_id_p in zip(persons_feet, tracker_PJ.results_persons['boxes_xyxy'],
                                                   tracker_PJ.results_persons['track_ids']):
        if is_person_crossing_rails(box_feet_p, wagons_rails_link, results_r, threshold_ioa)[1]:
            if show_task:
                cv2.putText(img, alert_text,
                            (int(box_xyxyx_p[0]), int(box_xyxyx_p[1] - txt_up_alert)), font,
                            fontscale_alert, color_alert, thick_alert)

            # Логика сработала
            database.triggered_4_2 = True
            database.triggered_time_4_2 = datetime.now()
            database.people_ids["4_2"] = int(track_id_p)

    return img


# ====================================================================================================================#
#     Task_5_2: Переход пути вблизи стоящего состава на расстоянии менее 5 метров с головы или хвоста состава         #
# ====================================================================================================================#
def task_5_2(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    database: DataBase,
    font: int,
    alert_distance: float,
    thrd_ioa_p_r: float,
    thrld_dist_w_r: float,
    feet_ratio: float,
    alert_text: str,
    color_alert: Tuple[int, int, int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
    show_task: bool,
    show_debug: bool,
) -> np.ndarray:
    """
    Функция для определения, переходит ли путь вблизи стоящего вагона на расстоянии менее 5 метров с головы или хвоста вагона.

    Args:
        img (np.ndarray): Исходное изображение.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        tracker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        results_r (Dict[str, np.ndarray]): Результаты трекинга рельс.
        database (DataBase): Экземпляр класса базы данных для хранения информации о срабатывании логики.
        font (int): Шрифт.
        alert_distance (float): Безопасное расстояние, на которое человек не должен приближаться к вагону.
        thrd_ioa_p_r (float): Пороговое значение IoA, когда срабатывает логика.
        thrld_dist_w_r (float): Доля от высоты вагона, на которой должен находиться центр пересечения рельсов и вагона.
        feet_ratio (float): Порог для определения положения ног человека.
        alert_text (str): Текст предупреждающего сообщения.
        color_alert (Tuple[int]): Цвет предупреждающего сообщения.
        txt_up_alert (int): Отступ текста предупреждающего сообщения от верхнего левого угла Bbox объекта.
        fontscale_alert (float): Масштаб текста предупреждающего сообщения.
        thick_alert (int): Толщина текста предупреждающего сообщения.
        show_task (bool): Флаг отображения текста предупреждающего сообщения на кадре.
        show_debug (bool): Флаг отображения отладочной информации на кадре.

    Returns:
        img (np.ndarray): Изображение с нанесенным текстом предупреждения.
    """
    # Срабатывание логики
    database.triggered_5_2 = False

    # Получаем список движущихся вагонов и пересекаемых рельсов
    wagons_rails_link, indexR_cornerPoints = get_wagons_rails(tracker_LWT, results_r, thrld_dist_w_r, show_debug, img)

    # Получаем координаты ног людей
    persons_feet = get_persons_feet(tracker_PJ.results_persons, feet_ratio)

    # Проверяем, пересекает ли кто-то из людей рельсы перед вагоном
    for box_feet_p, box_xywh_p, track_id_p in zip(persons_feet, tracker_PJ.results_persons['boxes'],
                                                  tracker_PJ.results_persons['track_ids']):
        index_r, cross_status = is_person_crossing_rails(box_feet_p, wagons_rails_link, results_r, thrd_ioa_p_r)
        if cross_status:
            # Находим увеличивающий коэффициент, по которому можно определить настоящее расстояние между вагоном и человеком
            # 1.7 - Средний рост человека в метрах
            k = 1.7 / box_xywh_p[3]

            # Находим все расстояния между человеком и вагоном, который стоит на рельсе, которую пересек человек
            if index_r in wagons_rails_link:
                # Находим координату центральной точки человека
                box_center_p = box_xywh_p[:2]
                # Угловая точка вагона №1
                corner_point_1_lwt = indexR_cornerPoints[index_r][0]
                corner_point_2_lwt = indexR_cornerPoints[index_r][1]
                # Находим наименьшее расстояние между человеком и угловыми точками вагона
                distance = float('inf')
                # Сохраняем координаты угловой точки вагона, с которой у человека наименьшее расстояние
                corner_point_lwt_min_distance = None
                for point in [corner_point_1_lwt, corner_point_2_lwt]:
                    d = euclidean_distance(box_center_p, point)
                    if d < distance:
                        distance = d
                        corner_point_lwt_min_distance = point

                # Переводим расстояние в настоящий масштаб
                distance = distance * k

                # Если расстояние меньше alert_distance (default = 5 метров)
                if distance < alert_distance:
                    if show_task:
                        # Отображение алерта события
                        cv2.putText(img, alert_text,
                                    (int(box_xywh_p[0]), int(box_xywh_p[1] - txt_up_alert)), font,
                                    fontscale_alert, color_alert, thick_alert)
                    # Логика сработала
                    database.triggered_5_2 = True
                    database.triggered_time_5_2 = datetime.now()
                    database.people_ids["5_2"] = int(track_id_p)

                if show_debug:
                    # Отображение расстояния между вагоном и человеком
                    cv2.putText(img, f"Расстояние: {distance} м",
                                (int(box_xywh_p[0]), int(box_xywh_p[1])), font,
                                fontscale_alert, [0, 255, 255], thick_alert)
                    # Отображение прямой между вагоном и человеком
                    cv2.line(img, box_center_p, corner_point_lwt_min_distance, color_alert, thick_alert)

    return img


# !!!! ФУНКЦИЯ task_6_2_frame в текущей реализации не используется
# ==========================================================================#
#          Task_6_2: Залезание работников на автосцепки или под них         #
# ==========================================================================#
def task_6_2_frame(
    img: np.ndarray,
    results_p: dict,
    results_lwt: dict,
    font: int,
    wagon_mask: bool,
    thrd_iou: float,
    thrd_p_hw_bbox: float,
    alert_text: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """
    Функция для определения, пролезает ли работник под стоящими вагонами, локомотивами или цистернами.
    
    Args:
        img (np.ndarray): Исходное изображение.
        results_p (dict): Результаты предсказания модели для людей, содержит координаты Bboxes и классы объектов.
        results_lwt (dict): Результаты предсказания модели для вагонов, локомотивов и цистерн, содержит координаты Bboxes и классы объектов.
        font: Шрифт для вывода текста.
        wagon_mask (bool): Использовать ли маску вагона для расчета пересечения.
        thrd_iou (float): Пороговое значение Intersection over Union (IoU) для определения пересечения человека и вагона.
        thrd_p_hw_bbox (float): Пороговое значение соотношения высоты к ширине Bbox человека.
        alert_text (str): Текст предупреждения.
        text_up (int): Смещение текста вверх от начальной точки Bbox.
        color (tuple): Цвет текста.
        fontscale (float): Масштаб шрифта.
        thickness (int): Толщина линии текста.

    Returns:
        img (np.ndarray): Изображение с нанесенным текстом предупреждения, если человек обнаружен под вагоном.
    """

    # Перебор всех обнаруженных объектов модели "человек/жилет" в предсказаниях
    for box_p in results_p['boxes_xyxy']:
        # Перебор всех обнаруженных объектов "вагон/локомотив/цистерна" в предсказаниях
        for _, box_lwt, mask_lwt, cls_lwt, _, _ in zip(*results_lwt.values()):
            # Проверка, что обнаруженный объект является "вагон/цистерна"
            if cls_lwt != 1:
                # Расчет высоты и ширины Bbox человека
                box_p_height = box_p[3] - box_p[1]
                box_p_width = box_p[2] - box_p[0]
                # Выбор метода расчета пересечения: через маску или Bbox
                if wagon_mask:
                    # Если используется маска, рассчитываем Intersection over Area (IoA)
                    IoA = intersection_polygon_rect_area(mask_lwt, box_p) / rect_area(box_p)
                else:
                    # Если не используется маска, рассчитываем Intersection over Area для Bboxes
                    IoA = intersection_over_area_bbox(box_p, box_lwt)
                # Проверка условий пересечения и соотношения размеров bbox человека
                if IoA > thrd_iou and box_p_width >= (box_p_height * thrd_p_hw_bbox):
                    # Наносим текст предупреждения на изображение
                    cv2.putText(img, alert_text,
                                (int(box_p[0]), int(box_p[1]) - text_up),
                                font, fontscale, color, thickness)
    return img


# ========================================================================================#
#                Task_6_1: Пролезание работников под стоящими вагонами                    #
# ========================================================================================#
def task_6_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    skip_interval: int,
    font: int,
    task_6_1_show: bool,
    time_absent_6_1: int,
    edge_threshold: int,
    thrd_ioa: float,
    alert_text_6_1: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """
    Анализирует изображение для определения работников, сидящих или находящихся под вагонами, 
    и аннотирует его при обнаружении таких случаев.

    Args:
        img (np.ndarray): Исходное изображение для аннотаций.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        traker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        database (Database): Экземаляр Класса Базы данных для хранения информации о срабатывании логики.
        skip_interval (int): Интервал пропуска кадров.
        font (int): Шрифт для текста.
        task_6_1_show (bool): Нужно ли отображать предупреждение для задачи 6.1.
        time_absent_6_1 (int): Время, в течение которого считается, что человек пропал.
        edge_threshold (int): Пороговое значение для определения границ кадра.
        thrd_ioa (float): Пороговое значение для пересечения вагона и человека.
        alert_text_6_1 (str): Текст предупреждения для задачи 6.1.
        text_up (int): Смещение текста вверх от начальной точки Bbox.
        color (tuple): Цвет текста.
        fontscale (float): Масштаб шрифта.
        thickness (int): Толщина линии текста.

    Returns:
        img: Аннотированное изображение.
    """
    # Срабатывание логик
    database.triggered_6_1 = False

    # Получение результатов трекинга людей и вагонов
    track_history_P = tracker_PJ.track_history
    last_update_time_p = tracker_PJ.last_update_time
    results_tracking_lwt = traker_LWT.results

    for track_id_p_abs, last_update_time in last_update_time_p.items():
        if (time() - last_update_time) > (time_absent_6_1 / skip_interval):
            # Проверка что человек пропал а не ушел за пределы кадра
            if track_history_P.get(f"person_{track_id_p_abs}_c_xy") is not None:
                last_known_position = np.array(list(track_history_P.get(f"person_{track_id_p_abs}_c_xy", [])))[-1]

                # Проверка что человек пропал а не ушел за пределы кадра
                if last_known_position is not None:
                    x, y = last_known_position[:2]
                    h, w = last_known_position[2:]
                    box_xyxyx_p = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                    frame_height, frame_width = img.shape[:2]

                    # Проверяем, что последняя известная позиция не находится близко к краям кадра
                    if (edge_threshold < x < frame_width - edge_threshold and
                            edge_threshold < y < frame_height - edge_threshold):

                        # Перебор результатов трекинга поезда
                        for _, _, mask_lwt, _, _, _ in zip(*results_tracking_lwt.values()):
                            # Расчет IoA для каждого вагона
                            IoA = intersection_polygon_rect_area(mask_lwt, box_xyxyx_p) / rect_area(box_xyxyx_p)
                            # Проверка условий пересечения человека с вагонами
                            if IoA > thrd_ioa:
                                # Отображение текста предупреждения на изображении
                                if task_6_1_show:
                                    cv2.putText(img, alert_text_6_1, (int(x), int(y) - text_up), font,
                                                fontscale, color, thickness)
                                # Указать, что логика была активирована
                                database.triggered_6_1 = True
                                database.triggered_time_6_1 = datetime.now()
                                database.people_ids["6_1"] = int(track_id_p_abs)
    return img


# ========================================================================================#
#                Task_6_2: Залезание работников на автосцепки или под них                #
#            Task_8.1.1: Определение работников, сидящих в зоне наблюдения               #
#                 (Совместная реализация логики для задач 6.2 и 8.1.1)                   #
# ========================================================================================#
def task_6_2_and_8_1_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    skip_interval: int,
    task_6_2_show: bool,
    task_8_1_1_show: bool,
    wagon_mask: bool,
    buffer_len_6_2: int,
    buffer_len_8_1_1: int,
    thrd_ioa: float,
    thrd_p_hw_6_2: float,
    thrd_p_hw_8_1_1: float,
    thrd_p_hw_8_1_1_down: float,
    show_p_hw: bool,
    alert_text_6_2: str,
    alert_text_8_1_1: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
    font,
) -> np.ndarray:
    """
    Анализирует изображение для определения работников, сидящих или находящихся под вагонами,
    и аннотирует их при обнаружении таких случаев.

    Args:
        img (np.ndarray): Исходное изображение для аннотаций.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        traker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов, локомотивов и цистерн.
        database (Database): Экземаляр Класса Базы данных для хранения информации о срабатывании логики.
        skip_interval (int): Интервал пропуска кадров.
        task_6_2_show (bool): Нужно ли отображать предупреждение для задачи 6.2.
        task_8_1_1_show (bool): Нужно ли отображать предупреждение для задачи 8.1.1.
        wagon_mask (bool): Использовать ли маску вагона для расчета пересечений.
        buffer_len_6_2 (int): Длина буфера для анализа истории задачи 6.2.
        buffer_len_8_1_1 (int): Длина буфера для анализа истории задачи 8.1.1.
        thrd_ioa (float): Порог IoA для определения опасности.
        thrd_p_hw_6_2 (float): Пороговое значение для определения пролезания под вагоном.
        thrd_p_hw_8_1_1 (float): Пороговое значение для определения сидящего положения.
        thrd_p_hw_8_1_1_down (float): Верхний порог для отсева ложн. срабатываний.
        show_p_hw (bool): Отображать ли отношение высоты к ширине bbox человека.
        alert_text_6_2 (str): Текст предупреждения для задачи 6.2.
        alert_text_8_1_1 (str): Текст предупреждения для задачи 8.1.1.
        text_up (int): Величина смещения текста вверх.
        color (tuple): Цвет текста.
        fontscale (float): Масштаб шрифта.
        thickness (int): Толщина линии текста.
        font (int): Шрифт для текста.

    Returns:
        img: Аннотированное изображение.
    """
    # Срабатывание логик
    database.triggered_6_2 = False
    database.triggered_8_1_1 = False

    # Получение результатов трекинга людей и вагонов
    results_tracking_p = tracker_PJ.results_persons
    track_history_P = tracker_PJ.track_history
    results_tracking_lwt = traker_LWT.results

    # Перебор результатов трекинга людей
    for box_p, box_xyxyx_p, track_id_p in zip(results_tracking_p['boxes'], results_tracking_p['boxes_xyxy'],
                                              results_tracking_p['track_ids']):
        # Извлечение истории изменений hw отношения человека для трека
        track_key_hw = f"p_{track_id_p}_hw"
        track_hw = list(track_history_P.get(track_key_hw, []))
        hw_ratio = box_p[2] / box_p[3]

        # Отрисовка hw отношения bbox человека
        if show_p_hw:
            cv2.putText(img, f'{hw_ratio:.2f} ID: {track_id_p}', (box_p[0], box_p[1]), font, fontscale, color,
                        thickness)

        # Список для хранения IoA для каждого вагона
        IoA_list = []
        # Перебор результатов трекинга поезда
        for box_lwt, box_xyxyx_lwt, mask_lwt, cls_lwt, _, _ in zip(*results_tracking_lwt.values()):
            # Выбор метода расчета пересечения: через маску или Bbox
            if wagon_mask:
                IoA = intersection_polygon_rect_area(mask_lwt, box_xyxyx_p) / rect_area(box_xyxyx_p)
            else:
                IoA = intersection_over_area_bbox(box_xyxyx_p, box_xyxyx_lwt)
            IoA_list.append(IoA)
            # Проверка, что обнаруженный объект является "вагон/цистерна"
            if cls_lwt != 1:
                # Проверка условий пересечения человека с вагонами
                if IoA > thrd_ioa:
                    # Задача 6.2: Пролезание работников под стоящими вагонами
                    # Проверка условий стории изменений hw отношения человека
                    if len(track_hw) >= buffer_len_6_2 and all(h > thrd_p_hw_6_2 for h in track_hw[-buffer_len_6_2:]):
                        if task_6_2_show:
                            # Отрисовка текста предупреждения 6_2
                            cv2.putText(img, alert_text_6_2, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - text_up), font,
                                        fontscale, color, thickness)
                        # Указание на то, что логика отработала
                        database.triggered_6_2 = True
                        database.triggered_time_6_2 = datetime.now()
                        database.people_ids["6_2"] = int(track_id_p)

        # Задача 8.1.1: Определение работников, сидящих в зоне наблюдения
        # Проверка, что IoA < thrd_ioa для всех вагонов
        if not any(IoA_list) > thrd_ioa:
            if len(track_hw) >= (buffer_len_8_1_1 / skip_interval) and all(
                    (thrd_p_hw_8_1_1 < h < thrd_p_hw_8_1_1_down) for h in track_hw[-(int(buffer_len_8_1_1 / skip_interval)):]):
                if task_8_1_1_show:
                    cv2.putText(img, alert_text_8_1_1, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - text_up), font,
                                fontscale, color, thickness)
                # Указание на то, что логика отработала
                database.triggered_8_1_1 = True
                database.triggered_time_8_1_1 = datetime.now()
                database.people_ids["8_1_1"] = int(track_id_p)
    return img


# ======================================================================================================#
#          Task_7 Идентификация пересечений ЖД-путей в запрещенной зоне или по опасной траектории       #
# ======================================================================================================#
def task_7(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    results_r: dict,
    database: DataBase,
    skip_interval: int,
    db_enabled: bool,
    task_1_2_enabled: bool,
    task_7_1_enabled: bool,
    font: int,
    move_velocity_p: float,
    triggered_1_2: dict,
    person_ids_1_2: dict,
    feet_ratio: float,
    thrd_ioa_p_r: float,
    color_danger: Tuple[int],
    color_safe: Tuple[int],
    start_track_7: int,
    direction_calc_buffer: int,
    thrd_angle_7_2_1: float,
    show_task_7_2_2: bool,
    show_task_7_2_1: bool,
    text_up_7: int,
    alert_text_7_1: str,
    alert_text_7_2_1: str,
    alert_text_7_2_2: str,
    color_alert: Tuple,
    fontscale: float,
    thickness: int,
    filter_rails: bool,
    r_p_filter_scale: int = 10,
    show_debug: bool = False,
) -> np.ndarray:
    """
    Выполняет задачи анализа трекинга и выполняет идентификацию пересечений ЖД-путей в запрещенной зоне или по опасной траектории.

    Args:
        img (np.ndarray): Изображение, на которое будут наноситься результаты анализа.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        results_r (dict): Результаты сегментации рельсов.
        database (Database): Экземпляр класса базы данных для хранения информации о срабатывании логики.
        skip_interval (int): Интервал пропуска кадров.
        db_enabled (bool): Флаг для включения базы данных.
        task_1_2_enabled (bool): Флаг для включения задачи 1_2.
        task_7_1_enabled (bool): Флаг для включения задачи 7_1.
        font (int): Шрифт для текста на изображении.
        triggered_1_2 (bool): Флаг срабатывания логики 1_2.
        person_id_1_2 (int): Идентификатор человека для задачи 1_2.
        move_velocity_p (float): Пороговая скорость движения человека.
        thrd_ioa_p_r (float): Порог пересечения рельсов человеком.
        color_danger (Tuple[int]): Цвет для отображения опасных зон.
        color_safe (Tuple[int]): Цвет для отображения безопасных зон.
        start_track_7 (int): Начальное количество отслеживаемых кадров для задачи 7.
        direction_calc_buffer (int): Буфер для расчета направления движения.
        thrd_angle_7_2_1 (float): Порог угла для задачи 7_2_1.
        show_task_7_2_2 (bool): Флаг для отображения задачи 7_2_2.
        show_task_7_2_1 (bool): Флаг для отображения задачи 7_2_1.
        text_up_7 (int): Смещение текста по оси Y для задач 7.
        alert_text_7_1 (str): Текст предупреждения для задачи 7_1.
        alert_text_7_2_1 (str): Текст предупреждения для задачи 7_2_1.
        alert_text_7_2_2 (str): Текст предупреждения для задачи 7_2_2.
        color_alert (Tuple): Цвет для текста предупреждений.
        fontscale (float): Масштаб шрифта для текста.
        thickness (int): Толщина шрифта для текста.
        filter_rails (bool): Флаг для фильтрации контура рельсов.
        r_p_filter_scale (int, optional): Масштаб фильтрации контура рельсов, по умолчанию 10.
        show_debug (bool): Отображать ли отладочную информацию на кадре или нет, по умолчанию False.

    Returns:
        img (np.ndarray): Изображение с нанесенным текстом предупреждений.
    """
    # Срабатывание логик
    database.triggered_7_1 = False
    database.triggered_7_2_2 = False
    database.triggered_7_2_1 = False

    # Получение результатов трекинга людей и вагонов
    results_tracking_p = tracker_PJ.results_persons
    track_history_P = tracker_PJ.track_history

    # Находим все Bboxes ног людей    
    persons_feet = get_persons_feet(results_tracking_p, feet_ratio)

    # Проходим по всем предсказаниям людей и вагонов для расчета логики
    for box_xyxy_p, person_feet, id_p in zip(results_tracking_p['boxes_xyxy'], persons_feet,
                                             results_tracking_p['track_ids']):
        # Нанесение на изображение площади ног человека
        if show_debug:
            img = plot_bbox(img, person_feet, (255, 128, 0), thickness)
        for mask_r in results_r['masks']:
            # Расчет площади пересечения человека и рельсы
            IoA_p_r = intersection_over_area_mask(person_feet, mask_r)

            # Проверка порога пересечения человеком рельсы
            if IoA_p_r > thrd_ioa_p_r:
                if task_7_1_enabled and db_enabled and task_1_2_enabled:
                    # Провера что сработала логика 1.2 и id человека пересекшего рельсу совпадает с id человека, кто наступает на рельсу
                    if triggered_1_2 is not None and person_ids_1_2 is not None:
                        if any(triggered_1_2.values()) and int(id_p) in person_ids_1_2.values():
                            # Отображение предупреждающего текста
                            cv2.putText(img, alert_text_7_1, (int(box_xyxy_p[0]), int(box_xyxy_p[1] + (text_up_7 + 30))),
                                        font,
                                        fontscale, color_danger, thickness)
                            # Указание на то, что логика отработала
                            database.triggered_7_1 = True
                            database.triggered_time_7_1 = datetime.now()
                            database.people_ids['7_1'] = int(id_p)

                # задача 7_2
                track_key_p_vel_ema = f"person_{id_p}_ema"
                track_p_ema_list = list(track_history_P.get(track_key_p_vel_ema, []))
                if len(track_p_ema_list) > start_track_7:
                    track_p_ema_list = track_p_ema_list[-start_track_7:]
                    # задача 7_2_2 - человек стоит на рельсах
                    if all(v < move_velocity_p for v in track_p_ema_list):
                        if show_task_7_2_2:
                            # Отображение предупреждающего текста
                            cv2.putText(img, alert_text_7_2_2, (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - text_up_7),
                                        font, fontscale, color_alert, thickness)
                        # Указание на то, что логика 7_2_2 отработала
                        database.triggered_7_2_2 = True
                        database.triggered_time_7_2_2 = datetime.now()
                        database.people_ids['7_2_2'] = int(id_p)
                    else:
                        # задача 7_2_1 - движение вдоль рельсов
                        # История перемещения человека
                        track_key_p_c_xy = f"person_{id_p}_c_xy"
                        track_p_c_xy_list = np.array(list(track_history_P.get(track_key_p_c_xy, [])))[:, :2]

                        # История направления движения человека
                        track_key_p_direction = f"person_{id_p}_dir_ema"
                        track_p_direction_list = list(track_history_P.get(track_key_p_direction, []))
                        p_angle_2x = track_p_direction_list[-1]

                        # Отрисовка траектории движения человека
                        if show_debug:
                            img = plot_line_from_point(img, track_p_c_xy_list[-1], p_angle_2x, color_safe,
                                                       thickness * 2, 400)

                        if filter_rails:
                            # Фильтрация контура рельсов по радиусу от человека
                            # Функция filter_contour_points фильтрует точки контура рельсов в зависимости от их расстояния
                            # до последней точки трека человека.
                            # Радиус фильтрации определяется как высота ног человека, умноженная на коэффициент r_p_filter_scale.
                            filtered_contour_r = filter_contour_points(mask_r, track_p_c_xy_list[-1], (
                                    int(person_feet[3]) - int(person_feet[1])) * r_p_filter_scale)
                            # Отрисовка контура отфильтрованных рельсов
                            if show_debug:
                                img = plot_contour(img, filtered_contour_r, color=[255, 128, 0], thickness=2)

                            # Аппроксимация линии рельсов
                            r_angle_2x, x_range, y_range = approximate_rail_line(filtered_contour_r, img.shape[0])

                        else:
                            r_angle_2x, x_range, y_range = approximate_rail_line(mask_r, img.shape[0])

                        # Отрисовка аппроксимированной линии рельсов
                        if show_debug:
                            img = plot_approximated_rail_line(img, x_range, y_range, color_danger, thickness * 2)

                        # Проверка на наличие рассчитанных углов
                        if len(track_p_direction_list) > (direction_calc_buffer / skip_interval) and r_angle_2x is not None:
                            track_p_dir_last = np.array(list(track_p_direction_list)[-int(direction_calc_buffer / skip_interval):],
                                                        dtype=np.float32)

                            # Отображение углов на изображении
                            # cv2.putText(img, f'p:{track_p_dir_last[-1]:.2f} r:{r_angle_2x:.2f}', (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - (text_up_7+20)), font, fontscale, color_alert, thickness)

                            # Проверка условия параллельности движения человека и рельсов
                            # Если разница между углами меньше порогового значения thrd_angle_7_2_1, то считается,
                            # что человек движется вдоль рельсов.
                            condition_met_percentage = np.mean(
                                (abs(track_p_dir_last - r_angle_2x) < thrd_angle_7_2_1).astype(int))
                            if condition_met_percentage > 0.6:
                                # Отработка предупреждения 7_2_1
                                if show_task_7_2_1:
                                    cv2.putText(img, alert_text_7_2_1,
                                                (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - text_up_7), font, fontscale,
                                                color_alert, thickness)
                                # Указание на то, что логика 7_2_1 отработала
                                database.triggered_7_2_1 = True
                                database.triggered_time_7_2_1 = datetime.now()
                                database.people_ids['7_2_1'] = int(id_p)

    return img


# ====================================================================================================#
#          Task_9  Определение посторонних предметов в заданной зоне станции                          #
# ====================================================================================================#
def task_9(
    img: np.ndarray,
    inferencer_world: InferencerWorld,
    world_model_params: dict,
    database: DataBase,
    task_9_show: bool,
    visualize_inference_params: dict,
    font: int
):
    """
        Выполняет детекцию посторонних предметов в зонах станции.

        Args:
            img (np.ndarray): Изображение, на которое будут наноситься результаты анализа.
            inferencer_world (InferencerWorld): экземпляр класса InferencerWorld для детекции посторонних предметов.
            world_model_params (dict): параметры модели YOLO WORLD.
            database (DataBase): Экземпляр класса базы данных для хранения информации о срабатывании логики.
            task_9_show (bool): Флаг для отображения задачи 9.
            visualize_inference_params (dict): параметры для отображения результатов инференса модели.
            font (int): Шрифт для текста на изображении.

        Returns:
            img (np.ndarray): Аннотированное изображение.
        """
    # Срабатывание логики
    database.triggered_9 = False

    # Определение посторонних предметов на кадре
    results_world, names_world = inferencer_world.predict_on_image_frame(img, "World",
                                                                         **world_model_params)
    # Если обнаружен хотя бы один посторонний объект на кадре
    if results_world["boxes_xyxy"].any():
        # Указание на то, что логика 9 отработала
        database.triggered_9 = True
        database.triggered_time_9 = datetime.now()

        if task_9_show:
            # Визуализация
            img = plot_results(results_world, names_world, img, 'world', font,
                               **visualize_inference_params)

    return img


# ====================================================================================================#
#                         Task_10_1  Определение нехарактерного положения                             #
# ====================================================================================================#
def task_10_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    database: DataBase,
    task_10_1_show: bool,
    alert_text_10_1: str,
    thrd_phw_10_1: float,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
    font: int,
) -> np.ndarray:
    """
        Определение нехарактерного положения человека, срабатывает при обнаружении человека в положении лежа.

        Args:
            img (np.ndarray): Изображение, на которое будут наноситься результаты анализа.
            tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
            database (Database): Экземпляр класса базы данных для хранения информации о срабатывании логики.
            task_10_1_show (bool): Флаг отображения результатов.
            alert_text_10_1 (str): Текст предупреждения.
            thrd_phw_10_1 (float): Порог для отношения высоты к ширине bbox.
            text_up (int): Смещение текста вверх.
            color (tuple): Цвет текста.
            fontscale (float): Масштаб текста.
            thickness (int): Толщина текста.
            font: Шрифт текста.

        Returns:
            img (np.ndarray): Аннотированное изображение.
        """
    # Срабатывание логики
    database.triggered_10_1 = False

    # Результаты трекинга людей
    results_tracking_p = tracker_PJ.results_persons

    # Определение нехарактерного положения человека на кадре
    for box_p, track_id_p in zip(results_tracking_p['boxes'], results_tracking_p['track_ids']):
        if box_p[3] / box_p[2] <= thrd_phw_10_1:
            if task_10_1_show:
                cv2.putText(img, alert_text_10_1, (int(box_p[0]), int(box_p[1]) - text_up), font,
                            fontscale, color, thickness)
            # Указание на то, что логика отработала
            database.triggered_10_1 = True
            database.triggered_time_10_1 = datetime.now()
            database.people_ids["10_1"] = int(track_id_p)

    return img


# ======================================================================================================#
#                     Task_10_2 Наличие признаков девиантного поведения (походка)                       #
# ======================================================================================================#
def task_10_2(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    tracker_LWT: TrackerLWT,
    database: DataBase,
    deviant_history: dict,
    skip_interval: int,
    move_velocity_p: float,
    task_10_2_show: bool,
    buffer_len_10_2: int,
    buffer_len_results_10_2: int,
    deviant_thrlds: dict,
    alert_text_10_2: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
    font: int,
    show_debug: bool = False,
) -> np.ndarray:
    """
    Анализирует и аннотирует изображение для определения признаков девиантной походки у работников.

    Args:
        img (np.ndarray): Изображение для анализа.
        tracker_PJ (TrackerPJ): Экземпляр класса трекера для людей и жилетов.
        tracker_LWT (TrackerLWT): Экземпляр класса трекера для вагонов.
        database (Database): Экземпляр класса базы данных для хранения информации о срабатывании логики.
        deviant_history (dict): История параметров девиантного поведения.
        skip_interval (int): Интервал пропуска кадров.
        move_velocity_p (float): Пороговая скорость движения человека.
        task_10_2_show (bool): Флаг отображения результатов.
        buffer_len_10_2 (int): Длина буфера для анализа.
        buffer_len_results_10_2 (int): Длина буфера для хранения результатов анализа.
        deviant_thrlds (dict): Пороговые значения для анализа девиантной походки.
        alert_text_10_2 (str): Текст предупреждения.
        text_up (int): Смещение текста вверх.
        color (tuple): Цвет текста.
        fontscale (float): Масштаб текста.
        thickness (int): Толщина текста.
        font: Шрифт текста.
        show_debug (bool): Отображать ли отладочную информацию на кадре или нет, по умолчанию False.

    Returns:
        img (np.ndarray): Аннотированное изображение.
    """
    # Инициализация флага срабатывания логики
    database.triggered_10_2 = False

    # Получение результатов трекинга людей и вагонов
    results_tracking_p = tracker_PJ.results_persons
    results_tracking_lwt = tracker_LWT.results
    track_history_P = tracker_PJ.track_history

    # Перебор результатов трекинга людей
    for box_p, box_xyxy_p, track_id_p in zip(results_tracking_p['boxes'], results_tracking_p['boxes_xyxy'], results_tracking_p['track_ids']):
        # Проверка условия достаточно большого размера bbox
        if box_p[3] > deviant_thrlds["p_height"]:
            # Проверка на пересечение bbox человека с bbox вагона
            if any(intersection_polygon_rect_area(mask_lwt, box_xyxy_p) / rect_area(box_xyxy_p) > deviant_thrlds["intersection_threshold"] 
                for _, _, mask_lwt, _, _, _ in zip(*results_tracking_lwt.values())):
                continue

            # Извлечение истории координат человека для трека
            track_p_coords_list = np.array(list(track_history_P.get(f"person_{track_id_p}_c_xy", [])))[:, :2]
            track_p_ema_list = np.array(list(track_history_P.get(f"person_{track_id_p}_ema", [])))
            track_p_direction_list = np.array(list(track_history_P.get(f"person_{track_id_p}_dir_ema", [])))

            # Извлечение истории изменений hw отношения человека для трека
            track_key_hw = f"p_{track_id_p}_hw"
            track_hw = list(track_history_P.get(track_key_hw, []))

            # Проверка достаточности размера истории для анализа
            if len(track_p_ema_list) >= buffer_len_10_2:
                try:
                    # Используем только последние buffer_len_10_2 значений
                    recent_coords = track_p_coords_list[-int(buffer_len_10_2 / skip_interval):]
                    recent_ema = track_p_ema_list[-int(buffer_len_10_2 / skip_interval):]
                    recent_directions = track_p_direction_list[-int(buffer_len_10_2 / skip_interval):]

                    # Вычисляем текущую скорость
                    current_velocity = recent_ema[-1]

                    # Анализ траектории движения
                    distance_short = norm(track_p_coords_list[-1] - track_p_coords_list[-int(buffer_len_10_2 / skip_interval)])
                    straightness = calculate_straightness(recent_coords)

                    # Анализ скорости
                    velocity_std = np.std(recent_ema)
                    velocity_changes = np.diff(recent_ema)
                    velocity_change_std = np.std(velocity_changes)

                    # Анализ направления
                    direction_changes = np.diff(recent_directions)
                    direction_change_std = np.std(direction_changes)

                    # Расчет z-scores
                    recent_straightness = [calculate_straightness(recent_coords[:i+2]) for i in range(len(recent_coords)-1)]
                    z_straightness = calculate_z_score(straightness, recent_straightness)
                    z_velocity_std = calculate_z_score(velocity_std, [np.std(recent_ema[i:i+2]) for i in range(len(recent_ema)-1)])
                    z_velocity_change_std = calculate_z_score(velocity_change_std, [np.std(velocity_changes[i:i+2]) for i in range(len(velocity_changes)-1)])
                    z_direction_change_std = calculate_z_score(direction_change_std, [np.std(direction_changes[i:i+2]) for i in range(len(direction_changes)-1)])

                    # Расчет девиантного поведения
                    sitting_buffer = int(deviant_thrlds["sitting_buffer"] / skip_interval)
                    if len(track_hw) >= sitting_buffer and all((h > deviant_thrlds['sitting']) for h in track_hw[-sitting_buffer:]):
                        deviant_score = 0  # Если человек сидит, считаем поведение не девиантным
                    elif current_velocity > move_velocity_p and distance_short >= (int(buffer_len_10_2 / skip_interval) * deviant_thrlds['distance']):
                        deviant_score = ((abs(z_straightness) + 
                                        abs(z_velocity_std) * deviant_thrlds['velocity_weight'] + 
                                        abs(z_direction_change_std)) / 3) / (box_p[3] / deviant_thrlds['height_factor'])
                    else:
                        deviant_score = 0  # Если человек остановился, считаем поведение не девиантным

                    # Запись данных о hw bbox людей в track_deviant
                    track_key_deviant = f"p_{track_id_p}_deviant"  # ключ для записи ширины/высоты bbox
                    track_deviant = deviant_history.get(track_key_deviant, deque(maxlen=int(buffer_len_results_10_2)))
                    track_deviant.append(deviant_score)  # высота/ширина bbox
                    deviant_history[track_key_deviant] = track_deviant
                    track_deviant_list = list(track_deviant)

                    # Проверка на превышение порога девиантного поведения
                    if len(track_deviant_list) >= (buffer_len_results_10_2 / skip_interval) and \
                        np.mean([int(dev > deviant_thrlds['deviant_score']) for dev in track_deviant_list[-int(buffer_len_results_10_2/skip_interval):]]) > deviant_thrlds['deviant_threshold']:
                        if task_10_2_show:
                            cv2.putText(img, alert_text_10_2, 
                                        (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - text_up), 
                                        font, fontscale, color, thickness)
                        database.triggered_10_2 = True
                        database.triggered_time_10_2 = datetime.now()
                        database.people_ids['10_2'] = int(track_id_p)

                    if show_debug:
                        print(np.mean([int(dev > deviant_thrlds['deviant_score']) for dev in track_deviant_list]))
                        debug_text = f"H: {box_p[3]}, Score: {deviant_score:.2f}, {z_straightness:.2f}, {z_velocity_std:.2f}, {z_velocity_change_std:.2f}, {z_direction_change_std:.2f}"
                        cv2.putText(img, debug_text,
                                    (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - (text_up + 30)),
                                    font, fontscale * 0.8, (0, 171, 255), thickness)

                except Exception as e:
                    if show_debug:
                        error_text = f"Error: {str(e)}"
                        cv2.putText(img, error_text,
                                    (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - (text_up + 30)),
                                    font, fontscale * 0.6, (0, 0, 255), thickness)
                        print(error_text)

    return img


def calculate_z_score(value, values):
    """
    Рассчитывает z-оценку для заданного значения относительно набора значений.

    Args:
        value: Значение, для которого требуется рассчитать z-оценку.
        values: Массив значений, относительно которых рассчитывается z-оценка.

    Returns:
        z_score: Значение z-оценки.
    """
    if len(values) < 2:
        return 0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0  # Возвращаем 0, если стандартное отклонение равно 0
    return (value - mean) / std


def calculate_straightness(coords):
    """
    Рассчитывает прямолинейность траектории. Range прямолинейности: (от 0 до 1).

    Args:
        coords: Массив координат траектории.
    """
    if len(coords) < 2:
        return 1.0
    total_distance = np.sum(norm(np.diff(coords, axis=0), axis=1))
    direct_distance = norm(coords[-1] - coords[0])
    return direct_distance / (total_distance + 1e-6)
