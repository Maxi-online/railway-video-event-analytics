from collections import deque
from typing import Dict, List, Deque

import cv2
import numpy as np
from src.utils.geometry_utils import fit_line, intersection_rect_area


def get_track_person_hw(
        track_history: Dict[str, List],
        track_id: int,
        track_history_len: int,
        box: List[int]
) -> None:
    """
    Записывает отношение ширины к высоте (aspect ratio) объекта в историю трекинга для последующего анализа.

    Входные параметры:
        track_history (dict): Словарь для хранения истории каждого объекта,
        track_id (int): Уникальный идентификатор объекта,
        track_history_len (int): Максимальное количество сохраняемых записей в истории,
        box (list): Список с координатами объекта [x_center, y_center, width, height]
    """
    # Запись данных о hw bbox людей в track_history
    track_key_hw = f"p_{track_id}_hw"  # ключ для записи ширины/высоты bbox
    track_hw = track_history.get(track_key_hw, deque(maxlen=track_history_len))
    track_hw.append((box[2] / box[3]))  # ширина/высота bbox
    track_history[track_key_hw] = track_hw


def get_track_velocity_ema(
        img,
        box: list,
        cls: int,
        track_id: int,
        track_history: dict,
        track_history_len: int,
        class_names: dict,
        new_frame_time: float,
        prev_frame_time: float,
        velocity_ema_alpha: float,
        show_velocity: bool,
        color_velocity: List,
        thick_velocity: int,
        fontscale_velocity: float,
        font: int,
        correction: dict,
        velocity_ema_start: int = 8,
):
    """
    Обрабатывает результаты трекинга и обновляет информацию о скорости и экспоненциальной скользящей средней скорости объекта.

    Входные параметры:
        img: Изображение, на которое будет нанесена информация о скорости,
        box (list): Список с координатами объекта [x_center, y_center, width, height],
        cls (int): Идентификатор класса объекта,
        track_id (int): Уникальный идентификатор трека объекта,
        track_history (dict): Словарь для хранения истории каждого объекта,
        track_history_len (int): Максимальное количество сохраняемых записей в истории,
        class_names (dict): Словарь с названиями классов объектов,
        new_frame_time (float): Метка времени для текущего кадра,
        prev_frame_time (float): Метка времени для предыдущего кадра,
        velocity_ema_alpha (float): Коэффициент сглаживания для экспоненциальной скользящей средней,
        show_velocity (bool): Флаг для отображения скорости на изображении,
        color_velocity (List): Цвет текста скорости,
        thick_velocity (int): Толщина текста скорости,
        fontscale_velocity (float): Масштаб шрифта для текста скорости,
        font (int): Шрифт текста скорости,
        correction (dict): Словарь с параметрами коррекции скорости,
        velocity_ema_start (int, optional): Начальное значение для EMA, по умолчанию 8.

    Возвращаемое значение:
        Tuple: Обновленный словарь track_history и изображение img.
    """
    if not isinstance(track_history, dict):
        raise TypeError("track_history должен быть словарем")
    # Запись истории из 10 значений координат  иширины-высоты центра Bbox объекта трекинга
    x, y, w, h = box
    track_key_c_xywh = f"{class_names[cls]}_{track_id}_c_xy"
    track_values_c_xywh = track_history.get(track_key_c_xywh, deque(maxlen=track_history_len))
    track_values_c_xywh.append((int(x), int(y), int(w), int(h)))  # x, y center point, width, height
    track_history[track_key_c_xywh] = track_values_c_xywh

    # Расчет и запись истории из 10 значений скорости объекта трекинга
    velocity = velocity_calc(track_history, track_key_c_xywh, new_frame_time, prev_frame_time)
    track_key_velocity = f"{class_names[cls]}_{track_id}_velocity"
    track_values_velocity = track_history.get(track_key_velocity, deque(maxlen=track_history_len))
    track_values_velocity.append(velocity)
    track_history[track_key_velocity] = track_values_velocity

    # Расчет и запись истории из 10 значений экспоненциальной средней скорости объекта трекинга
    track_key_ema = f"{class_names[cls]}_{track_id}_ema"
    track_values_ema = track_history.get(track_key_ema, deque(maxlen=track_history_len))

    ema = update_ema_velocity(track_values_c_xywh,
                              track_values_velocity,
                              track_values_ema,
                              correction=correction,
                              alpha=velocity_ema_alpha,
                              velocity_ema_start=velocity_ema_start)

    track_values_ema.append(ema)
    track_history[track_key_ema] = track_values_ema

    # Отображение скорости объекта на изображении
    if show_velocity:
        if ema is not None:
            cv2.putText(
                img,
                f'{ema:.1f} ID: {track_id}',
                (x, y),
                font,
                fontscale_velocity,
                color_velocity,
                thick_velocity
            )
        if velocity is not None:
            cv2.putText(
                img,
                f'{velocity:.1f} ID: {track_id}',
                (x, y - 30),
                font,
                fontscale_velocity,
                (0, 255, 255),
                thick_velocity
            )


def velocity_calc(track_history: dict, track_key_c_xy: str, new_frame_time: float, prev_frame_time: float):
    """
    Рассчитывает скорость перемещения объекта на основе изменения его позиции между двумя последовательными кадрами.

    Входные параметры:
        track_history (dict): Словарь с историей трекинга объекта,
        track_key_c_xy (str): Ключ для доступа к данным о позиции центра объекта,
        new_frame_time (float): Метка времени для текущего кадра,
        prev_frame_time (float): Метка времени для предыдущего кадра

    Выходные параметры:
        velocity: Значение скорости перемещения объекта. float
    """
    track_data = track_history.get(track_key_c_xy, [])
    track_data = np.array(list(track_data), dtype=np.float32)
    if len(track_data) < 2:
        return 0

    # Расчет скорости на основе последних двух точек
    displacement = track_data[-1, :2] - track_data[-2, :2]

    # #корректировка перемещения центра обьекта на изменнение его размеров
    width_correction = (track_data[-2, 2] - track_data[-1, 2]) / 2
    height_correction = (track_data[-2, 3] - track_data[-1, 3]) / 2
    displacement_corrected = displacement - np.array([width_correction, height_correction])

    # Расчет скорректированного расстояния между двумя точками
    distance = np.linalg.norm(displacement_corrected)

    return distance


def intersection_over_union_c(
    boxA: List[float],
    boxB: List[float],
) -> float:
    """Функция рассчитывает IoU: степень пересечения двух Bboxes.

    Входные параметры:
        boxA (List[float]): Координаты Bbox 1-го объекта. [x1, y1, x2, y2],
        boxB (List[float]): Координаты Bbox 2-го объекта. [a1, b1, a2, b2]

    Выходные параметры:
        IoU (float): Степень пересечения двух Bboxes
    """
    # Преобразование координат Bbox в формат [x1, y1, x2, y2]
    def convert_to_corners(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]

    boxA = convert_to_corners(boxA)
    boxB = convert_to_corners(boxB)

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


def update_ema_velocity(track_values_c_xywh: List | Deque,
                        track_values_velocity: List | Deque,
                        track_values_ema: List | Deque,
                        correction: dict,
                        alpha: float = 0.5,
                        velocity_ema_start: int = 8):
    """
    Обновляет значение экспоненциальной скользящей средней (EMA) скорости.

    Входные параметры:
        track_values_c_xywh (np.ndarray): Значения параметров центра объекта,
        track_values_velocity (List): Значения скорости движения объекта,
        track_values_ema (List): Значения EMA объекта,
        alpha (float): Коэффициент сглаживания для EMA,
        correction (dict): Словарь с параметрами коррекции скорости,
        velocity_ema_start (int): Буфер для начала расчета EMA.

    Выходные параметры:
        velocity_ema (float): Обновленное значение EMA. float
    """
    # Преобразуем список значений c_xywh в массив NumPy с типом данных float16
    track_values_c_xywh = np.array(list(track_values_c_xywh), dtype=np.float16)
    # Извлекаем только координаты центра (x, y) объекта из track_values_c_xywh
    track_data_xy = track_values_c_xywh[:, :2]

    if correction['enabled']:
        len_track = len(track_values_velocity) if len(track_values_velocity) < int(correction['velocity_corr_history_len']) \
            else int(correction['velocity_corr_history_len'])    
        # Проверка, достаточно ли данных для расчета EMA
        if len(track_values_velocity) < velocity_ema_start:
            return None
        else:
            # Если данных недостаточно, возвращаем начальное значение EMA
            ema_first = np.mean(
                track_values_velocity) if track_values_velocity else 0  # Возвращаем начальное значение, если данных недостаточно

            IoU_list = []
            distance_list = []

            # Проверка, является ли объект неподвижным
            displacement = track_data_xy[-1] - track_data_xy[-len_track]
            # корректировка перемещения центра обьекта на изменнение его размеров
            width_correction = (track_values_c_xywh[-len_track, 2] - track_values_c_xywh[-1, 2]) / 2
            height_correction = (track_values_c_xywh[-len_track, 3] - track_values_c_xywh[-1, 3]) / 2
            displacement_corrected = displacement - np.array([width_correction, height_correction])
            # Вычисляем расстояние до нового положения с учетом корректировки
            distance = np.linalg.norm(displacement_corrected)

            # Анализ изменчивости скорости
            velocities = np.array(list(track_values_velocity), dtype=np.float16)
            velocity_differences = np.diff(velocities[-len_track:])
            # Определение резких изменений скорости, где абсолютное значение разности >= 4
            abrupt_change = np.abs(velocity_differences) >= correction['abrupt_change_diff']
            # Подсчитываем количество таких резких изменений
            abrupt_change_sum = np.sum(abrupt_change)
            # Извлекаем те разности скоростей, которые были признаны резкими изменениями
            abrupt_velocity_differences = velocity_differences[abrupt_change]
            # Анализируем многократные изменения направления скорости
            multidirectional = np.sum(np.diff(np.sign(abrupt_velocity_differences)) != 0)

            # Анализ изменчивости направления движения
            direction_changes = 0
            previous_direction = None
            # Проверка количества резкого изменения направления движения
            for i in range(1, len_track):
                # Расчет вектора направления движения
                direction = track_data_xy[i] - track_data_xy[i - 1]
                # Проверка на изменение направления движения
                if previous_direction is not None and np.dot(direction, previous_direction) < 1:
                    # Подсчет количества изменений направления движения
                    direction_changes += 1
                previous_direction = direction

                # Вычисление Intersection over Area (IoA) для последнего bbox и предыдущих
                IoU = intersection_over_union_c(track_values_c_xywh[-(i + 1)], track_values_c_xywh[-1])
                IoU_list.append(IoU)

            IoU_list = np.array(IoU_list)
            # Доля кадров, в которых IoU было больше порога
            IoU_small_share = np.mean(IoU_list[5:] > correction['IoU_thrd'])

            distance_list = np.array(distance_list)

            # УСЛОВИЯ ДЛЯ КОРРЕКЦИИ СКОРОСТИ
            # Если объект не двигался на значительное расстояние
            if distance < (len_track * correction['distance_thrd_hard']):
                return 0
            # Если объект двигался на умеренное расстояние, но при этом часто менял направление и был стабилен больше чем в половине случаев
            elif (abrupt_change_sum > int(len_track / correction['abrupt_change_sum']) or 
                  IoU_small_share >= correction['IoU_small_share'] or 
                  multidirectional >= int(len_track / correction['multidirectional'])
                  ) and distance < (len_track *  correction['distance_thrd_soft']):
                return 0
            
            # Расчет и обновление EMA
            # Если предыдущего значения EMA нет, используем начальное значение
            if track_values_ema[-1] is None:
                ema = ema_first
            else:
                ema = alpha * track_values_velocity[-1] + (1 - alpha) * track_values_ema[-1]
        return ema

    # EMA без коррекции
    else:
        if len(track_values_velocity) < 2:
            return track_values_velocity[
                0] if track_values_velocity else 0  # Возвращаем начальное значение, если данных недостаточно
        # Расчет EMA
        return alpha * track_values_velocity[-1] + (1 - alpha) * track_values_ema[-1]


def get_track_direction_ema(
        cls: int,
        track_id: int,
        track_history: dict,
        track_history_len: int,
        class_names: dict,
        alpha: float = 0.3,
) -> None:
    """
    Записывает историю направления движения объекта на основе последних точек в истории трекинга.

    Входные параметры:
        cls (int): Класс объекта,
        track_id (int): Уникальный идентификатор объекта,
        track_history (dict): Словарь для хранения истории каждого объекта,
        track_history_len (int): Максимальное количество сохраняемых записей в истории,
        class_names (dict): Словарь с названиями классов объектов,
        alpha (float): Коэффициент сглаживания для EMA,
    """
    track_data_xyhw = track_history.get(f"{class_names[cls]}_{track_id}_c_xy", [])
    if len(track_data_xyhw) < 4:
        return None

    # Расчет направления движения объекта на основе последних точек
    points_array = np.array(list(track_data_xyhw)[-4:], dtype=np.float32)
    points_array_xy = points_array[:, :2]
    angle_2_x, _ = fit_line(points_array_xy)
    track_key_direction = f"{class_names[cls]}_{track_id}_direction"
    track_values_direction = track_history.get(track_key_direction, deque(maxlen=track_history_len))
    track_values_direction.append(angle_2_x)
    track_history[track_key_direction] = track_values_direction

    # Расчет и запись истории из 10 значений экспоненциальной средней
    track_key_dir_ema = f"{class_names[cls]}_{track_id}_dir_ema"
    track_values_ema = track_history.get(track_key_dir_ema, deque(maxlen=track_history_len))
    if len(track_values_direction) < 2:
        ema = track_values_direction[
            -1] if track_values_direction else 0  # Возвращаем начальное значение, если данных недостаточно
    else:
        ema = alpha * track_values_direction[-1] + (1 - alpha) * track_values_ema[-1]
    
    track_values_ema.append(ema)
    track_history[track_key_dir_ema] = track_values_ema
