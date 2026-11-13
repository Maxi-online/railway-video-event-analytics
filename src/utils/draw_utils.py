import warnings
from typing import List, Tuple

import cv2
import numpy as np

# Игнорирование Warning Numpy
warnings.simplefilter('ignore', np.RankWarning)


def plot_bbox(image: np.ndarray, bbox: List[float], color: Tuple[int], thickness: int=1) -> np.ndarray:
    """
    Функция строит Bbox вокруг детектированного объекта на кадре.

    Входные параметры:
        frame (np.ndarray): Кадр, на который будет нанесен BBox объекта,
        bbox (np.ndarray): Координаты Bbox объекта в формате [x0, y0, x1, y1],
        color (Tuple[int]): Цвет Bbox,
        thickness (int): Толщина линии Bbox.

    Выходные параметры:
        image_bbox (np.ndarray): Изображение с нанесенным BBox объекта.
    """
    image_bbox = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                               color=color, thickness=thickness)   
    return image_bbox


def plot_binary_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int], alpha: float=0.5) -> np.ndarray:
    """
    Функция накладывает маску на исходное изображение и возвращает результат в виде np.ndarray.

    Входные параметры:
        frame (np.ndarray): Кадр, на который будет нанесена маска,
        mask (np.ndarray): Сегментационная маска,
        color (Tuple[int]): Цвет маски,
        alpha (float): Прозрачность цвета маски.
    Выходные параметры:
        image_combined (np.ndarray): Изображение с маской на объекте.
    """
    if mask is not None:
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    else:
        return image
    return image_combined


def plot_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int],
    thickness: int,
) -> np.ndarray:
    """
    Функция накладывает контур маски на исходное изображение и возвращает результат в виде np.ndarray.
    
    Args:
        image (np.ndarray): Кадр, на который будет нанесен контур маски.
        mask (np.ndarray): Сегментационная маска.
        color (Tuple[int]): Цвет контура маски.
        thickness (int): Толщина контура маски.

    Returns:
        frame (np.ndarray): Изображение с контуром маски на объекте.
    """

    if mask.any():
        image_contour = cv2.drawContours(image, [mask], -1, color, thickness=thickness)
    else:
        # Возврат исходного изображения, если маска отсутствует
        return image

    return image_contour


def plot_results(
    results: dict,
    class_names: dict,
    frame: np.ndarray,
    task: str,
    font: int,
    show_mask: bool,
    show_class: bool,
    colors_classes: dict,
    thickness: int,
    alpha: float,
    fontscale: float,
    is_tracking: bool = False,
) -> np.ndarray:
    """Функция наносит Bboxes и Masks на кадр и возвращает результат в виде np.ndarray.
    Поддерживает как обычную детекцию, так и трекинг, а также особый случай task="world".

    Args:
        results (dict): Результаты предсказания модели.
        class_names (dict): Словарь с названиями классов.
        frame (np.ndarray): Кадр для визуализации.
        task (str): Тип задачи ("PJ", "LWT", "R", "world").
        font (int): Шрифт текста (cv2.FONT_...).
        show_mask (bool): Рисовать маску или только контур.
        show_class (bool): Показывать ли класс и вероятность.
        colors_classes (dict): Цвета для классов.
        thickness (int): Толщина линий.
        alpha (float): Прозрачность маски (0-1).
        fontscale (float): Масштаб текста.
        is_tracking (bool): True, если results содержат данные трекинга (доп. поля).

    Returns:
        np.ndarray: Кадр с нарисованными BBoxes и Masks.
    """
    for items in zip(*results.values()):
        if is_tracking:
            _, bbox, mask, cls, prob, _ = items
        else:
            bbox, mask, cls, prob = items

        label = class_names[int(cls)]

        # Выбираем цвет в зависимости от задачи
        color_key = "foreign_objects" if task == "world" else label
        color = colors_classes[color_key]

        # Отрисовка BBox или Mask/Contour
        if task == "PJ":
            frame = plot_bbox(frame, bbox, color=color, thickness=thickness)
        elif task != "world" or is_tracking:  # Для "world" без трекинга маски не рисуются
            if show_mask:
                frame = plot_binary_mask(frame, mask, color=color, alpha=alpha)
            else:
                frame = plot_contour(frame, mask, color=color, thickness=thickness)

        # Отрисовка текста (класс + вероятность)
        if show_class and (task != "world" or is_tracking):
            cv2.putText(
                frame,
                f"{label} {prob:.2f}",
                (int(bbox[0]), int(bbox[1])),
                fontFace=font,
                fontScale=fontscale,
                color=color,
                thickness=thickness,
            )

    return frame


def plot_results_unet(
    results: dict,
    class_names: dict,
    img: np.ndarray,
    show_mask: bool,
    show_class: bool,
    color: list,
    thickness: int,
    fontscale: float,
    alpha: float,
    font: int,
) -> np.ndarray:
    """
    Функция для отрисовки контуров объектов на изображении.

    Args:
        results (dict): Содержит контуры ('contours'), ограничивающие рамки ('boxes') и ID классов ('cls'),
        class_names (dict): Словарь с названиями классов объектов,
        img (np.ndarray): Исходное изображение,
        show_mask (bool): Если True, показывать цветную маску на изображении,
        show_class (bool): Если True, отображать название класса на изображении,
        color (Tuple[int]): Цвет контура объектов на изображении,
        thickness (int): Толщина контура объектов на изображении,
        fontscale (float): Масштаб контура объектов на изображении,
        alpha (float): Прозрачность цветной маски,
        font (int): Шрифт для отображения текста.

    Returns:
        (np.ndarray): Изображение с нарисованными контурами и, возможно, классами объектов.
    """
    if results['masks'] is not None:
        for contour in results['masks']:
            if contour.any():
                if show_mask:
                    img = plot_binary_mask(img, contour, color=color, alpha=alpha)
                else:
                    img = cv2.drawContours(img, [contour], -1, color, thickness)
                # Отрисовка названия класса
                if show_class:
                    for contour, cls in zip(results['masks'],results['cls']):
                        label = class_names[int(cls)]
                        min_y_index = np.argmin(contour[:, 1])
                        min_y_x = contour[min_y_index, 0]
                        min_y_y = contour[min_y_index, 1]
                        label = class_names[int(cls)]
                        cv2.putText(img, label, (min_y_x, min_y_y),
                                    fontFace=font, fontScale=fontscale, color=color,
                                    thickness=thickness)
        return img


def plot_fps(
    frame: np.ndarray,
    frame_fps: float,
    avg_fps: float,
    color: Tuple[int],
    font,
):
    """Функция наносит количество кадров в секунду (FPS) в левом верхнем углу кадра.

    Args:
        frame (np.ndarray): Исходное изображение.
        frame_fps (float): Текущее количество кадров в секунду.
        avg_fps (float): Среднее количество кадров в секунду.
        color (Tuple[int]): Цвет отображения текста с FPS.
        font: Шрифт текста.
    """
    cv2.putText(frame, f"Frame FPS: {frame_fps:.1f}", (7, 70), fontFace=font,
                fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA)
    if avg_fps > 0:
        cv2.putText(frame, f"Average FPS: {avg_fps:.1f}", (7, 120), fontFace=font,
                    fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA)
    return


def plot_object_trajectory(
    img: np.ndarray,
    pt1: Tuple[int],
    pt2: Tuple[int],
    color: Tuple[int],
    thickness: int,
):
    """Функция наносит на кадр траекторию движения объекта.

    Args:
        img (np.ndarray): Исходное изображение.
        pt1 (Tuple[int]): Координаты точки начала траектории движения объекта.
        pt2 (Tuple[int]): Координаты точки конца траектории движения объекта.
        color (Tuple[int]): Цвет траектории движения объекта в формате BGR.
        thickness (int): Толщина линии траектории движения объекта.
    """
    cv2.line(img, pt1, pt2, color, thickness)
    return


def plot_danger_zone(
    img: np.ndarray,
    danger_zone_cords: np.ndarray,
    color: Tuple[int],
    thickness: int
) -> np.ndarray:
    """Функция наносит на кадр опасную зону (прямоугольник или полигон).

    Args:
        img (np.ndarray): Исходное изображение.
        danger_zone_cords (np.ndarray): Координаты опасной зоны.
        color (Tuple[int]): Цвет опасной зоны.
        thickness (int): Толщина линии фигуры опасной зоны.

    Returns:
        img (np.ndarray): Кадр с опасной зоной.
    """
    # Если опасная зона - прямоугольник
    if danger_zone_cords.ndim == 1:
        img = cv2.rectangle(img, (danger_zone_cords[0], danger_zone_cords[1]), (danger_zone_cords[2], danger_zone_cords[3]), color, thickness)
    else:
        # Если опасная зона - полигон
        img = cv2.polylines(img, [danger_zone_cords], True, color, thickness)

    return img


def plot_line_from_point(
    img: np.ndarray,
    center_point: Tuple[int, int],
    angle: float,
    color: Tuple[int, int, int],
    thickness: int,
    radius: int = 400,
) -> np.ndarray:
    """
    Функция для рисования линии от заданной точки с заданным радиусом и углом относительно оси x.

    Args:
        img (np.ndarray): Исходное изображение.
        center_point (Tuple[int, int]): Начальная точка линии (x, y).
        angle (float): Угол относительно оси x в градусах.
        color (Tuple[int, int, int]): Цвет линии (BGR).
        thickness (int): Толщина линии.
        radius (int): Радиус линии.

    Returns:
        img (np.ndarray): Изображение с нарисованной линией.
    """
    # Преобразование угла из градусов в радианы
    angle_rad = np.deg2rad(angle)

    # Вычисление половины радиуса
    half_radius = radius / 2

    # Вычисление начальной и конечной точек линии с использованием половины радиуса и угла
    start_point = (
        int(center_point[0] - half_radius * np.cos(angle_rad)),
        int(center_point[1] - half_radius * np.sin(angle_rad))
    )

    end_point = (
        int(center_point[0] + half_radius * np.cos(angle_rad)),
        int(center_point[1] + half_radius * np.sin(angle_rad))
    )

    # Ограничение координат начальной и конечной точек в пределах изображения
    start_point = (
        np.clip(start_point[0], 0, img.shape[1] - 1),
        np.clip(start_point[1], 0, img.shape[0] - 1)
    )

    end_point = (
        np.clip(end_point[0], 0, img.shape[1] - 1),
        np.clip(end_point[1], 0, img.shape[0] - 1)
    )

    # Рисование линии на изображении
    img = cv2.line(img, start_point, end_point, color, thickness)

    return img


def plot_approximated_rail_line(
    img: np.ndarray,
    x_range: np.ndarray,
    y_range: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    """
    Рисует аппроксимированную линию рельса на изображении.

    Args:
        img (np.ndarray): Исходное изображение.
        x_range (np.ndarray): X-координаты линии.
        y_range (np.ndarray): Y-координаты линии.
        color (Tuple[int, int, int]): Цвет линии в BGR.
        thickness (int): Толщина линии.

    Returns:
        img (np.ndarray): Изображение с нарисованной линией.
    """
    if len(x_range) < 2 or len(y_range) < 2:
        return img

    for i in range(len(x_range) - 1):
        img = cv2.line(img, (x_range[i], y_range[i]), (x_range[i + 1], y_range[i + 1]), color, thickness)

    return img
