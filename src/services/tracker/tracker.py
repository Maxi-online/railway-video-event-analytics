from collections import defaultdict
from time import time
from typing import List, Dict, Optional

import cv2
import numpy as np
from ultralytics.utils.ops import scale_image

# === our utils ===
from src.utils.preprocess import preprocess_frame
from src.utils.nms_utils import soft_nms

from src.services.tracker.tracker_utils import (
    get_track_person_hw, get_track_velocity_ema, get_track_direction_ema
)
from src.utils.geometry_utils import transform_mask_2_max_polygon


class Tracker:
    def __init__(self, model):
        # Определение модели
        self.model = model

        # Инициализация времени появления кадра
        self.new_frame_time = time()
        self.prev_frame_time = 0

        # Инициализация истории трекинга
        self.track_history = defaultdict(lambda: [])

        # Словарь для хранения времени последнего обновления каждого трека
        self.last_update_time = {}

        # Параметр для определения, сколько времени трек может быть неактивным перед удалением
        self.track_timeout = 120 # в секундах

    def predict_on_image_track(
        self,
        img: np.ndarray,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        tracker: str = "botsort.yaml",
    ) -> Dict:
        """
        Функция выполняет обработку изображения с помощью модели YOLO и отслеживает движение объектов.

        Args:
            img (np.ndarray): Изображение в формате np.ndarray для анализа.
            iou (float): Порог для расчёта пересечения объектов (по умолчанию 0.5).
            conf (float): Порог уверенности в детекции объекта (по умолчанию 0.5).
            imgsz (int): Размер изображения для обработки (по умолчанию 640).
            verbose (bool): Включение режима подробного вывода информации (по умолчанию False).
            tracker (str): Используемая конфигурация трекера (по умолчанию "botsort.yaml").

        Returns:
            Dict: Словарь с результатами обработки изображения.
        """
        # ----- Pre-processing (CLAHE контраст) -----
        img_proc = preprocess_frame(img)

        # ----- Inference & трекинг -----
        result = self.model.track(img_proc, iou=iou, conf=conf, imgsz=imgsz,
                                  verbose=verbose, tracker=tracker, persist=True)[0]

        # Получение результатов детекции
        cls = result.boxes.cls.cpu().numpy()
        self.class_names = result.names
        probs = result.boxes.conf.cpu().numpy()
        # Координаты Bbox для трекинга возвращаем в виде [xc, yc, w, h] - координаты центра Bbox,
        # его ширина и высота. Сделано для повышения точности расчета скорости перемещения объекта
        boxes = result.boxes.xywh.int().cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        track_ids = []  # по умолчанию

        # ----- Soft-NMS для устранения дублей без потери Recall -----
        if len(boxes_xyxy) > 0:
            keep = soft_nms(boxes_xyxy.copy(), probs.copy(), cls.copy(), iou_thr=0.5, sigma=0.5)
            boxes = boxes[keep]
            boxes_xyxy = boxes_xyxy[keep]
            cls = cls[keep]
            probs = probs[keep]
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().numpy()[keep]
            else:
                track_ids = []
        # если боксов нет — keep пропускаем

        # Сохранение результатов
        results = {'boxes': boxes, 'boxes_xyxy': boxes_xyxy, 'masks': [], 'cls': cls, 'probs': probs,
                   'track_ids': track_ids}

        try:
            # Получение масок объектов
            masks = result.masks.data.cpu().numpy()
            masks = np.moveaxis(masks, 0, -1)
            masks = scale_image(masks, result.orig_shape)
            masks = np.moveaxis(masks, -1, 0)
            masks = [transform_mask_2_max_polygon(mask) for mask in masks]
            results['masks'] = masks
        except:
            # Обработка случая, когда маски объектов недоступны
            results['masks'] = [None] * len(results['boxes'])

        # Обновление временных меток для трекинга
        self.prev_frame_time = self.new_frame_time
        self.new_frame_time = time()

        # Обновляем время последнего обновления для каждого трека
        current_time = time()
        for track_id in track_ids:
            self.last_update_time[track_id] = current_time

        # Проверяем и удаляем устаревшие треки
        self.remove_stale_tracks()

        return results

    def remove_stale_tracks(self):
        """Удаляет треки, которые не обновлялись в течение определенного времени."""
        current_time = time()
        stale_tracks = []
        for track_id, last_update in self.last_update_time.items():
            if current_time - last_update > self.track_timeout:
                stale_tracks.append(track_id)

        for track_id in stale_tracks:
            del self.last_update_time[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
            # print(f"Трек с ID {track_id} был удален из-за отсутствия обновлений.")

    def set_track_timeout(self, timeout):
        """Устанавливает время ожидания для удаления неактивных треков."""
        self.track_timeout = timeout


class TrackerPJ(Tracker):
    def __init__(self, model):
        super().__init__(model)
        # Инициализация словарей с результатами
        self.results_persons = dict()
        self.results_jackets = dict()

        # Словарь для хранения данных пересечения ног человека и движущегося вагона
        self.feet_wagon_intersections = {}


    def predict_on_image_track(
        self,
        img: np.ndarray,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        tracker: str = "botsort.yaml",
        track_history_len: int = 10,
        velocity_ema_start: int = 8,
        velocity_ema_alpha: float = 0.9,
        dir_ema_alpha: float = 0.5,
        show_velocity: bool = False,
        color_velocity: List = (0, 171, 255),
        thick_velocity: int = 2,
        fontscale_velocity: float = 1.2,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
    ) -> np.ndarray:
        """
        Функция выполняет обнаружение объектов на изображении и отслеживает их движение.

        Args:
            img (np.ndarray): Входное изображение.
            iou (float, optional): Порог перекрытия для подавления не максимумов. По умолчанию 0.5.
            conf (float, optional): Порог уверенности для фильтрации предсказаний. По умолчанию 0.5.
            imgsz (int, optional): Размер изображения для масштабирования. По умолчанию 640.
            verbose (bool, optional): Флаг для вывода подробной информации. По умолчанию False.
            tracker (str, optional): Путь к файлу конфигурации трекера. По умолчанию "botsort.yaml".
            track_history_len (int, optional): Длина истории трека. По умолчанию 10.
            velocity_ema_start (int, optional): Номер кадра, с которого начинается расчет скорости. По умолчанию 8.
            velocity_ema_alpha (float, optional): Коэффициент сглаживания экспоненциального скользящего среднего для скорости. По умолчанию 0.9.
            dir_ema_alpha (float, optional): Коэффициент сглаживания экспоненциального скользящего среднего для направления. По умолчанию 0.5.
            show_velocity (bool, optional): Флаг для отображения скорости на изображении. По умолчанию False.
            color_velocity (List, optional): Цвет для отображения скорости на изображении. По умолчанию (0, 171, 255).
            thick_velocity (int, optional): Толщина линии для отображения скорости на изображении. По умолчанию 2.
            fontscale_velocity (float, optional): Масштаб шрифта для отображения скорости на изображении. По умолчанию 1.2.
            font (int, optional): Шрифт для отображения скорости на изображении. По умолчанию cv2.FONT_HERSHEY_SIMPLEX.

        Returns:
            img (np.ndarray): Изображение с отслеживаемыми объектами.
        """
        results = super().predict_on_image_track(
            img, iou, conf, imgsz, verbose, tracker
            )

        self.results_persons = {'boxes': [], 'boxes_xyxy': [], 'masks': [], 'cls': [], 'probs': [], 'track_ids': []}
        self.results_jackets = {'boxes': [], 'boxes_xyxy': [], 'masks': [], 'cls': [], 'probs': [], 'track_ids': []}

        for box, box_xyxyx, mask, cls, prob, track_id in zip(*results.values()):
            if cls == 0:
                self.results_persons['boxes'].append(box)
                self.results_persons['boxes_xyxy'].append(box_xyxyx)
                self.results_persons['masks'].append(mask)
                self.results_persons['cls'].append(cls)
                self.results_persons['probs'].append(prob)
                self.results_persons['track_ids'].append(track_id)

                get_track_person_hw(self.track_history, track_id, track_history_len, box)
                get_track_velocity_ema(img, box, cls, track_id,
                                       self.track_history, track_history_len, self.class_names,
                                       self.new_frame_time, self.prev_frame_time, velocity_ema_alpha,
                                       show_velocity, color_velocity, thick_velocity,
                                       fontscale_velocity, font, correction={'enabled': False},
                                       velocity_ema_start=velocity_ema_start)
                get_track_direction_ema(cls, track_id, self.track_history,
                                        track_history_len,
                                        self.class_names, dir_ema_alpha)
            else:
                self.results_jackets['boxes'].append(box)
                self.results_jackets['boxes_xyxy'].append(box_xyxyx)
                self.results_jackets['masks'].append(mask)
                self.results_jackets['cls'].append(cls)
                self.results_jackets['probs'].append(prob)
                self.results_jackets['track_ids'].append(track_id)

        return img


class TrackerLWT(Tracker):
    def __init__(self, model):
        super().__init__(model)
        # Инициализация словаря с результатами
        self.results = dict()
        self.velocity_ema_start = None

        # Длина истории Track IDs
        self.track_history_len = None


    def predict_on_image_track(
        self,
        img: np.ndarray,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        tracker: str = "botsort.yaml",
        velocity_correction: Optional[dict] = None,
        track_history_len: int = 10,
        velocity_ema_start: int = 8,
        velocity_ema_alpha: float = 0.9,
        dir_ema_alpha: float = 0.5,
        show_velocity: bool = False,
        color_velocity: List = (0, 171, 255),
        thick_velocity: int = 2,
        fontscale_velocity: float = 1.2,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
        """
        Прогнозирует объекты на изображении и отслеживает их движение.

        Args:
            img (np.ndarray): Входное изображение.
            iou (float, optional): Порог перекрытия для подавления не максимумов. По умолчанию 0.5.
            conf (float, optional): Порог уверенности для фильтрации предсказаний. По умолчанию 0.5.
            imgsz (int, optional): Размер изображения для масштабирования. По умолчанию 640.
            verbose (bool, optional): Флаг для вывода подробной информации. По умолчанию False.
            tracker (str, optional): Путь к файлу конфигурации трекера. По умолчанию "botsort.yaml".
            velocity_correction (dict, optional): Параметры коррекции скорости. По умолчанию {'enable': False}.
            track_history_len (int, optional): Длина истории трека. По умолчанию 10.
            velocity_ema_start (int, optional): Номер кадра, с которого начинается расчет скорости. По умолчанию 8.
            velocity_ema_alpha (float, optional): Коэффициент сглаживания экспоненциального скользящего среднего для скорости. По умолчанию 0.9.
            dir_ema_alpha (float, optional): Коэффициент сглаживания экспоненциального скользящего среднего для направления. По умолчанию 0.5.
            show_velocity (bool, optional): Флаг для отображения скорости на изображении. По умолчанию False.
            color_velocity (List, optional): Цвет для отображения скорости на изображении. По умолчанию (0, 171, 255).
            thick_velocity (int, optional): Толщина линии для отображения скорости на изображении. По умолчанию 2.
            fontscale_velocity (float, optional): Масштаб шрифта для отображения скорости на изображении. По умолчанию 1.2.
            font (int, optional): Шрифт для отображения скорости на изображении. По умолчанию cv2.FONT_HERSHEY_SIMPLEX.

        Returns:
            img (np.ndarray): Изображение с отслеживаемыми объектами.
        """
        if velocity_correction is None:
            velocity_correction = {'enable': False}

        self.results = super().predict_on_image_track(
            img, iou, conf, imgsz, verbose, tracker
            )

        self.track_history_len = track_history_len

        for box, _, _, cls, _, track_id in zip(*self.results.values()):
            get_track_velocity_ema(img, box, cls, track_id,
                                self.track_history, track_history_len, self.class_names,
                                self.new_frame_time, self.prev_frame_time, velocity_ema_alpha,
                                show_velocity, color_velocity, thick_velocity,
                                fontscale_velocity, font, correction=velocity_correction, velocity_ema_start=velocity_ema_start)
            get_track_direction_ema(cls, track_id, self.track_history,
                                    track_history_len,
                                    self.class_names, dir_ema_alpha)

        return img
