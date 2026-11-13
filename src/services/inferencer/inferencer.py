from typing import Tuple, List

import cv2
import numpy as np
from ultralytics.utils.ops import scale_image

from src.utils.geometry_utils import transform_mask_2_max_polygon
from src.services.inferencer.inferencer_utils import (
    create_binary_mask, sigmoid
)


class Inferencer:
    def __init__(self, model):
        # Определение моделей
        self.model = model


    @staticmethod
    def onnx_preprocessing(
        image: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Подготовка изображения для вывода модели ONNX Runtime.

        Args:
            image (np.ndarray): Изображение в формате numpy.
            image_size (Tuple[int, int]): Желаемый размер изображения после изменения размера.

        Returns:
            (np.ndarray): Обработанное изображение, готовое для вывода модели.
        """
        # Изменение размера изображения
        image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)
        # Нормализация изображения
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0  # Нормализация Imagenet
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0   # Нормализация Imagenet
        denominator = np.reciprocal(std, dtype=np.float32)
        image = image.astype(np.float32)
        image -= mean
        image *= denominator
        # Транспонирование каналов изображения для совместимости с ONNX
        image = image.transpose((2, 0, 1))[None]
        return image


    def _predict_on_image_unet(
        self,
        img: np.ndarray,
        area_ratio: float,
        threshold: float = 0.4,
        imgsz_unet: Tuple[int, int] = (384, 384),
    ) -> Tuple:
        """
        Функция для предсказания маски объекта на изображении с использованием модели UNet и извлечения
        bounding boxes согласно заданному порогу площади.

        Args:
            img (np.ndarray): np.ndarray - Исходное изображение в формате numpy ndarray.
            area_ratio (float):  Порог для фильтрации контуров по отношению к площади изображения.
            threshold (float), optional (default=0.4):Порог для бинаризации предсказанной маски.
            imgsz_unet (Tuple[int, int]), optional (default=(384, 384)): Целевой размер изображения для предсказания.

        Returns:
            mask_R (np.ndarray): Бинаризованная маска объекта на изображении.
            result (dict): Содержит bounding boxes, контуры, и классы объектов.
            class_names (dict): Словарь с названиями классов.
        """
        # Предобработка изображения для модели ONNX
        onnx_input = self.onnx_preprocessing(img, image_size=imgsz_unet)
        onnx_input_name = self.model.get_inputs()[0].name
        ort_inputs = {onnx_input_name: onnx_input}
        # Получение предсказания модели
        results_R = self.model.run(None, ort_inputs)[0]
        prob_mask_R = sigmoid(results_R[0])[0]
        # Получение бинарной маски и приведение к размеру исходного изображения
        mask_R = create_binary_mask(prob_mask_R, desired_size=(img.shape[1], img.shape[0]), threshold=threshold)

        mask_uint8 = np.uint8(mask_R * 255)

        # Извлечение контуров из маски
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Расчет порога площади для фильтрации контуров
        image_area = img.shape[1] * img.shape[0]
        area_threshold = image_area * area_ratio

        # Добавляем контуры в список, если их площадь больше порога
        contours_threshold = [contour.reshape(-1, 2) for contour in contours if
                              cv2.contourArea(contour) > area_threshold]

        boxes = []
        for contour in contours_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))

        # Формирование результата в формате YOLO
        results_final = {
            'boxes_xyxy': boxes,
            'masks': contours_threshold,
            'cls': [0] * len(boxes),
            'probs': [0.7] * len(boxes),
        }

        class_names = {0: 'rail'}

        return results_final, class_names


    def predict_on_image_frame(
        self,
        img: np.ndarray,
        task: str,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        area_ratio: float = 0.002
    ) -> Tuple:
        """Функция выполняет инференс модели YOLO на изображении и возвращает результаты в виде [boxes, masks, cls,
        probs]. Маски в реальном размере относительно изображения.

        Args:
            img (np.ndarray): Исходное изображение.
            task (str): Название задачи детекции, с которой работает модель (PJ, LWT, R).
            iou (float): Значение Intersection Over Union. float=0.5.
            conf (float): Значение уверенности в точности предсказанного класса. float=0.5.
            imgsz (int): Размер входного изображения.
            verbose (bool): Выводить логи предсказания модели или нет.
            area_ratio (float): Порога площади для фильтрации контуров рельсов.

        Returns:
            results (dict): Результаты детекции объектов.
            class_names (dict): Названия классов.
        """
        result = self.model(img, iou=iou, conf=conf, imgsz=imgsz, verbose=verbose)[0]

        # detection
        cls = result.boxes.cls.cpu().numpy()
        class_names = result.names
        probs = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        results = {'boxes_xyxy': boxes, 'masks': [], 'cls': cls, 'probs': probs}

        # segmentation
        try:
            masks = result.masks.data.cpu().numpy()  # masks, (N, H, W)
            masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
            # rescale masks to original image
            masks = scale_image(masks, result.orig_shape)
            masks = np.moveaxis(masks, -1, 0)  # masks, (N, H, W)
            # Перевод masks в polygons
            if task == 'R':
                # Расчет порога площади для фильтрации контуров
                image_area = img.shape[1] * img.shape[0]
                area_threshold = image_area * area_ratio
                masks_countour_list = []
                for mask in masks:
                    # Добавляем контуры в список, если их площадь больше порога
                    mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                    contours_list = [contour.reshape(-1, 2) for contour in mask_contours if
                                     cv2.contourArea(contour) > area_threshold]
                    for contour in contours_list:
                        masks_countour_list.append(contour)
                # Добавляем координаты масок в словарь
                results['masks'] = masks_countour_list
            else:
                masks = [transform_mask_2_max_polygon(mask) for mask in masks]
                # Добавляем координаты масок в словарь
                results['masks'] = masks
        except:
            # Записываю значение None для масок в словарь
            results['masks'] = [None] * len(results['boxes_xyxy'])

        return results, class_names


    def predict_rails(
        self,
        img: np.ndarray,
        unet_enabled: bool,
        r_unet_params: dict,
        rail_model_params: dict,
    ):
        """
        Функция детектирует рельсы с помощью модели YOLO или Unet. Обрабатывает результаты и предоставляет их в корректном виде.

        Args:
            img (np.ndarray): Входное изображение.
            unet_enabled (bool): Использовать модель Unet или YOLO.
            r_unet_params (dict): Параметры для модели Unet.
            rail_model_params (dict): Параметры для модели YOLO.

        Returns:
            results_R (dict): Результаты детекции рельсов.
            names_R (dict): Названия классов.
        """
        if unet_enabled:
            results_R, names_R = self._predict_on_image_unet(img, **r_unet_params)
        else:
            results_R, names_R = self.predict_on_image_frame(img, task='R', **rail_model_params)

        return results_R, names_R


class InferencerWorld (Inferencer):
    def __init__(self, model, text_prompts: List):
        super().__init__(model)

        # Промты для поиска кастомных классов
        self.text_prompts = text_prompts

        if len(self.text_prompts) != 0:
            self.model.set_classes(self.text_prompts)
