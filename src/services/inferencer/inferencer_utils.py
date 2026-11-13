from typing import Tuple
import numpy as np
import cv2


def sigmoid(x):
    """
    Функция сигмоид.

    Эта функция принимает в качестве аргумента число или массив чисел и возвращает
    результат применения сигмоидной функции. Сигмоидная функция описывается формулой:

        S(x) = 1 / (1 + exp(-x))

    где exp(-x) — это математическая константа e, возведенная в степень -x.

    Параметры:
        x (float или np.array): Входное значение или массив значений.

    Возвращает:
        float или np.array: Значение сигмоидной функции для каждого элемента входа.
    """
    return 1 / (1 + np.exp(-x))  # Возвращаем результат сигмоидной функции


def create_binary_mask(
        prob_mask: np.ndarray,
        desired_size: Tuple[int, int],
        threshold: float = 0.4,
) -> np.ndarray:
    """
    Функция для бинаризации маски согласно порогу и изменения ее размера до desired_size.

    Входные параметры:
        prob_mask (np.ndarray): Маска вероятности.
        desired_size (Tuple[int, int]): Желаемый размер маски.
        threshold (float): Порог для бинаризации маски.

    Выходные параметры:
        (np.ndarray): Бинаризованная маска после изменения размера.
    """
    mask = (prob_mask > threshold).astype(np.uint8)
    mask = cv2.resize(mask, desired_size, interpolation=cv2.INTER_NEAREST)

    return mask
