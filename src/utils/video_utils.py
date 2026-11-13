from time import time

from pathlib import Path
from typing import Deque, Tuple

import cv2


def save_video(output: str, video_name: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    """Создает директорию output, если она не существует, и возвращает объект VideoWriter.

    Args:
        output: Путь к директории, куда будет сохранено видео. Будет создана, если не существует.
        video_name: Имя выходного файла видео.
        cap: Объект VideoCapture, используемый для получения свойств видео (fps, разрешение и т.д.).

    Returns:
        Настроенный объект VideoWriter, готовый к обработке видео.
"""

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / video_name

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    return cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size
    )


def calculate_average_fps(frame_history: Deque[Tuple[float, int]]) -> float:
    """Вычисляет средний FPS на основе истории кадров.

    Args:
        frame_history (Deque[Tuple[float, int]]): История кадров.

    Returns:
        float: Средний FPS.

    """
    if len(frame_history) < 30:
        return 0

    oldest_time, oldest_frames = frame_history[0]
    newest_time, newest_frames = frame_history[-1]

    time_diff = newest_time - oldest_time
    frame_diff = newest_frames - oldest_frames

    return frame_diff / time_diff if time_diff > 0 else 0


def calculate_fps(
    total_frames: int,
    prev_frame_time: float,
    frame_history: Deque[Tuple[float, int]],
) -> Tuple[float, float, float]:
    """Функция рассчитывает количество кадров в секунду (FPS).

    Args:
        total_frames (int): Общее количество прочитанных кадров.
        prev_frame_time (float): Время отображения предыдущего кадра.
        frame_history (Deque[Tuple[float, int]]): История кадров.

    Returns:
        Tuple[float, float, float]: (prev_frame_time, frame_fps, avg_fps).
    """
    new_frame_time = time()
    elapsed_time = new_frame_time - prev_frame_time
    frame_fps = 1 / elapsed_time

    # Обновляем историю кадров
    frame_history.append((new_frame_time, total_frames))

    # Вычисляем средний FPS
    avg_fps = calculate_average_fps(frame_history)

    return new_frame_time, frame_fps, avg_fps
