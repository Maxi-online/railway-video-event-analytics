from time import time

from pathlib import Path
from typing import Deque, Tuple

import cv2


def save_video(output: str, video_name: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    """Creates an output directory if it doesn't exist and returns a VideoWriter object.

    Args:
        output: Path to the directory where the video will be saved. To be created if not in existence.
        video_name: The name of the video output file.
        cap: VideoCapture for video properties (fps, resolution, etc.).).

    Returns:
        VideoWriter, set up for video processing.
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
    """Calculates average FPS based on human history.

    Args:
        frame_history (Deque[Tuple[float,int]]]: human history.

    Returns:
        float: Medium FPS.

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
    """The function calculates the number of people per second (FS).

    Args:
        total_frames (int): Total number of people read.
        prev_frame_time (float): Presentation time of the previous frame.
        frame_history (Deque[Tuple[float,int]]]: human history.

    Returns:
        Tuple[float, float, float]: (prev_frame_time, frame_fps, avg_fps).
    """
    new_frame_time = time()
    elapsed_time = new_frame_time - prev_frame_time
    frame_fps = 1 / elapsed_time

    # Update human resources history
    frame_history.append((new_frame_time, total_frames))

    # Compute average FPS
    avg_fps = calculate_average_fps(frame_history)

    return new_frame_time, frame_fps, avg_fps
