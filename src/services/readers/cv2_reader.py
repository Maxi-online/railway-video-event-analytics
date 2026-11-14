from src.services.readers.base_reader import BaseFrameReader
import cv2
import time
from pathlib import Path
import logging

# Initialization of Logger
logger = logging.getLogger("CV2FrameReader")


class CV2FrameReader(BaseFrameReader):
    def __init__(self, file_path, frame_queue, stop_event, skip_interval=1):
        super().__init__(frame_queue, stop_event)
        self.file_path = file_path

        # Checking the existence of the file
        self._check_filepath()

        self.skip_interval = skip_interval
        self.cap = cv2.VideoCapture(file_path)

    def _check_filepath(self):
        msg = "The video isn't found or doesn't exist!"
        if not Path(self.file_path).exists():
            logger.error(msg)
            raise FileNotFoundError(msg)

    def _read_frame(self):
        if self.total_read_frames % self.skip_interval != 0:
            self.cap.grab()
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        return frame

    def _reconnect(self):
        self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.file_path)
