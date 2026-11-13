from abc import ABC, abstractmethod
import logging
from threading import Thread
from time import sleep

from src import ThreadStopEventException

# Инициализация логгера
logger = logging.getLogger(__name__)


class BaseFrameReader(ABC, Thread):
    def __init__(self, frame_queue, stop_event):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.total_read_frames = 0
        self.failures = 0

    @abstractmethod
    def _read_frame(self):
        """Прочитать один кадр из источника."""
        pass

    @abstractmethod
    def _reconnect(self):
        """Восстановить соединение с источником при необходимости."""
        pass

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self._read_frame()
                if frame is not None and self.frame_queue.qsize() < self.frame_queue.maxsize:
                    self.frame_queue.put(frame, timeout=1.0)

                    # Сброс счетчика ошибок
                    if self.failures > 0:
                        self.failures = 0
                        logger.info("Сброшен счетчик ошибок чтения кадров.")
                        sleep(0.01)

                    self.total_read_frames += 1
            except Exception as e:
                logger.error(f"Ошибка чтения кадра: {e}")
                self.failures += 1
                if self.failures > 5:
                    logger.critical("Превышено количество ошибок чтения кадров. Остановка чтения.")
                    self.stop_event.set()
                    raise ThreadStopEventException()
