from abc import ABC, abstractmethod
import logging
from threading import Thread
from time import sleep

from src import ThreadStopEventException

# Initialization of Logger
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
        'Read one frame from the source.'
        pass

    @abstractmethod
    def _reconnect(self):
        'Re-establish the connection to the source if necessary.'
        pass

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self._read_frame()
                if frame is not None and self.frame_queue.qsize() < self.frame_queue.maxsize:
                    self.frame_queue.put(frame, timeout=1.0)

                    # Removing the error calculator
                    if self.failures > 0:
                        self.failures = 0
                        logger.info('Lost calculator of read errors.')
                        sleep(0.01)

                    self.total_read_frames += 1
            except Exception as e:
                logger.error(f"Error in reading of the picture: {e}")
                self.failures += 1
                if self.failures > 5:
                    logger.critical('The number of errors in reading. Stopping Reading.')
                    self.stop_event.set()
                    raise ThreadStopEventException()
