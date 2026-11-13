from src.services.readers.base_reader import BaseFrameReader
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable, KafkaError
import cv2
import numpy as np
import time
import logging

# Инициализация логгера
logger = logging.getLogger("KafkaFrameReader")


class KafkaFrameReader(BaseFrameReader):
    def __init__(self, kafka_address, topic_name, cam_id, frame_queue, stop_event, skip_interval=1):
        super().__init__(frame_queue, stop_event)
        self.kafka_address = kafka_address
        self.topic_name = topic_name
        self.cam_id = cam_id
        self.skip_interval = skip_interval

        try:
            self.consumer = self._create_consumer()
        except NoBrokersAvailable:
            self._reconnect()

    def _create_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            self.topic_name,
            bootstrap_servers=self.kafka_address,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
            group_id=f'video_consumer_group_slot_{self.cam_id}',
        )

    def _read_frame(self):
        if self.total_read_frames % self.skip_interval != 0:
            return None

        try:
            message = next(self.consumer)
            np_arr = np.frombuffer(message.value, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except KafkaError:
            self._reconnect()

    def _reconnect(self):
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts and not self.stop_event.is_set():
            try:
                attempt += 1
                logger.warning(f"Попытка переподключения {attempt}/{max_attempts}")

                if hasattr(self, 'consumer'):
                    self.consumer.close()

                self.consumer = self._create_consumer()
                logger.info("Переподключение успешно завершено")
                return

            except Exception as e:
                logger.error(f"Переподключение не удалось: (попытка {attempt}): {str(e)}")
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)

        if not self.stop_event.is_set():
            logger.critical("Достигнут лимит попыток переподключения. Останавливаем поток.")
            self.stop_event.set()
