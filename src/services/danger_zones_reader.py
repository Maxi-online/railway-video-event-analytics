import logging
from time import sleep
from threading import Thread
import json

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

logger = logging.getLogger("DangerZonesReader")


class DangerZonesReader(Thread):
    def __init__(self, kafka_address, topic_name, api_client, data_base, cam_id, stop_event, delay=0.1):
        super().__init__(daemon=True)

        self.kafka_address = kafka_address
        self.topic_name = topic_name

        self.api_client = api_client
        self.data_base = data_base
        self.cam_id = cam_id
        self.stop_event = stop_event
        self.delay = self._validate_delay(delay)
        self.target_keys = [f"state_camera_slot_{cam_id}", "state_camera_slot_all"]

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
            group_id=f"camera_group_slot_{self.cam_id}",
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None,
        )


    def _reconnect(self):
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts and not self.stop_event.is_set():
            try:
                attempt += 1
                logger.warning(f"Reconnect attempt {attempt}/{max_attempts}")

                if hasattr(self, 'consumer'):
                    self.consumer.close()

                self.consumer = self._create_consumer()
                logger.info('Reconnected successfully completed')
                return

            except Exception as e:
                logger.error(f"Unable to reconnect: (attempted) {attempt}): {str(e)}")
                if attempt < max_attempts:
                    sleep(2 ** attempt)

        if not self.stop_event.is_set():
            logger.critical('Reconnect attempt limit achieved. Stop the flow.')
            self.stop_event.set()


    def _validate_delay(self, delay):
        try:
            return abs(float(delay))
        except (TypeError, ValueError):
            logger.warning(f"Error delay value, default value used: 0.1")
            return 0.1


    def _process_message(self, message):
        if message.key not in self.target_keys:
            return False

        try:
            if self.api_client.read_camera_data(message, self.cam_id):
                zones = self.api_client.get_danger_zones()
                if zones is not None:
                    self.data_base.update_task_1_2_dir(zones)
                    return True
        except Exception as e:
            logger.error(f"Error in communication processing: {e}", exc_info=True)
        return False


    def run(self):
        logger.info(f"The reading stream of the dangerous areas for the camera is started {self.cam_id}")

        while not self.stop_event.is_set():
            try:
                messages = self.consumer.poll(timeout_ms=1000)
                if not messages:
                    sleep(0.1)
                    continue

                for tp, msgs in messages.items():
                    for msg in msgs:
                        if msg and msg.value:
                            self._process_message(msg)

                if self.delay > 0:
                    sleep(self.delay)

            except Exception as e:
                logger.error(f"Error in reading messages: {e}", exc_info=True)
                sleep(self.delay)

        logger.info(f"The reading of dangerous areas for the camera has been stopped {self.cam_id}")
