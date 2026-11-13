from datetime import datetime
from time import time
from typing import Tuple

import cv2
import numpy as np


class DataBase:
    def __init__(self, token, login, password, triggers_count,
                 triggers_count_task_4_2, triggers_count_task_5_2,
                 db_check_time, task_1_2_enabled, task_1_3_enabled,
                 task_3_1_enabled, task_3_3_enabled, task_4_2_enabled,
                 task_5_2_enabled, task_6_1_enabled, task_6_2_enabled,
                 task_7_enabled, task_8_1_1_enabled, task_9_enabled,
                 task_10_1_enabled, task_10_2_enabled
                 ):
        # Токен для возможности записи событий в Backend API
        self.token = token
        # Логин и пароль от аккаунта с ролью ai
        self.login = login
        self.password = password

        # Значение счетчиков срабатываний бизнес-логики,
        # после которого информация о срабатывании будет записана в БД
        self.triggers_count = triggers_count
        self.triggers_count_task_4_2 = triggers_count_task_4_2
        self.triggers_count_task_5_2 = triggers_count_task_5_2
        # Промежуток времени, в который проверяются срабатывания
        # бизнес-логик, надо ли что-либо записывать в БД в секундах
        self.check_db_time = db_check_time
        # Время старта видео
        self.start_time = time()
        # Словарь-счетчик срабатывания бизнес-логик задач
        self.task_logic_trigger_count = {"1_2": 0, "1_3": 0, "3_1": 0,
                                         "3_3": 0, "4_2": 0, "5_2": 0,
                                         "6_1": 0, "6_2": 0, "7_1": 0, "7_2_1": 0,
                                         "7_2_2": 0, "8_1_1": 0, "9": 0,
                                         "10_1": 0, "10_2": 0,
                                         }
        # Словарь, записывать ли ситуацию в БД, где ключ - номер task, значение - время
        # срабатывания бизнес логики
        self.task_write_to_db = {"1_2": 0, "1_3": 0, "3_1": 0,
                                 "3_3": 0, "4_2": 0, "5_2": 0,
                                 "6_1": 0, "6_2": 0, "7_1": 0,
                                 "7_2_1": 0, "7_2_2": 0, "8_1_1": 0,
                                 "9": 0, "10_1": 0, "10_2": 0,
                                 }
        # Словарь, в котором ключ - номер task, значение - np.ndarray изображения, на котором
        # видно срабатывание бизнес-логики
        self.task_snapshot_write_to_db = {"1_2": 0, "1_3": 0, "3_1": 0,
                                          "3_3": 0, "4_2": 0, "5_2": 0,
                                          "6_1": 0, "6_2": 0, "7_1": 0,
                                          "7_2_1": 0, "7_2_2": 0, "8_1_1": 0,
                                          "9": 0, "10_1": 0, "10_2": 0,
                                          }
        # Словарь с категориями Alerts
        self.categories_ids = {"1_2": 1, "1_3": 2, "3_1": 1,
                               "3_3": 1, "4_2": 1, "5_2": 1,
                               "6_1": 1, "6_2": 2, "7_1": 2,
                               "7_2_1": 2, "7_2_2": 2, "8_1_1": 2,
                               "9": 2, "10_1": 2, "10_2": 2,
                               }

        # Словарь с ID людей
        self.people_ids = {"1_2": 0, "1_3": 0, "3_1": 0,
                           "3_3": 0, "4_2": 0, "5_2": 0,
                           "6_1": 0, "6_2": 0, "7_1": 0,
                           "7_2_1": 0, "7_2_2": 0, "8_1_1": 0,
                           "9": None, "10_1": 0, "10_2": 0,
                           }

        # Инициализация параметров - включена ли запись бизнес-задачи
        self.task_1_2_enabled = task_1_2_enabled
        self.task_1_3_enabled = task_1_3_enabled
        self.task_3_1_enabled = task_3_1_enabled
        self.task_3_3_enabled = task_3_3_enabled
        self.task_4_2_enabled = task_4_2_enabled
        self.task_5_2_enabled = task_5_2_enabled
        self.task_6_1_enabled = task_6_1_enabled
        self.task_6_2_enabled = task_6_2_enabled
        self.task_7_enabled = task_7_enabled
        self.task_8_1_1_enabled = task_8_1_1_enabled
        self.task_9_enabled = task_9_enabled
        self.task_10_1_enabled = task_10_1_enabled
        self.task_10_2_enabled = task_10_2_enabled

        # Предопределение времен срабатывания бизнес-логик
        self.triggered_1_2 = self.triggered_time_1_2 = None
        self.triggered_1_3 = self.triggered_time_1_3 = None
        self.triggered_3_1 = self.triggered_time_3_1 = None
        self.triggered_3_3 = self.triggered_time_3_3 = None
        self.triggered_4_2 = self.triggered_time_4_2 = None
        self.triggered_5_2 = self.triggered_time_5_2 = None
        self.triggered_6_1 = self.triggered_time_6_1 = None
        self.triggered_6_2 = self.triggered_time_6_2 = None
        self.triggered_7_1 = self.triggered_time_7_1 = None
        self.triggered_7_2_1 = self.triggered_time_7_2_1 = None
        self.triggered_7_2_2 = self.triggered_time_7_2_2 = None
        self.triggered_8_1_1 = self.triggered_time_8_1_1 = None
        self.triggered_9 = self.triggered_time_9 = None
        self.triggered_10_1 = self.triggered_time_10_1 = None
        self.triggered_10_2 = self.triggered_time_10_2 = None


    @staticmethod
    def proccess_if_statement(
        img: np.ndarray,
        task_trigger: bool,
        task_trigger_time: datetime,
        task_logic_trigger_count: dict,
        task_write_to_db: dict,
        task_snapshot_write_to_db: dict,
        triggers_count: int,
        task_number: str
    ) -> Tuple:
        """Вспомогательная функция для обработки сложного if statement при работе основной функции для обработки того,
           сколько раз подряд сработала бизнес-логика определенной задачи"""
        if task_trigger and task_logic_trigger_count[task_number] < triggers_count:
            task_logic_trigger_count[task_number] += 1
            if task_logic_trigger_count[task_number] == triggers_count:
                task_write_to_db[task_number] = task_trigger_time
                task_snapshot_write_to_db[task_number] = img
        else:
            task_logic_trigger_count[task_number] = 0


    def _trigger_process(
        self,
        img: np.ndarray,
        task_trigger: bool,
        task_trigger_time: datetime,
        task_logic_trigger_count: dict,
        task_write_to_db: dict,
        task_snapshot_write_to_db: dict,
        triggers_count: int,
        triggers_count_task_4_2: int,
        triggers_count_task_5_2: int,
        task_number: str
    ) -> Tuple:
        """Функция обрабатывает срабатывание логик бизнес-задач и записывает результаты обработки в словари для дальнейшего принятия
        решения, записывать ли результаты в Базу Данных

        Входные параметры:
            img (np.ndarray): Кадр, на который будут нанесены BBoxes и Masks.
            task_trigger (bool): Значение True или False. Сработала логика бизнес-задачи на кадре видео или нет.
            task_trigger_time (datetime): Время срабатывания триггера.
            task_logic_trigger_count (dict): Словарь, хранящий в себе значения, сколько раз подряд сработала каждая бизнес-задача.
            task_write_to_db (dict): Словарь, хранящий в себе значения, записывать ли в базу данных информацию по отработанной логике или нет.
            task_snapshot_write_to_db (dict): Словарь, хранящий в себе snapshot каждой логики, если она отработала на кадре, иначе 0.
            triggers_count (int): Значение счетчика срабатываний бизнес-логики, после которого информация о срабатывании будет записана в БД.
            triggers_count_task_4_2 (int): Значение счетчика срабатываний бизнес-логики 4.2.
            triggers_count_task_5_2 (int): Значение счетчика срабатываний бизнес-логики 5.2.
            task_number (str): Номер бизнес-задачи по ПиМи.

        Выходные параметры:
            task_logic_trigger_count (dict): Словарь с обновленными значениями.
            task_write_to_db (dict): Словарь с обновленными значениями.
            task_snapshot_write_to_db (dict): Словарь с обновленными значениями.
        """
        # Запись в базу данных задачи 4.2 осуществляется, если задача сработала хотя бы triggers_count_task_4_2 раз
        if task_number == "4_2":
            self.proccess_if_statement(img, task_trigger, task_trigger_time, task_logic_trigger_count,
                                       task_write_to_db, task_snapshot_write_to_db, triggers_count_task_4_2, "4_2")
        elif task_number == "5_2":
            self.proccess_if_statement(img, task_trigger, task_trigger_time, task_logic_trigger_count,
                                       task_write_to_db, task_snapshot_write_to_db, triggers_count_task_5_2, "5_2")
        else:
            self.proccess_if_statement(img, task_trigger, task_trigger_time, task_logic_trigger_count,
                                       task_write_to_db, task_snapshot_write_to_db, triggers_count, task_number)

        return task_logic_trigger_count, task_write_to_db, task_snapshot_write_to_db


    def update_task_1_2_dir(self, danger_zones_cords):
        # Добавление категорий опасных зон в словарь categories_ids
        self.categories_ids.update({"1_2_" + str(key): 1 for key, _ in danger_zones_cords.items()})
        # Вспомогательный словарь с ID опасных зон для задачи 1.2
        zones_ids_1_2_dict = {"1_2_" + str(key): 0 for key, _ in danger_zones_cords.items()}
        # Добавление в task_logic_trigger_count значения для каждой опасной зоны
        self.task_logic_trigger_count.update(zones_ids_1_2_dict)
        # Добавление в task_write_to_db значения для каждой опасной зоны
        self.task_write_to_db.update(zones_ids_1_2_dict)
        # Добавление в task_snapshot_write_to_db значения для каждой опасной зоны
        self.task_snapshot_write_to_db.update(zones_ids_1_2_dict)
        # Добавление в словарь с ID людей значения для каждой опасной зоны
        self.people_ids.update(zones_ids_1_2_dict)


    def process_1_2(self, img, task_number):
        if self.task_1_2_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_1_2,
                                                                                                                   self.triggered_time_1_2,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number=task_number)


    def process_business_tasks(self, img):
        # task_1_3
        if self.task_1_3_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_1_3,
                                                                                                                   self.triggered_time_1_3,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="1_3")
        # task_3_1
        if self.task_3_1_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_3_1,
                                                                                                                   self.triggered_time_3_1,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="3_1")

        # task_3_3
        if self.task_3_3_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_3_3,
                                                                                                                   self.triggered_time_3_3,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="3_3")
        # task_4_2
        if self.task_4_2_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_4_2,
                                                                                                                   self.triggered_time_4_2,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="4_2")
        # task_5_2
        if self.task_5_2_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_5_2,
                                                                                                                   self.triggered_time_5_2,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="5_2")

        # task_6_1
        if self.task_6_1_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_6_1,
                                                                                                                   self.triggered_time_6_1,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="6_1")

        # task_6_2
        if self.task_6_2_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_6_2,
                                                                                                                   self.triggered_time_6_2,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="6_2")
        # task_7
        if self.task_7_enabled:
            # task_7_1
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_7_1,
                                                                                                                   self.triggered_time_7_1,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="7_1")
            # task_7_2_1
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_7_2_1,
                                                                                                                   self.triggered_time_7_2_1,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="7_2_1")
            # task_7_2_2
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_7_2_2,
                                                                                                                   self.triggered_time_7_2_2,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="7_2_2")
        # task_8_1_1
        if self.task_8_1_1_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_8_1_1,
                                                                                                                   self.triggered_time_8_1_1,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="8_1_1")
        # task_9
        if self.task_9_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_9,
                                                                                                                   self.triggered_time_9,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="9")
        # task_10_1
        if self.task_10_1_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_10_1,
                                                                                                                   self.triggered_time_10_1,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="10_1")
        # task_10_2
        if self.task_10_2_enabled:
            self.task_logic_trigger_count, self.task_write_to_db, self.task_snapshot_write_to_db = self._trigger_process(img,
                                                                                                                   self.triggered_10_2,
                                                                                                                   self.triggered_time_10_2,
                                                                                                                   self.task_logic_trigger_count,
                                                                                                                   self.task_write_to_db,
                                                                                                                   self.task_snapshot_write_to_db,
                                                                                                                   self.triggers_count,
                                                                                                                   self.triggers_count_task_4_2,
                                                                                                                   self.triggers_count_task_5_2,
                                                                                                                   task_number="10_2")


    @staticmethod
    def _jpegcompression(image: np.ndarray, quality: int, convert2bytes: bool = False) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image, encode_param)

        return encimg.tobytes() if convert2bytes else encimg


    def prepare_vals_for_insert(self, key, value, cam_id, convert_snap2bytes: bool = False):
        # Обнуляем время срабатывания логики бизнес-задачи
        self.task_write_to_db[key] = 0

        # Ключ к ID человека и категории по названию задачи
        person_id_key = category_id_key = key
        # Если в логике не было пересечения опасной зоны (не отработала задача 1.2)
        zone_id = None
        # Если в названии задачи присутствует ID опасной зоны (задача 1.2)
        if key.startswith("1_2"):
            # Разбивание названия задачи с ID опасной зоны на отдельные название и ID
            key = key.split("_")
            # ID опасной зоны
            zone_id = int(key[-1])
            # Номер задачи
            key = '_'.join(key[:-1])

        print(f"Запись логики {key} в БД, время: {value}")
        # Логика записи лога в Базу Данных
        # Параметры которые записываются в таблицу people
        values_events = {"timestamp": value, "date_time": value, "cam_id": cam_id,
                         "zone_id": zone_id, "event": key.replace("_", "."),
                         "people_id": self.people_ids[person_id_key],
                         "category_id": self.categories_ids[category_id_key]}

        # Параметры которые записываются в таблицу snapshots
        snapshot = self._jpegcompression(self.task_snapshot_write_to_db[person_id_key], 10, convert_snap2bytes)
        values_snapshots = {"timestamp": value, "snapshot": snapshot}

        return values_events, values_snapshots

    # ====================================================================================#
    #                      Методы для интеграции Backend API                              #
    # ====================================================================================#
