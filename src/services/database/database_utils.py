from datetime import datetime
from typing import Tuple

import numpy as np


# ============================================================#
#            Функции для работы с Базой Данных                #
# ============================================================#
def proccess_if_statement(
        img: np.ndarray,
        task_trigger: bool,
        task_trigger_time: datetime,
        task_logic_trigger_count: dict,
        task_write_to_db: dict,
        task_snapshot_write_to_db: dict,
        triggers_count: int,
        task_number: str
):
    """Вспомогательная функция для обработки сложного if statement при работе основной функции для обработки того,
       сколько раз подряд сработала бизнес-логика определенной задачи"""
    if task_trigger and task_logic_trigger_count[task_number] < triggers_count:
        task_logic_trigger_count[task_number] += 1
        if task_logic_trigger_count[task_number] == triggers_count:
            task_write_to_db[task_number] = task_trigger_time
            task_snapshot_write_to_db[task_number] = img
    else:
        task_logic_trigger_count[task_number] = 0


def trigger_process(img: np.ndarray,
                    task_trigger: bool,
                    task_trigger_time: datetime,
                    task_logic_trigger_count: dict,
                    task_write_to_db: dict,
                    task_snapshot_write_to_db: dict,
                    triggers_count: int,
                    triggers_count_task_4_2: int,
                    triggers_count_task_5_2: int,
                    task_number: str) -> Tuple:
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
        proccess_if_statement(img, task_trigger, task_trigger_time, task_logic_trigger_count,
                              task_write_to_db, task_snapshot_write_to_db, triggers_count_task_4_2, "4_2")
    elif task_number == "5_2":
        proccess_if_statement(img, task_trigger, task_trigger_time, task_logic_trigger_count,
                              task_write_to_db, task_snapshot_write_to_db, triggers_count_task_5_2, "5_2")
    else:
        proccess_if_statement(img, task_trigger, task_trigger_time, task_logic_trigger_count,
                              task_write_to_db, task_snapshot_write_to_db, triggers_count, task_number)

    return task_logic_trigger_count, task_write_to_db, task_snapshot_write_to_db
