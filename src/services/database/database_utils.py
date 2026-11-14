from datetime import datetime
from typing import Tuple

import numpy as np


# ============================================================#
# Functions for working with the Data Base #
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
    """Auxiliary function to process a complex if status when operating a core function to process that function,
       How many times has the business logic of a given task worked in a row"""
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
    """The function processes the logic processing of business tasks and records the processing results in dictionaries for further acceptance
    Decisions on whether to record results in the Database

    Input parameters:
        img (np.ndarray): The frame on which the BBoxes and Masks will be applied.
        task_trigger (bool): Value of True or False. The logic of the business problem on the video shot worked or not.
        task_trigger_time (datetime): Trigger response time.
        task_logic_trigger_count (dict): a dictionary that keeps the meanings of how many times a business task has been performed in a row.
        task_write_to_db (dict): a dictionary that keeps in itself values whether to record in the database information based on the logic or not.
        task_snapshot_write_to_db (dict): The dictionary that keeps a snapshot of each logic if it worked on a frame, otherwise 0.
        triggers_count (int): The value of the business logic computing calculator, after which the computing information will be recorded in the BD.
        triggers_count_task_4_2 (int): Business logic counter value 4.2.
        triggers_count_task_5_2 (int): Business logic counter value 5.2.
        task_number (str): Business challenge number for PMI.

    Output parameters:
        task_logic_trigger_count (dict): up-to-date dictionary.
        task_write_to_db (dict): up-to-date dictionary.
        task_snapshot_write_to_db (dict): up-to-date dictionary.
    """
    # 4.2 is entered in the task database if the task has worked at least triggers_count_task_4_2 times
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
