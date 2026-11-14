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
        # Token to record events in Backend API
        self.token = token
        # Login and password for the ai account
        self.login = login
        self.password = password

        # The importance of business logic counters,
        # after which the activation information will be recorded in the database
        self.triggers_count = triggers_count
        self.triggers_count_task_4_2 = triggers_count_task_4_2
        self.triggers_count_task_5_2 = triggers_count_task_5_2
        # Time to be checked for response
        # Business logic, do you need to record anything in the database in seconds
        self.check_db_time = db_check_time
        # Video start time
        self.start_time = time()
        # Business logic task dictionary
        self.task_logic_trigger_count = {"1_2": 0, "1_3": 0, "3_1": 0,
                                         "3_3": 0, "4_2": 0, "5_2": 0,
                                         "6_1": 0, "6_2": 0, "7_1": 0, "7_2_1": 0,
                                         "7_2_2": 0, "8_1_1": 0, "9": 0,
                                         "10_1": 0, "10_2": 0,
                                         }
        # Vocabulary, do you want to record a situation in an database where the key is a task number, the value is time
        # Business logic
        self.task_write_to_db = {"1_2": 0, "1_3": 0, "3_1": 0,
                                 "3_3": 0, "4_2": 0, "5_2": 0,
                                 "6_1": 0, "6_2": 0, "7_1": 0,
                                 "7_2_1": 0, "7_2_2": 0, "8_1_1": 0,
                                 "9": 0, "10_1": 0, "10_2": 0,
                                 }
        # The dictionary in which the key is a task number, the value is np.ndarray of the image on which
        # See how business logic works.
        self.task_snapshot_write_to_db = {"1_2": 0, "1_3": 0, "3_1": 0,
                                          "3_3": 0, "4_2": 0, "5_2": 0,
                                          "6_1": 0, "6_2": 0, "7_1": 0,
                                          "7_2_1": 0, "7_2_2": 0, "8_1_1": 0,
                                          "9": 0, "10_1": 0, "10_2": 0,
                                          }
        # Alerts-category dictionary
        self.categories_ids = {"1_2": 1, "1_3": 2, "3_1": 1,
                               "3_3": 1, "4_2": 1, "5_2": 1,
                               "6_1": 1, "6_2": 2, "7_1": 2,
                               "7_2_1": 2, "7_2_2": 2, "8_1_1": 2,
                               "9": 2, "10_1": 2, "10_2": 2,
                               }

        # A dictionary with ID people
        self.people_ids = {"1_2": 0, "1_3": 0, "3_1": 0,
                           "3_3": 0, "4_2": 0, "5_2": 0,
                           "6_1": 0, "6_2": 0, "7_1": 0,
                           "7_2_1": 0, "7_2_2": 0, "8_1_1": 0,
                           "9": None, "10_1": 0, "10_2": 0,
                           }

        # Initialization of parameters - is the business task record included
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

        # Determination of business logic response times
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
        """Auxiliary function to process a complex if status when operating a core function to process that function,
           How many times has the business logic of a given task worked in a row"""
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
        # Addition of categories of hazardous areas to categories_ids dictionary
        self.categories_ids.update({"1_2_" + str(key): 1 for key, _ in danger_zones_cords.items()})
        # Support dictionary with ID hazardous areas for the task 1.2
        zones_ids_1_2_dict = {"1_2_" + str(key): 0 for key, _ in danger_zones_cords.items()}
        # Addition to task_logic_trigger_count for each dangerous area
        self.task_logic_trigger_count.update(zones_ids_1_2_dict)
        # Addition to task_write_to_db for each dangerous area
        self.task_write_to_db.update(zones_ids_1_2_dict)
        # Addition to task_snapshot_write_to_db for each dangerous area
        self.task_snapshot_write_to_db.update(zones_ids_1_2_dict)
        # Adding a value to a dictionary with ID for each dangerous area
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
        # Fixing the business logic response time.
        self.task_write_to_db[key] = 0

        # Key to human ID and mission category
        person_id_key = category_id_key = key
        # If the logic did not cross the dangerous area (the task 1.2 did not work))
        zone_id = None
        # If an ID dangerous area is present in the task name (task 1.2))
        if key.startswith("1_2"):
            # Battering the mission name from the ID of the dangerous area to a separate name and ID
            key = key.split("_")
            # ID of the hazardous zone
            zone_id = int(key[-1])
            # Task number
            key = '_'.join(key[:-1])

        print(f"Logistics {key} In database, time: {value}")
        # Log Log Log Log Log Log Logs at the Database
        # Options that are recorded in the People table
        values_events = {"timestamp": value, "date_time": value, "cam_id": cam_id,
                         "zone_id": zone_id, "event": key.replace("_", "."),
                         "people_id": self.people_ids[person_id_key],
                         "category_id": self.categories_ids[category_id_key]}

        # Options that are recorded in the snapshots table
        snapshot = self._jpegcompression(self.task_snapshot_write_to_db[person_id_key], 10, convert_snap2bytes)
        values_snapshots = {"timestamp": value, "snapshot": snapshot}

        return values_events, values_snapshots

    # ====================================================================================#
    # Methods for integration of Backend API                              #
    # ====================================================================================#
