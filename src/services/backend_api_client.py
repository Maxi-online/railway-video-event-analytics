import logging
import threading
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BackendAPIClient:
    def __init__(self, token, login, password, danger_zones_cords):
        # Токен для возможности записи событий в Backend API
        self.token = token
        # Логин и пароль от аккаунта с ролью ai
        self.login = login
        self.password = password
        
        # Координаты опасных зон
        self.danger_zones_cords = danger_zones_cords
        self._zones_lock = threading.Lock()  # Lock для потокобезопасного доступа к координатам опасных зон

    def _get_token(self, backend_api_base_url: str):
        login_url = backend_api_base_url + "/video_events/api/v1/auth/login"
        login_data = {
            "login": self.login,
            "password": self.password
        }

        try:
            response = requests.post(login_url, json=login_data).json()["token"]
        except Exception:
            print("API недоступна. Не получилось обновить токен.")
            return None
        
        return response

    # Функция для записи всех событий в Backend API в отдельном потоке
    def _create_events_and_send_snapshots(self,
                                          data_base,
                                          backend_api_base_url: str,
                                          cam_id: int):

        for key, value in data_base.task_write_to_db.items():
            if value != 0:
                values_events, values_snapshots = data_base.prepare_vals_for_insert(key, value, cam_id, False)

                # Запись в Базу Данных
                del values_events["timestamp"]
                del values_events["date_time"]
                # Запись события через Backend API
                event_id = self._create_event(backend_api_base_url, **values_events)
                response = self._upload_snapshot(backend_api_base_url, event_id, values_snapshots["snapshot"])
                if isinstance(response, str) and "error" in response:
                    logger.info("API недоступна. Событие не записано в БД!")

    def create_events_and_send_snapshots(self,
                                         data_base,
                                         backend_api_base_url: str,
                                         cam_id: int):

        create_events_and_send_snapshots_thread = threading.Thread(target=self._create_events_and_send_snapshots,
                                                                   args=(data_base, backend_api_base_url, cam_id))
        create_events_and_send_snapshots_thread.daemon = True
        create_events_and_send_snapshots_thread.start()

    def _create_event(self, backend_api_base_url: str, cam_id: int, zone_id: int,
                      event: str, people_id: int, category_id: int):
        create_url = backend_api_base_url + "/video_events/api/v1/events/create"

        event_data = {
            "cam_slot": cam_id,
            "zone_id": zone_id,
            "event": event,
            "people_id": people_id,
            "category_id": category_id
        }

        if zone_id is None:
            event_data.pop("zone_id")
        if people_id is None:
            event_data.pop("people_id")

        try:
            response = requests.post(create_url, json=event_data)
            if response.status_code == 401:
                logger.info("Токен устарел. Обновляю ...")
                self.token = self._get_token(backend_api_base_url)
                response = requests.post(create_url, headers={"Authorization": f"Bearer {self.token}"}, json=event_data)
            
            return response.json() if response.status_code == 200 else {"error": f"API request failed", "status_code": response.status_code}
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}

    def _upload_snapshot(self, backend_api_base_url: str, event_id: int, file):
        headers = {
            'Authorization': f'Bearer {self.token}',
        }

        upload_snapshot_url = backend_api_base_url + f'/video_events/api/v1/events/upload_picture?event_id={event_id}'

        files = {'file': ("image.jpg", file, 'image/jpeg')}
        
        # Если запрос для создания события вернул ошибку API
        
        if isinstance(event_id, str) and "error" in event_id:
            return "error: Ошибка получения данных из API"
        
        try:
            # response = requests.post(upload_snapshot_url, headers=headers, files=files)
            response = requests.post(upload_snapshot_url, files=files)
            logger.info(response)
        except Exception:
            logger.info("API недоступна.")
            response = None

        if response is not None:
            # Вывод кода статуса события
            # Обновляем токен если он устарел
            if response.status_code == 401:
                logger.info("Токен устарел. Обновляю ...")
                self.token = self._get_token(backend_api_base_url)
                response = requests.post(upload_snapshot_url, headers=headers, files=files)
            if response.status_code == 200:
                logger.info(f"Запись данных в таблицу events выполнена успешно!")
                logger.info(f"Запись данных в таблицу snapshots выполнена успешно!")
            else:
                logger.info(response.content.decode('utf-8', errors='ignore'))

            # Попытка распарсить JSON
            try:
                return response.json()
            except ValueError:
                return {"error": "Invalid JSON response", "content": response.content.decode('utf-8', errors='ignore')}
        else:
            return "error: Ошибка получения данных из API"

    def read_zones_data(self, backend_api_base_url, cam_id):
        read_zones_url = backend_api_base_url + f"/video_events/api/v1/cameras/read_verbose?cam_slot={cam_id}"
        headers = {
            'Authorization': f'Bearer {self.token}',
        }
        try:
            # response = requests.get(read_zones_url, headers=headers)
            response = requests.get(read_zones_url)
            logger.info(response)
        except:
            logger.info("API недоступна.")
            response = None

        if response is not None:
            # Обновляем токен если он устарел
            if response.status_code == 401:
                logger.info("Токен устарел. Обновляю ...")
                self.token = self._get_token(backend_api_base_url)
                response = requests.get(read_zones_url, headers=headers)
            if response.status_code == 200:
                logger.info("Координаты опасных зон успешно получены!")
        else:
            return "error: Ошибка получения данных из API"

        response = response.json()

        if response is None:
            zones = None
        else:
            zones = response.get('zones', None)

        # Словарь в котором хранятся все зоны с координатами
        zones_dict = {}

        if zones is not None: 
            # Словарь в котором хранятся все зоны с координатами
            for zone in zones:
                zone_id, vertices = zone["zone_id"], zone["vertices"]
                if zone_id not in zones_dict:
                    zones_dict[zone_id] = []

                # Добавляем все координаты (точки) опасной зоны в словарь
                for v in vertices:
                    point = list(v.values())
                    zones_dict[zone_id].append(point)

        self.danger_zones_cords = zones_dict

    def read_camera_data(self, message, target_camera_slot):
        """Process a single camera update message"""
        try:
            if message is None or message.value is None:
                return False

            # Check message key
            key = message.key
            if not key:
                return False
                
            data = message.value
            logger.info(f"Processing message with key: {key}")

            # Handle different message types based on key
            if key == f"state_camera_slot_{target_camera_slot}":
                # Handle direct camera update
                return self._process_single_camera_update(data, target_camera_slot)
            elif key == "state_camera_slot_all":
                # Handle full state update
                return self._process_all_cameras_update(data, target_camera_slot)
            else:
                return False

        except Exception as e:
            logger.error(f"Error processing camera data: {str(e)}", exc_info=True)
            return False

    def _process_single_camera_update(self, data, target_camera_slot):
        """Process update for single camera"""
        try:
            camera_slot = data.get("camera_slot")
            if camera_slot != target_camera_slot:
                return False
                
            logger.info(f"Processing direct update for camera {camera_slot}")
            
            # Process zones data
            return self._extract_and_update_zones(data.get("zones", []), camera_slot)

        except Exception as e:
            logger.error(f"Error processing single camera update: {str(e)}")
            return False

    def _process_all_cameras_update(self, data, target_camera_slot):
        """Process update from full state message"""
        try:
            cameras = data.get("cameras", [])
            for camera in cameras:
                if camera.get("camera_slot") == target_camera_slot:
                    logger.info(f"Found target camera {target_camera_slot} in full state update")
                    return self._extract_and_update_zones(camera.get("zones", []), target_camera_slot)
            
            return False

        except Exception as e:
            logger.error(f"Error processing all cameras update: {str(e)}")
            return False

    def _extract_and_update_zones(self, zones, camera_slot):
        """Extract and update zones data"""
        try:
            if not zones:
                self.danger_zones_cords = {}
                logger.warning(f"No zones data for camera {camera_slot}")
                return False

            zones_dict = {}
            for zone in zones:
                zone_id = zone.get("zone_id")
                vertices = zone.get("vertices", [])
                if zone_id is None or not vertices:
                    continue

                if zone_id not in zones_dict:
                    zones_dict[zone_id] = []

                # Extract and scale coordinates
                for v in vertices:
                    if "x" not in v or "y" not in v:
                        continue
                    x = int(v['x'] * 1280 / 1920)
                    y = int(v['y'] * 720 / 1080)
                    zones_dict[zone_id].append([x, y])

            if zones_dict:
                with self._zones_lock:
                    self.danger_zones_cords = zones_dict
                    logger.info(f"Updated zones for camera {camera_slot}: {len(zones_dict)} zones")
                    logger.debug(f"Zones details: {zones_dict}")
                return True
            
            return False

        except Exception as e:
            logger.error(f"Error extracting zones: {str(e)}")
            return False

    def get_danger_zones(self):
        """Потокобезопасный метод для получения текущих опасных зон"""
        with self._zones_lock:
            return self.danger_zones_cords.copy() if self.danger_zones_cords else None
