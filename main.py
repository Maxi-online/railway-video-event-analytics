import logging
import queue
import threading
from collections import defaultdict, deque
from pathlib import Path
from time import time, sleep

import cv2
import hydra
import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig
from ultralytics import YOLO, YOLOWorld

from src.services.backend_api_client import BackendAPIClient
from src.services.database.database import DataBase
from src.services.inferencer.inferencer import Inferencer, InferencerWorld
from src.services.tracker.tracker import TrackerPJ, TrackerLWT
from src.utils.video_utils import calculate_fps
# === Live MJPEG preview ===
from threading import Thread
from src import preview_server
from src.utils.draw_utils import plot_results, plot_results_unet, plot_fps
from src.utils.load_object import load_object
from src.utils.tasks_logic import (
    task_1_3, task_1_2, task_3_1, task_3_3,
    task_4_2, task_5_2, task_6_1, task_7,
    task_6_2_and_8_1_1, task_9, task_10_1,
    task_10_2
)
from src.services.readers import KafkaFrameReader, CV2FrameReader
from src.services.danger_zones_reader import DangerZonesReader
from src import ThreadStopEventException

# Определение логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Отключаем INFO логи от kafka-python
logging.getLogger('kafka').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
            

@hydra.main(version_base=None, config_path="src/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Путь до папки с моделями
    root_path = Path(__file__).parent

    # Инициализация переменных которые могут вызвать ошибку программы если их не определить заранее
    person_ids_1_2 = None
    triggered_1_2_dict = None
    danger_zones_cords = None
    rail_cache = None
    results_R = None
    names_R = None
    model_PJ = None
    model_LWT = None
    tracker_PJ = None
    tracker_LWT = None
    model_R = None
    model_WORLD = None
    inferencer_R = None
    inferencer_WORLD = None
    deviant_history = None

    # Настройки preview window
    if cfg.show_video_preview:
        preview_width, preview_height = cfg.preview_screen_size.width, cfg.preview_screen_size.height
        # WINDOW_NORMAL позволяет изменять размер
        cv2.namedWindow(cfg.video_visual_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.video_visual_name, preview_width, preview_height)

        # Перемещаем окно в центр экрана
        screen_width = 1920
        screen_height = 1080
        pos_x = (screen_width - preview_width) // 2
        pos_y = (screen_height - preview_height) // 2
        cv2.moveWindow(cfg.video_visual_name, pos_x, pos_y)

    # =================================================#
    #              Запуск MJPEG-сервера                #
    # =================================================#
    Thread(target=preview_server.run, daemon=True).start()

    # =================================================#
    #            Загрузка моделей YOLOv8               #
    # =================================================#
    try:
        if cfg.pj_enabled:
            model_PJ = YOLO(root_path / cfg.model_pj)
            model_PJ.fuse()
        if cfg.lwt_enabled:
            model_LWT = YOLO(root_path / cfg.model_lwt)
            model_LWT.fuse()
        if cfg.task_9_enabled:
            model_WORLD = YOLOWorld(root_path / cfg.model_world)

        if cfg.r_enabled:
            if cfg.rails_unet:
                model_R = ort.InferenceSession(root_path / cfg.model_r_unet,
                                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            else:
                model_R = YOLO(root_path / cfg.model_r)
                model_R.fuse()
    except:
        raise ModuleNotFoundError("Не удалось загрузить одну из моделей!")

    # =================================================#
    #              Обработка видеофайла                #
    # =================================================#

    # Папка для хранения картинок для последующего вывода их на StreamAPI
    images_temp_path = root_path / "images_temp" / cfg.topic_name
    images_temp_path.mkdir(parents=True, exist_ok=True)

    # Очередь для хранения кадров
    frame_queue = queue.Queue(maxsize=cfg.c)
    stop_event = threading.Event()
    skip_interval = 1

    # Шрифт
    font = load_object(cfg.font)
    # ID камеры
    cam_id = cfg.cam_id
    
    # Запуск потока для чтения кадров из Broker
    if cfg.kafka_enabled:
        reader = KafkaFrameReader(
            cfg.kafka_address,
            cfg.topic_name,
            cam_id,
            frame_queue,
            stop_event,
            skip_interval,
        )
    else:
        reader = CV2FrameReader(
            cfg.local_video_path,
            frame_queue,
            stop_event,
            skip_interval,
        )
    reader.start()

    # Инициализация объекта База Данных
    data_base = DataBase(cfg.token, cfg.login, cfg.password, cfg.triggers_count,
                         cfg.triggers_count_task_4_2, cfg.triggers_count_task_5_2,
                         cfg.db_check_time, cfg.task_1_2_enabled, cfg.task_1_3_enabled,
                         cfg.task_3_1_enabled, cfg.task_3_3_enabled, cfg.task_4_2_enabled,
                         cfg.task_5_2_enabled, cfg.task_6_1_enabled, cfg.task_6_2_enabled,
                         cfg.task_7_enabled, cfg.task_8_1_1_enabled, cfg.task_9_enabled,
                         cfg.task_10_1_enabled, cfg.task_10_2_enabled
                         )

    # Инициализация объекта Backend API
    api_client = BackendAPIClient(cfg.token, cfg.login, cfg.password, danger_zones_cords)

    # Запуск потока для чтения координат опасных зон отдельным потоком
    danger_zones_reader = None
    if cfg.kafka_enabled:
        danger_zones_reader = DangerZonesReader(
            cfg.kafka_address,
            cfg.topic_name,
            api_client,
            data_base,
            cam_id,
            stop_event,
            cfg.check_danger_zones_cords_delay,
        )
        danger_zones_reader.start()

    # Объекты: Трекер, Инференсер
    if cfg.pj_enabled:
        tracker_PJ = TrackerPJ(model_PJ)
    if cfg.lwt_enabled:
        tracker_LWT = TrackerLWT(model_LWT)

    # Установка времени отслеживания (после пропавший трек удаляется)
    if cfg.pj_enabled:
        tracker_PJ.set_track_timeout(cfg.track_timeout)
    if cfg.lwt_enabled:
        tracker_LWT.set_track_timeout(cfg.track_timeout)

    if cfg.r_enabled:
        inferencer_R = Inferencer(model_R)
    if cfg.task_9_enabled:
        inferencer_WORLD = InferencerWorld(model_WORLD, cfg.text_prompts)

    if cfg.task_10_2_enabled:
        deviant_history = defaultdict(lambda: [])

    # Переменные для кэширования результатов сегментации рельсов
    rail_cache_frame_count = 0

    # Время для отсчета интервала
    timer_time = 0

    # Переменные для расчета FPS и пропуска кадров
    processed_frames = 0
    prev_frame_time = time()
    target_fps = cfg.target_fps
    frame_history = deque(maxlen=30)  # Хранит историю за последние 30 кадров
    last_skip_update_time = prev_frame_time
    last_log_time = time()

    # ======================================================#
    #            Чтение и пропуск кадров                    #
    # ======================================================#
    frame_count = 0
    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # ======================================================#
            #                    Инференс моделей                   #
            # ======================================================#
            # Делаем копию изображения
            prev_frame_time = time()
            annotated_frame = np.copy(frame)
            # Сегментация rails
            if cfg.r_enabled:
                if cfg.rails_cache:
                    # Обновление кэша сегментации рельсов
                    if rail_cache is None or rail_cache_frame_count >= cfg.rail_cache_update_interval:
                        results_R, names_R = inferencer_R.predict_rails(frame, cfg.rails_unet, cfg.r_unet_params,
                                                                        cfg.rail_model_params)
                        rail_cache = (results_R, names_R)
                        rail_cache_frame_count = 0
                    else:
                        results_R, names_R = rail_cache
                        rail_cache_frame_count += 1
                else:
                    results_R, names_R = inferencer_R.predict_rails(frame, cfg.rails_unet, cfg.r_unet_params,
                                                    cfg.rail_model_params)

            # ============================================#
            #                    Трекинг                  #
            # ============================================#
            # Люди + жилеты
            if cfg.pj_enabled:
                annotated_frame = tracker_PJ.predict_on_image_track(
                    annotated_frame, **cfg.pj_model_params, **cfg.tracking_params,
                    **cfg.velocity_params, font=font)

            # Вагоны
            if cfg.lwt_enabled:
                annotated_frame = tracker_LWT.predict_on_image_track(
                    annotated_frame, **cfg.lwt_model_params, **cfg.tracking_params,
                    velocity_correction=cfg.velocity_correction_params,
                    **cfg.velocity_params, font=font)

            # =================================================================#
            #                      Визуализация результатов                    #
            # =================================================================#
            # locomotive + wagon + tank
            if cfg.lwt_enabled:
                if cfg.show_lwt:
                    annotated_frame = plot_results(
                        tracker_LWT.results, tracker_LWT.class_names,
                        annotated_frame,'LWT', font, is_tracking=True,**cfg.visualize_inference_params,
                    )
            # rails
            if cfg.r_enabled:
                if cfg.show_rails:
                    if cfg.rails_unet:
                        annotated_frame = plot_results_unet(results_R, names_R, annotated_frame,
                                                            **cfg.visualize_r_unet_params, font=font)
                    else:
                        annotated_frame = plot_results(
                            results_R, names_R,
                            annotated_frame, 'R', font, is_tracking=False, **cfg.visualize_inference_params,
                        )

            # ======================================================#
            #                      Бизнес-задачи                    #
            # ======================================================#
            # Бизнес Задача 1.2 Определение сотрудника в опасной зоне
            if cfg.task_1_2_enabled and cfg.db_enabled and api_client.danger_zones_cords is not None:
                person_ids_1_2 = {}
                triggered_1_2_dict = {}

                for zone_id, zone_info in api_client.danger_zones_cords.items():
                    annotated_frame = task_1_2(annotated_frame, tracker_PJ.results_persons,
                                               data_base, zone_id, font, zone_info, **cfg.task_1_2_params)

                    # # Запись параметра для id человека с id опасной зоны для задачи 7_1
                    person_ids_1_2[zone_id] = data_base.people_ids[f"1_2_{zone_id}"]
                    triggered_1_2_dict[zone_id] = data_base.triggered_1_2

                    # Обновление значений в task_logic_trigger_count, task_write_to_db
                    data_base.process_1_2(annotated_frame, task_number=f"1_2_{zone_id}")
            # Бизнес задача 1.3 Определение сотрудника без СИЗ
            if cfg.pj_enabled:
                if cfg.task_1_3_enabled:
                    annotated_frame = task_1_3(annotated_frame, tracker_PJ,
                                               data_base, font, **cfg.task_1_3_params)
                elif cfg.show_pj:
                    # Простое отображение результатов детекции человека и жилетов
                    annotated_frame = plot_results(
                        tracker_PJ.results_persons, tracker_PJ.class_names,
                        annotated_frame, 'PJ', font, is_tracking=True, **cfg.visualize_inference_params,
                    )
                    annotated_frame = plot_results(
                        tracker_PJ.results_jackets, tracker_PJ.class_names,
                        annotated_frame, 'PJ', font, is_tracking=True, **cfg.visualize_inference_params,
                    )

            # Бизнес задача 3.1 Спрыгивание с подвижного состава, платформы
            if cfg.task_3_1_enabled:
                annotated_frame = task_3_1(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    data_base,
                    font,
                    reader.skip_interval,
                    cfg.move_velocity_lwt,
                    **cfg.task_3_1_params,
                )

            # Бизнес задача 3.3 Проезд сотрудника на подножке, наружной лестнице цистерны, совмещенной с подножкой
            if cfg.task_3_3_enabled:
                annotated_frame = task_3_3(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    data_base,
                    font,
                    cfg.move_velocity_lwt,
                    cfg.move_velocity_p,
                    **cfg.task_3_3_params,
                )

            # Бизнес задача 4.2 Переход через железнодорожный путь перед движущимся подвижным составом
            if cfg.task_4_2_enabled:
                annotated_frame = task_4_2(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    results_R,
                    data_base,
                    font, cfg.move_velocity_lwt, **cfg.task_4_2_params,
                )

            # Бизнес задача 5.2 Переход пути вблизи стоящего состава на расстоянии менее 5 метров с головы или хвоста состава
            if cfg.task_5_2_enabled:
                annotated_frame = task_5_2(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    results_R,
                    data_base,
                    font, **cfg.task_5_2_params,
                )

            # Бизнес задача 6.1 Пролезание работников под стоящими вагонами
            if cfg.task_6_1_enabled:
                annotated_frame = task_6_1(
                        annotated_frame,
                        tracker_PJ,
                        tracker_LWT,
                        data_base,
                        reader.skip_interval,
                        font,
                        **cfg.task_6_1_params,
                    )

            # Бизнес задача 6.2 Залезание работников на автосцепки или под них
            # Бизнес задача 8.1.1 Определение работников, сидящих в зоне наблюдения
            if cfg.task_6_2_enabled or cfg.task_8_1_1_enabled:
                annotated_frame = task_6_2_and_8_1_1(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    data_base,
                    reader.skip_interval,
                    **cfg.task_6_2_and_8_1_1_params,
                    font=font,
                )

            # Бизнес задача 7:
            # 7.1 Идентификация пересечений ЖД-путей в запрещенных зонах
            # 7.2.1 Идентификация пересечений ЖД-путей по опасной траектории
            # 7.2.2 Идентификация длительного нахождения на ЖД-путях без движения
            if cfg.task_7_enabled:
                annotated_frame = task_7(
                    annotated_frame,
                    tracker_PJ,
                    results_R,
                    data_base,
                    reader.skip_interval,
                    cfg.db_enabled,
                    cfg.task_1_2_enabled,
                    cfg.task_7_1_enabled,
                    font, cfg.move_velocity_p,
                    triggered_1_2_dict, person_ids_1_2,
                    **cfg.task_7_params)

            # Бизнес задача 9 Обнаружение посторонних предметов
            if cfg.task_9_enabled:
                if time() - timer_time >= cfg.world_time_interval:
                    # Обновление времени таймера
                    timer_time = time()
                    annotated_frame = task_9(annotated_frame, inferencer_WORLD, cfg.world_model_params, data_base,
                                             cfg.task_9_show, cfg.visualize_inference_params, font)

            # Бизнес задача 10.1 Определение нехарактерного положения
            if cfg.task_10_1_enabled:
                annotated_frame = task_10_1(
                    annotated_frame,
                    tracker_PJ,
                    data_base,
                    **cfg.task_10_1_params,
                    font=font)

            # Бизнес задача 10.2 Определение нехарактерного поведения
            if cfg.task_10_2_enabled:
                annotated_frame = task_10_2(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    data_base,
                    deviant_history,
                    reader.skip_interval,
                    cfg.move_velocity_p,
                    **cfg.task_10_2_params,
                    font=font)

            # Передаём кадр в MJPEG-сервер
            preview_server.latest_frame = annotated_frame

            # Расчет и отрисовка FPS
            new_frame_time, frame_fps, avg_fps = calculate_fps(
                reader.total_read_frames, prev_frame_time,  frame_history
            )
            if cfg.show_fps:
                plot_fps(annotated_frame, frame_fps, avg_fps, cfg.color_fps, font)

            # Динамический пропуск кадров на основе frame_fps
            if processed_frames > 5 and frame_fps > 0:
                if frame_fps >= target_fps:
                    reader.skip_interval = 1  # Не пропускаем кадры, если frame_fps уже достаточно высокий
                else:
                    reader.skip_interval  = max(1, round(target_fps / frame_fps))
            else:
                reader.skip_interval  = 1  # Защита от деления на ноль

            # Дополнительная корректировка на основе avg_fps (выполняется реже)
            if new_frame_time - last_skip_update_time >= 1:  # Проверяем каждую секунду
                if avg_fps > 0:
                    if (frame_fps < target_fps * 0.9) and (avg_fps < target_fps * 0.9):  # Если avg_fps слишком низкий
                        reader.skip_interval = min(reader.skip_interval + 1, 10)  # Увеличиваем интервал пропуска, но не более 10
                    elif avg_fps > target_fps * 1.1:  # Если avg_fps слишком высокий
                        reader.skip_interval = max(1, reader.skip_interval - 1)  # Уменьшаем интервал пропуска, но не меньше 1
                last_skip_update_time = new_frame_time
            
            # Корректировка времени отслеживания (после пропавший трек удаляется)
            if cfg.pj_enabled:
                tracker_PJ.set_track_timeout(int(cfg.track_timeout / reader.skip_interval))
            if cfg.lwt_enabled:
                tracker_LWT.set_track_timeout(int(cfg.track_timeout / reader.skip_interval))

            processed_frames += 1

            # Логирование каждые 5 секунд
            if new_frame_time - last_log_time >= 5:
                logger.info(f"Total frames read cam_id {cam_id}: {reader.total_read_frames}, "
                            f"Processed frames: {processed_frames}, "
                            f"Queue size: {frame_queue.qsize()}, "
                            f"Skip interval: {reader.skip_interval}")

                last_log_time = new_frame_time

            # Отображение кадра
            if cfg.show_video_preview:
                cv2.imshow(cfg.video_visual_name, annotated_frame)
                key = cv2.waitKey(1)
                # Выход по ESC или 'q'
                if key == 27 or key == ord('q'):
                    break
                sleep(0.01)

                # Запись картинки в папку ./images_temp
                if cfg.save_to_images_temp:
                    # Сжатие картинки
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
                    _, encimg = cv2.imencode('.jpg', annotated_frame, encode_param)

                    # Сохранение кадра в папку ./images_temp
                    file_path = Path(images_temp_path) / f"frame{frame_count + 1}.jpg"
                    with file_path.open('wb') as f:
                        f.write(encimg.tobytes())
                frame_count += 1

            # ======================================================================================#
            #                      Логика обработки бизнес-задач для записи в БД                    #
            # ======================================================================================#
            if cfg.db_enabled:
                data_base.process_business_tasks(annotated_frame)

                # Проверка, есть ли результаты для записи в БД (раз в check_db_time)
                if time() - data_base.start_time >= data_base.check_db_time:
                    # Засекаем новый промежуток старта видео
                    data_base.start_time = time()

                    # Запись всех событий в Backend API
                    api_client.create_events_and_send_snapshots(data_base, cfg.backend_api_base_url, cam_id)


        # Если поток закончился, перейти в режим ожидания
        # print("Поток завершился. Ожидание нового видеопотока.")
        sleep(1)  # Пауза перед повторной проверкой сообщений

    # Остановка основного потока если поток чтения координат опасных зон
    # и поток чтения видео закончились
    except ThreadStopEventException:
        logger.info("Остановка основного потока т.к. поток чтения координат опасных зон и потока чтения видео остановлены.")
        raise Exception()
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания с клавиатуры. Остановка программы.")
    finally:
        # Лог для тестирования работы потоков
        logger.info(f"Кол-во активных потоков: {threading.active_count()}")
        # Очистка ресурсов
        cv2.destroyAllWindows()

        # Ожидание окончания работы потоков
        stop_event.set()
        reader.join()
        if danger_zones_reader is not None:
            danger_zones_reader.join()


if __name__ == "__main__":
    main()
