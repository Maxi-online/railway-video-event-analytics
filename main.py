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

# Definition of logic
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Disable INFO logs from kafka-python
logging.getLogger('kafka').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
            

@hydra.main(version_base=None, config_path="src/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # The Path to the Model Folder
    root_path = Path(__file__).parent

    # Initializing variables that can cause a program error if they are not identified in advance
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

    # Preview Windows settings
    if cfg.show_video_preview:
        preview_width, preview_height = cfg.preview_screen_size.width, cfg.preview_screen_size.height
        # WINDOW_NORMAL allows for dimensional changes
        cv2.namedWindow(cfg.video_visual_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.video_visual_name, preview_width, preview_height)

        # Move the window to the center of the screen
        screen_width = 1920
        screen_height = 1080
        pos_x = (screen_width - preview_width) // 2
        pos_y = (screen_height - preview_height) // 2
        cv2.moveWindow(cfg.video_visual_name, pos_x, pos_y)

    # =================================================#
    # Start MJPEG server #
    # =================================================#
    Thread(target=preview_server.run, daemon=True).start()

    # =================================================#
    # Uploading of YOLov8 models #
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
        raise ModuleNotFoundError("Couldn't load one of the models!")

    # =================================================#
    # Video file processing #
    # =================================================#

    # File to store images for later release on StreetAPI
    images_temp_path = root_path / "images_temp" / cfg.topic_name
    images_temp_path.mkdir(parents=True, exist_ok=True)

    # Storage line
    frame_queue = queue.Queue(maxsize=cfg.c)
    stop_event = threading.Event()
    skip_interval = 1

    # Font
    font = load_object(cfg.font)
    # ID Cameras
    cam_id = cfg.cam_id
    
    # Starting a stream to read from Broker
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

    # Initialization of the database object
    data_base = DataBase(cfg.token, cfg.login, cfg.password, cfg.triggers_count,
                         cfg.triggers_count_task_4_2, cfg.triggers_count_task_5_2,
                         cfg.db_check_time, cfg.task_1_2_enabled, cfg.task_1_3_enabled,
                         cfg.task_3_1_enabled, cfg.task_3_3_enabled, cfg.task_4_2_enabled,
                         cfg.task_5_2_enabled, cfg.task_6_1_enabled, cfg.task_6_2_enabled,
                         cfg.task_7_enabled, cfg.task_8_1_1_enabled, cfg.task_9_enabled,
                         cfg.task_10_1_enabled, cfg.task_10_2_enabled
                         )

    # Initialization of the Backend API
    api_client = BackendAPIClient(cfg.token, cfg.login, cfg.password, danger_zones_cords)

    # Starting the flux to read the coordinates of hazardous areas by separate stream
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

    # Subjects: Tracker, Inferencer
    if cfg.pj_enabled:
        tracker_PJ = TrackerPJ(model_PJ)
    if cfg.lwt_enabled:
        tracker_LWT = TrackerLWT(model_LWT)

    # Installation of tracking time (after the missing track is removed))
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

    # Changed to cache rail segmentation results
    rail_cache_frame_count = 0

    # Time for interval reference
    timer_time = 0

    # Changed to calculate FPS and pass
    processed_frames = 0
    prev_frame_time = time()
    target_fps = cfg.target_fps
    frame_history = deque(maxlen=30)  # Keeps history from the last 30 shots.
    last_skip_update_time = prev_frame_time
    last_log_time = time()

    # ======================================================#
    # Reading and missing personnel #
    # ======================================================#
    frame_count = 0
    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # ======================================================#
            # Model inference #
            # ======================================================#
            # Make a copy of the image.
            prev_frame_time = time()
            annotated_frame = np.copy(frame)
            # Rails segmentation
            if cfg.r_enabled:
                if cfg.rails_cache:
                    # Updating the rail segmentation cache
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
            # Treking                  #
            # ============================================#
            # People + vests
            if cfg.pj_enabled:
                annotated_frame = tracker_PJ.predict_on_image_track(
                    annotated_frame, **cfg.pj_model_params, **cfg.tracking_params,
                    **cfg.velocity_params, font=font)

            # Wagons
            if cfg.lwt_enabled:
                annotated_frame = tracker_LWT.predict_on_image_track(
                    annotated_frame, **cfg.lwt_model_params, **cfg.tracking_params,
                    velocity_correction=cfg.velocity_correction_params,
                    **cfg.velocity_params, font=font)

            # =================================================================#
            # Visualization of results #
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
            # Business agenda #
            # ======================================================#
            # Business Task 1.2 Identification of a staff member in a hazardous area
            if cfg.task_1_2_enabled and cfg.db_enabled and api_client.danger_zones_cords is not None:
                person_ids_1_2 = {}
                triggered_1_2_dict = {}

                for zone_id, zone_info in api_client.danger_zones_cords.items():
                    annotated_frame = task_1_2(annotated_frame, tracker_PJ.results_persons,
                                               data_base, zone_id, font, zone_info, **cfg.task_1_2_params)

                    # # Record option for id person from id dangerous area for task 7_1
                    person_ids_1_2[zone_id] = data_base.people_ids[f"1_2_{zone_id}"]
                    triggered_1_2_dict[zone_id] = data_base.triggered_1_2

                    # Update of task_logic_trigger_count, task_write_to_db
                    data_base.process_1_2(annotated_frame, task_number=f"1_2_{zone_id}")
            # Business task 1.3 Identification of staff member without PPE
            if cfg.pj_enabled:
                if cfg.task_1_3_enabled:
                    annotated_frame = task_1_3(annotated_frame, tracker_PJ,
                                               data_base, font, **cfg.task_1_3_params)
                elif cfg.show_pj:
                    # A simple display of the results of a human and vest detective
                    annotated_frame = plot_results(
                        tracker_PJ.results_persons, tracker_PJ.class_names,
                        annotated_frame, 'PJ', font, is_tracking=True, **cfg.visualize_inference_params,
                    )
                    annotated_frame = plot_results(
                        tracker_PJ.results_jackets, tracker_PJ.class_names,
                        annotated_frame, 'PJ', font, is_tracking=True, **cfg.visualize_inference_params,
                    )

            # Business task 3.1 jumping off rolling stock, platform
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

            # Business task 3.3 Travel of staff member on foot, outside staircase of tank reciprocally incorporated with step̆
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

            # Business task 4.2 Transition to moving rolling rolling stock
            if cfg.task_4_2_enabled:
                annotated_frame = task_4_2(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    results_R,
                    data_base,
                    font, cfg.move_velocity_lwt, **cfg.task_4_2_params,
                )

            # Business task 5.2 Passing the track near a standing convoy less than 5 metres from the head or tail of the convoy
            if cfg.task_5_2_enabled:
                annotated_frame = task_5_2(
                    annotated_frame,
                    tracker_PJ,
                    tracker_LWT,
                    results_R,
                    data_base,
                    font, **cfg.task_5_2_params,
                )

            # Business task 6.1 Slapping workers under standing wagons
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

            # Business task 6.2 Intruding or Undercoming Workers
            # Business task 8.1.1 Identification of employees sitting in the observation zone
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

            # Business challenge 7:
            # 7.1 Identification of intersections of W-tracks in prohibited areas
            # 7.2.1 Identification of W-track intersections along a dangerous trajectory
            # 7.2.2 Identification of long distance on the W-way without movement
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

            # Business objective 9 Detection of foreign items
            if cfg.task_9_enabled:
                if time() - timer_time >= cfg.world_time_interval:
                    # Update timer time
                    timer_time = time()
                    annotated_frame = task_9(annotated_frame, inferencer_WORLD, cfg.world_model_params, data_base,
                                             cfg.task_9_show, cfg.visualize_inference_params, font)

            # Business task 10.1 Definition of non-maintenance
            if cfg.task_10_1_enabled:
                annotated_frame = task_10_1(
                    annotated_frame,
                    tracker_PJ,
                    data_base,
                    **cfg.task_10_1_params,
                    font=font)

            # Business task 10.2 Definition of non-maintenance behaviour
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

            # We're gonna transfer the shot to the MJPEG server.
            preview_server.latest_frame = annotated_frame

            # FPS calculation and drawing
            new_frame_time, frame_fps, avg_fps = calculate_fps(
                reader.total_read_frames, prev_frame_time,  frame_history
            )
            if cfg.show_fps:
                plot_fps(annotated_frame, frame_fps, avg_fps, cfg.color_fps, font)

            # Dynamic frame_fps personnel pass
            if processed_frames > 5 and frame_fps > 0:
                if frame_fps >= target_fps:
                    reader.skip_interval = 1  # Do not miss the footage if frame_fps is already high enough
                else:
                    reader.skip_interval  = max(1, round(target_fps / frame_fps))
            else:
                reader.skip_interval  = 1  # Protection against division by zero

            # Additional adjustment based on avg_fps (implemented less frequently))
            if new_frame_time - last_skip_update_time >= 1:  # Check every second.
                if avg_fps > 0:
                    if (frame_fps < target_fps * 0.9) and (avg_fps < target_fps * 0.9):  # If avg_fps is too low
                        reader.skip_interval = min(reader.skip_interval + 1, 10)  # Increase the pass interval, but not more than 10
                    elif avg_fps > target_fps * 1.1:  # If avg_fps is too high
                        reader.skip_interval = max(1, reader.skip_interval - 1)  # Reduce the pass interval, but not less than 1
                last_skip_update_time = new_frame_time
            
            # Adjustment of tracking time (after the missing track is removed))
            if cfg.pj_enabled:
                tracker_PJ.set_track_timeout(int(cfg.track_timeout / reader.skip_interval))
            if cfg.lwt_enabled:
                tracker_LWT.set_track_timeout(int(cfg.track_timeout / reader.skip_interval))

            processed_frames += 1

            # Logging every 5 seconds
            if new_frame_time - last_log_time >= 5:
                logger.info(f"Total frames read cam_id {cam_id}: {reader.total_read_frames}, "
                            f"Processed frames: {processed_frames}, "
                            f"Queue size: {frame_queue.qsize()}, "
                            f"Skip interval: {reader.skip_interval}")

                last_log_time = new_frame_time

            # Image of frame
            if cfg.show_video_preview:
                cv2.imshow(cfg.video_visual_name, annotated_frame)
                key = cv2.waitKey(1)
                # ESC or 'q' exit
                if key == 27 or key == ord('q'):
                    break
                sleep(0.01)

                # Record image in folder ./images_temp
                if cfg.save_to_images_temp:
                    # Compressing the image
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
                    _, encimg = cv2.imencode('.jpg', annotated_frame, encode_param)

                    # Save frame in folder ./images_temp
                    file_path = Path(images_temp_path) / f"frame{frame_count + 1}.jpg"
                    with file_path.open('wb') as f:
                        f.write(encimg.tobytes())
                frame_count += 1

            # ======================================================================================#
            # Logic of Business Tasks to Record in database                    #
            # ======================================================================================#
            if cfg.db_enabled:
                data_base.process_business_tasks(annotated_frame)

                # Check if results are available for recording in the database ( once in check_db_time))
                if time() - data_base.start_time >= data_base.check_db_time:
                    # We're picking up a new period of video launch.
                    data_base.start_time = time()

                    # Recording all events in Backend API
                    api_client.create_events_and_send_snapshots(data_base, cfg.backend_api_base_url, cam_id)


        # If the flow is over, go to waiting mode.
        # print("The flows over. Waiting for a new video stream.")
        sleep(1)  # Pause before rechecking messages

    # Stopping the main stream when reading the coordinates of the hazardous areas
    # and the video reading stream is over.
    except ThreadStopEventException:
        logger.info('Stopping the main stream because the reading of the coordinates of the hazardous areas and the reading of the video are stopped.')
        raise Exception()
    except KeyboardInterrupt:
        logger.info('I got a break from the keyboard. Programme freeze.')
    finally:
        # A log to test flow performance
        logger.info(f"Number of active flows: {threading.active_count()}")
        # Cleaning of resources
        cv2.destroyAllWindows()

        # Waiting for the end of the flow
        stop_event.set()
        reader.join()
        if danger_zones_reader is not None:
            danger_zones_reader.join()


if __name__ == "__main__":
    main()
