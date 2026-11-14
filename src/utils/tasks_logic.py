from src.utils.zone_events import feet_in_zone
import logging
from collections import deque
from datetime import datetime
from time import time
from typing import Tuple, Dict

import cv2
import numpy as np
from src.services.database.database import DataBase
from src.services.inferencer.inferencer import InferencerWorld
from src.services.tracker.tracker import TrackerPJ, TrackerLWT
from numpy.linalg import norm

from src.utils.draw_utils import (
    plot_bbox, plot_contour,
    plot_results, plot_danger_zone,
    plot_line_from_point, plot_approximated_rail_line
)
from src.utils.geometry_utils import (
    euclidean_distance, intersection_over_union, intersection_over_area_mask,
    intersection_rect_area, rect_area, intersection_polygon_rect_area,
    intersection_over_area_bbox, filter_contour_points,
)
from src.utils.tasks_funcs import (
    get_wagons_rails, approximate_rail_line,
    get_moving_wagons_rails, get_persons_feet,
    is_person_crossing_rails,
)

# Defining the logic
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================#
# Task_1_3 Identification of staff member without PPE. #
# ==============================================================#
def task_1_3(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    database: DataBase,
    font: int,
    threshold_pj: float,
    color_ok: Tuple[int, int, int],
    color_no: Tuple[int, int, int],
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """The function solves the task of 1.3 - Identification of a staff member without PPE.

    Args:
        img (np.ndarray): Image with warnings.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        database (DataBase): Database for the storage of logic response information.
        font (int): Text font.
        threshold_pj (float): Cut-off threshold to determine what a human vest is wearing, IoU.
        color_ok (Tuple[int, int, int]): Colour for BGR-style marking of people in vests.
        color_no (Tuple[int, int, int]): Colour for BGR-style marking of people without vests.
        fontscale (float): Text size.
        thickness (int): Thickness of bbox lines and text.

    Returns:
        img (np.ndarray): A frame showing the person in the vest and without.
    """
    # Logistical activation
    database.triggered_1_3 = False

    # The ID of the person who's gonna work the logic.
    persons_ids = tracker_PJ.results_persons["track_ids"]

    # Visualization of people and vests according to logic
    for bbox_p, id in zip(tracker_PJ.results_persons['boxes_xyxy'], persons_ids):
        IoU = float('-inf')  # Initialization of the variable for maximum IoU storage
        # Search for maximum IoU value between bboxes of humans and vests
        for bbox_j in tracker_PJ.results_jackets['boxes_xyxy']:
            IoU = max(IoU, intersection_over_union(bbox_p, bbox_j))

        # Check if the vest is on a person based on the IoU threshold_pj threshold
        if IoU > threshold_pj:
            # Visualization when wearing a vest
            img = plot_bbox(img, bbox_p, color_ok, thickness)
            cv2.putText(img, 'In the vest.', (int(bbox_p[0]), int(bbox_p[1])), font, fontscale, color_ok, 2)
        else:
            # Visualization when the vest isn't wearing
            img = plot_bbox(img, bbox_p, color_no, thickness)
            cv2.putText(img, 'No vest.', (int(bbox_p[0]), int(bbox_p[1])), font, fontscale, color_no, 2)
            # Pointing out that logic worked.
            database.triggered_1_3 = True
            database.triggered_time_1_3 = datetime.now()
            database.people_ids["1_3"] = int(id)

        # cv2.putText(img, f"ID: {id}", (int(bbox_p[0]), int(bbox_p[1] - 30)), font, fontscale, (240, 211, 105), 2)

    return img


# ====================================================================#
# Task_1_2 Identification of a staff member in a hazardous area             #
# ====================================================================#
def task_1_2(
    img: np.ndarray,
    results_p: dict,
    database: DataBase,
    zone_id: int,
    font: int,
    danger_zone_cords: list,
    danger_zone_thrsd: float,
    feet_ratio: float,
    color_danger: Tuple[int, int, int],
    color_safe: Tuple[int, int, int],
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """
    The function solves the task of 1.2 - Identification of a staff member in a hazardous area.

    Args:
        img (np.ndarray): Image with warnings.
        results_p (dict): Model prediction results.
        database (DataBase): Database for the storage of logic response information.
        zone_id (int): ID zone.
        font (int): Text font (cv2.FONT_HERSHEY_*).
        danger_zone_cords (list): Coordinates of the dangerous area. There may be either a list of the coordinates of the rectangle or a range of coordinates of the area.
        danger_zone_thrsd (float): Cut-off threshold for a person in a dangerous area. If the proportion of bbox legs crossing the hazardous zone exceeds this threshold, the person is considered to be in the hazardous zone.
        feet_ratio (float): Human leg position threshold.
        color_danger (Tuple[int, int, int]): Colour for marking of a dangerous area in BGR format.
        color_safe (Tuple[int, int, int]): Colour for marking the safe area in BGR format.
        fontscale (float): Text size.
        thickness (int): Thickness of bbox lines and text.

    Returns:
        img (np.ndarray): A frame showing the person in the vest and without.
    """
    if not isinstance(danger_zone_cords, (list, dict, np.ndarray)) or len(danger_zone_cords) < 3:
        logger.warning(f"Invalid danger zone coordinates for zone {zone_id}")
        return img
    # Logistical activation
    database.triggered_1_2 = False

    # Coordinates of the dangerous area
    danger_zone_cords = np.array(danger_zone_cords)
    # # We find all the Bboxes people's feet
    persons_feet = get_persons_feet(results_p, feet_ratio)

    # The ID of the person who's gonna work the logic.
    persons_ids = results_p["track_ids"]

    # We're going through all the people's legs and we're checking that they're not all in the restricted area
    count = 0
    for bbox, id in zip(persons_feet, persons_ids):
        if feet_in_zone(bbox, danger_zone_cords, danger_zone_thrsd):
            count += 1
            database.people_ids[f"1_2_{zone_id}"] = int(id)

    # Visualization
    if count != 0:
        img = plot_danger_zone(img, danger_zone_cords, color_danger, thickness)
        if danger_zone_cords.ndim == 1:  # If a dangerous area is a rectangle
            cv2.putText(img, 'A man in a dangerous zone.', (danger_zone_cords[0], danger_zone_cords[1]), font, fontscale,
                        color_danger, thickness)
        else:  # If hazardous zone - landfills
            cv2.putText(img, 'A man in a dangerous zone.', (danger_zone_cords[:, 0].max(), danger_zone_cords[:, 1].min()),
                        font, fontscale, color_danger, thickness)
        # Pointing out that logic worked.
        database.triggered_1_2 = True
        database.triggered_time_1_2 = datetime.now()
    else:
        img = plot_danger_zone(img, danger_zone_cords, color_safe, thickness)
        if danger_zone_cords.ndim == 1:  # If a dangerous area is a rectangle
            cv2.putText(img, 'Zone clear.', (danger_zone_cords[0], danger_zone_cords[1]), font, fontscale, color_safe,
                        thickness)
        else:  # If hazardous zone - landfills
            cv2.putText(img, 'Zone clear.', (danger_zone_cords[:, 0].max(), danger_zone_cords[:, 1].min()), font,
                        fontscale, color_safe, thickness)

    return img


# =================================================================================================================#
# Task_3_1: Jumping from rolling stock, platforms                                  #
# =================================================================================================================#
def task_3_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    font: int,
    skip_interval: int,
    move_velocity_lwt: float,
    feet_wagon_intersections_history_len: int,
    feet_ratio: float,
    threshold_ioa: float,
    show_task: bool,
    alert_text: str,
    color_alert: Tuple[int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
) -> np.ndarray:
    """
    Function to determine whether a staff member jumped from a moving wagon.

    Args:
        img (np.ndarray): Reference image.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        traker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        database (Database): A copy of the Database &apos; s Logistical Information Storage Class.
        font (int): Font.
        skip_interval (int): Staff pass interval.
        move_velocity_lwt (float): Threshold value of the speed at which the wagon starts to move.
        feet_wagon_intersections_history_len (int): Length of the history of stored human intersections at id with a moving wagon.
        feet_ratio (float): Human leg position threshold.
        threshold_ioa (float): IoA threshold value when logic activates.
        show_task (bool): Show warning message or not.
        alert_text (str): Warning message text.
        color_alert (Tuple[int]): Logic warning message color.
        txt_up_alert (int): How high the warning message distances from the upper boundary of the object frame in pixels.
        fontscale_alert (float): The size of the warning message text.
        thick_alert (int): Thickness of the warning message text.

    Returns:
        img (np.ndarray): Image with warning text.
    """
    # Logistical activation
    database.triggered_3_1 = False

    # Getting the results of people and wagons tracking
    results_tracking_p = tracker_PJ.results_persons
    # Getting people's feet coordinates.
    persons_feet = get_persons_feet(tracker_PJ.results_persons, feet_ratio)

    results_tracking_lwt = traker_LWT.results
    track_history_LWT = traker_LWT.track_history
    class_names_LWT = traker_LWT.class_names

    # Overtaken people's track results
    for box_feet_p, box_xyxyx_p, track_id_p in zip(persons_feet, results_tracking_p['boxes_xyxy'],
                                                   results_tracking_p['track_ids']):
        # Review of train track results
        for box_lwt, _, mask_lwt, cls_lwt, _, track_id_lwt in zip(*results_tracking_lwt.values()):
            # Definition of the key for finding the history of human intersections with a moving wagon
            track_key_p_feet_wagon_intersection = f"person{track_id_p}_{track_id_lwt}"
            # The current speed of the wagon for the track
            track_key_lwt_vel_ema = f"{class_names_LWT[cls_lwt]}_{track_id_lwt}_ema"
            # Getting the train speed list from the tracking history
            track_lwt_ema_list = list(track_history_LWT.get(track_key_lwt_vel_ema, []))
            # Use of the last element in the speed list as the current speed
            current_lwt_vel = track_lwt_ema_list[-1]
            # Checking that the train is moving
            if current_lwt_vel:
                # Check that the current speed of the wagon is greater than the set threshold for the driving of the wagon
                if current_lwt_vel > move_velocity_lwt:
                    # Get the values of the intersections of a person with a moving wagon
                    p_feet_wagon_intersections_vals = tracker_PJ.feet_wagon_intersections.get(
                        track_key_p_feet_wagon_intersection,
                        deque(maxlen = int(feet_wagon_intersections_history_len) + 1))
                    feet_wagon_IoA = intersection_polygon_rect_area(mask_lwt, box_feet_p) / rect_area(box_feet_p)

                    # For tests
                    # print(f"person{track_id_p} = {feet_wagon_IoA}")

                    # Check that the intersection of human legs and moving wagon is greater than threshold_ioa
                    # Record '1' if a person's feet overlap with a moving wagon, otherwise '0''
                    if feet_wagon_IoA > threshold_ioa:
                        p_feet_wagon_intersections_vals.append(1)
                    else:
                        p_feet_wagon_intersections_vals.append(0)

                    # Record up-to-date human intersections
                    tracker_PJ.feet_wagon_intersections[
                        track_key_p_feet_wagon_intersection] = p_feet_wagon_intersections_vals

                    # Business logic test
                    if 1 in p_feet_wagon_intersections_vals and 0 in list(p_feet_wagon_intersections_vals)[-int(feet_wagon_intersections_history_len/skip_interval):]:
                        if show_task:
                            # Image of the warning text
                            cv2.putText(img, alert_text, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - txt_up_alert),
                                        font, fontscale_alert, color_alert, thick_alert)
                        # Pointing out that logic worked.
                        database.triggered_3_1 = True
                        database.triggered_time_3_1 = datetime.now()
                        database.people_ids["3_1"] = int(track_id_p)

    # For tests
    # print(tracker_PJ.feet_wagon_intersections)

    return img


# =================================================================================================================#
# Task_3_3: Travel of staff member on the step, outside staircase of the tank reciprocally incorporated with the step̆            #
# =================================================================================================================#
def task_3_3(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    font: int,
    move_velocity_lwt: float,
    move_velocity_p: float,
    atol_velocity: float,
    threshold_ioa: float,
    feet_ratio: float,
    show_task: bool,
    alert_text: str,
    color_alert: Tuple[int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
) -> np.ndarray:
    """
    Function to determine whether a staff member is caught on a moving wagon, locomotive, tank.
    
    Args:
        img (np.ndarray): Reference image.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        traker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        database (Database): A copy of the Database &apos; s Logistical Information Storage Class.
        font (int): Font.
        move_velocity_lwt (float): Threshold value of the speed at which the wagon starts to move.
        move_velocity_p (float): Threshold value of human speed at which motion begins.
        atol_velocity (float): Absolute tolerance velocity - the difference between the speeds of the wagon and the person, less than which is believed to be equal to the speeds.
        threshold_ioa (float): IoA threshold value when logic activates.
        feet_ratio (float): Human leg position threshold.
        show_task (bool): Show warning message or not.
        alert_text (str): Warning message text.
        color_alert (Tuple[int]): Logic warning message color.
        txt_up_alert (int): How high the warning message distances from the upper boundary of the object frame in pixels.
        fontscale_alert (float): The size of the warning message text.
        thick_alert (int): Thickness of the warning message text.

    Returns:
        img (np.ndarray): Image with warning text when a person is caught on a wagon.
    """
    # Logistical activation
    database.triggered_3_3 = False

    # Getting the results of people and wagons tracking
    results_tracking_p = tracker_PJ.results_persons
    track_history_P = tracker_PJ.track_history
    results_tracking_lwt = traker_LWT.results
    track_history_LWT = traker_LWT.track_history
    class_names_LWT = traker_LWT.class_names

    # We find all the Bboxes people's legs.
    persons_feet = get_persons_feet(results_tracking_p, feet_ratio)

    # Overtaken people's track results
    for box_xyxyx_p, box_feet_p, track_id_p in zip(results_tracking_p['boxes_xyxy'], persons_feet,
                                                   results_tracking_p['track_ids']):
        # Extract of current human speed for the track
        track_key_p_vel_ema = f"person_{track_id_p}_ema"
        # Getting a human speed list from the tracking history
        track_p_ema_list = list(track_history_P.get(track_key_p_vel_ema, []))
        # Use of the last element in the speed list as the current speed
        current_p_vel = track_p_ema_list[-1]

        # We find the lowest point in a man's legs.
        box_feet_p_y_lowest = box_feet_p[1::2].max()

        # Review of train track results
        for box_lwt, _, mask_lwt, cls_lwt, _, track_id_lwt in zip(*results_tracking_lwt.values()):
            # Man's feet are on the wagon.
            # We find the lowest point of the wagon.
            mask_lwt_y_lowest = mask_lwt[:, 1].max()
            if box_feet_p_y_lowest > mask_lwt_y_lowest:
                continue

            # The current speed of the wagon for the track
            track_key_lwt_vel_ema = f"{class_names_LWT[cls_lwt]}_{track_id_lwt}_ema"
            # Getting the train speed list from the tracking history
            track_lwt_ema_list = list(track_history_LWT.get(track_key_lwt_vel_ema, []))
            # Use of the last element in the speed list as the current speed
            current_lwt_vel = track_lwt_ema_list[-1]
            # Checking that the train is moving
            if current_lwt_vel:
                # Check that the current speed of the wagon is greater than the set threshold for the driving of the wagon
                if current_lwt_vel > move_velocity_lwt and current_p_vel > move_velocity_p:
                    # Verification of the conditions of speed convergence and intersection of regions
                    if (current_lwt_vel - current_p_vel) < atol_velocity:
                        # Compute Intersection over Area - IoA)
                        IoA = intersection_polygon_rect_area(mask_lwt, box_xyxyx_p) / rect_area(box_xyxyx_p)
                        # Verification that the coefficient of intersection is greater than the threshold
                        if IoA > threshold_ioa:
                            if show_task:
                                # Image of the warning text
                                cv2.putText(img, alert_text, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - txt_up_alert),
                                            font, fontscale_alert, color_alert, thick_alert)
                            # Pointing out that logic worked.
                            database.triggered_3_3 = True
                            database.triggered_time_3_3 = datetime.now()
                            database.people_ids["3_3"] = int(track_id_p)

    return img


# ====================================================================================================#
# Task_4_2 Railway crossing in front of rolling rolling stock           #
# ====================================================================================================#

def task_4_2(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    database: DataBase,
    font: int,
    move_velocity_lwt: float,
    threshold_ioa: float,
    thrld_dist_w_r: float,
    feet_ratio: float,
    alert_text: str,
    color_alert: Tuple[int, int, int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
    show_task: bool,
    show_debug: bool,
) -> np.ndarray:
    """
    Function to determine whether a staff member passes through rails in front of moving rolling stock.
    
    Args:
        img (np.ndarray): Reference image.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        tracker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        database (DataBase): A copy of the database class to store logic response information.
        font (int): Font.
        move_velocity_lwt (float): Threshold value of the speed of the wagon at which the logic begins.
        threshold_ioa (float): Railway intersection threshold.
        thrld_dist_w_r (float): Proportion of the height of the wagon on which the centre of the intersection of the rails and the wagon shall be situated.
        feet_ratio (float): Human leg position threshold.
        alert_text (str): Warning message text.
        color_alert (Tuple[int]): Colour of the warning message.
        txt_up_alert (int): Detachment of the text of the warning message from the upper left corner of the Bbox object.
        fontscale_alert (float): The size of the warning message text.
        thick_alert (int): Thickness of the warning message text.
        show_task (bool): Key message display flag.
        show_debug (bool): Debug display flag on frame.
    
    Returns:
        img (np.ndarray): Image with warning text.
    """
    # Logistical activation
    database.triggered_4_2 = False

    # Get a list of moving wagons and crossed rails
    wagons_rails_link = get_moving_wagons_rails(tracker_LWT, results_r, move_velocity_lwt,
                                                thrld_dist_w_r, show_debug, img)

    # If there are no moving wagons, we don't do anything.
    if not wagons_rails_link:
        return img

    # Getting people's feet coordinates.
    persons_feet = get_persons_feet(tracker_PJ.results_persons, feet_ratio)

    # See if any of the people cross the rails in front of a moving car
    for box_feet_p, box_xyxyx_p, track_id_p in zip(persons_feet, tracker_PJ.results_persons['boxes_xyxy'],
                                                   tracker_PJ.results_persons['track_ids']):
        if is_person_crossing_rails(box_feet_p, wagons_rails_link, results_r, threshold_ioa)[1]:
            if show_task:
                cv2.putText(img, alert_text,
                            (int(box_xyxyx_p[0]), int(box_xyxyx_p[1] - txt_up_alert)), font,
                            fontscale_alert, color_alert, thick_alert)

            # Logic worked.
            database.triggered_4_2 = True
            database.triggered_time_4_2 = datetime.now()
            database.people_ids["4_2"] = int(track_id_p)

    return img


# ====================================================================================================================#
# Task_5_2: Passing the track near a standing convoy less than 5 m from the head or tail of the convoy         #
# ====================================================================================================================#
def task_5_2(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    database: DataBase,
    font: int,
    alert_distance: float,
    thrd_ioa_p_r: float,
    thrld_dist_w_r: float,
    feet_ratio: float,
    alert_text: str,
    color_alert: Tuple[int, int, int],
    txt_up_alert: int,
    fontscale_alert: float,
    thick_alert: int,
    show_task: bool,
    show_debug: bool,
) -> np.ndarray:
    """
    Function to determine whether the road passes near the standing wagon at a distance of less than 5 meters from the head or tail of the wagon.

    Args:
        img (np.ndarray): Reference image.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        tracker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        results_r (Dict[str, np.ndarray]): Results of tracking rails.
        database (DataBase): A copy of the database class to store logic response information.
        font (int): Font.
        alert_distance (float): Safe distance at which a person shall not approach a wagon.
        thrd_ioa_p_r (float): IoA threshold value when logic activates.
        thrld_dist_w_r (float): Proportion of the height of the wagon on which the centre of the intersection of the rails and the wagon shall be situated.
        feet_ratio (float): Human leg position threshold.
        alert_text (str): Warning message text.
        color_alert (Tuple[int]): Colour of the warning message.
        txt_up_alert (int): Detachment of the text of the warning message from the upper left corner of the Bbox object.
        fontscale_alert (float): The size of the warning message text.
        thick_alert (int): Thickness of the warning message text.
        show_task (bool): Key message display flag.
        show_debug (bool): Debug display flag on frame.

    Returns:
        img (np.ndarray): Image with warning text.
    """
    # Logistical activation
    database.triggered_5_2 = False

    # Get a list of moving wagons and crossed rails
    wagons_rails_link, indexR_cornerPoints = get_wagons_rails(tracker_LWT, results_r, thrld_dist_w_r, show_debug, img)

    # Getting people's feet coordinates.
    persons_feet = get_persons_feet(tracker_PJ.results_persons, feet_ratio)

    # Check if any of the people cross the rails in front of the car
    for box_feet_p, box_xywh_p, track_id_p in zip(persons_feet, tracker_PJ.results_persons['boxes'],
                                                  tracker_PJ.results_persons['track_ids']):
        index_r, cross_status = is_person_crossing_rails(box_feet_p, wagons_rails_link, results_r, thrd_ioa_p_r)
        if cross_status:
            # We're finding an increasing factor by which we can determine the actual distance between the wagon and the person
            # 1.7 - Average human growth in metres
            k = 1.7 / box_xywh_p[3]

            # We find every distance between a man and a wagon standing on a track that man crossed
            if index_r in wagons_rails_link:
                # We find the coordinates of the human center.
                box_center_p = box_xywh_p[:2]
                # Corner point of wagon No. 1
                corner_point_1_lwt = indexR_cornerPoints[index_r][0]
                corner_point_2_lwt = indexR_cornerPoints[index_r][1]
                # We find the smallest distance between man and the corner points of the wagon
                distance = float('inf')
                # Save the coordinates of the angular point of the wagon from which a person has the smallest distance
                corner_point_lwt_min_distance = None
                for point in [corner_point_1_lwt, corner_point_2_lwt]:
                    d = euclidean_distance(box_center_p, point)
                    if d < distance:
                        distance = d
                        corner_point_lwt_min_distance = point

                # Move the distance to the real scale.
                distance = distance * k

                # If the distance is less than alert_distance (default = 5 m))
                if distance < alert_distance:
                    if show_task:
                        # An allergic image of the event
                        cv2.putText(img, alert_text,
                                    (int(box_xywh_p[0]), int(box_xywh_p[1] - txt_up_alert)), font,
                                    fontscale_alert, color_alert, thick_alert)
                    # Logic worked.
                    database.triggered_5_2 = True
                    database.triggered_time_5_2 = datetime.now()
                    database.people_ids["5_2"] = int(track_id_p)

                if show_debug:
                    # Image of distance between the wagon and the person
                    cv2.putText(img, f"Distance: {distance} m",
                                (int(box_xywh_p[0]), int(box_xywh_p[1])), font,
                                fontscale_alert, [0, 255, 255], thick_alert)
                    # Image of the straight line between the wagon and the person
                    cv2.line(img, box_center_p, corner_point_lwt_min_distance, color_alert, thick_alert)

    return img


# !!!! task_6_2_frame function is not used in the current implementation
# ==========================================================================#
# Task_6_2: Working on or under couplings         #
# ==========================================================================#
def task_6_2_frame(
    img: np.ndarray,
    results_p: dict,
    results_lwt: dict,
    font: int,
    wagon_mask: bool,
    thrd_iou: float,
    thrd_p_hw_bbox: float,
    alert_text: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """
    Function to determine whether an employee is passing under a standing wagon, locomotive or tank.
    
    Args:
        img (np.ndarray): Reference image.
        results_p (dict): Model prediction results for humans, contain Bboxes coordinates and object classes.
        results_lwt (dict): Prognosis results for wagons, locomotives and tanks, contains Bboxes coordinates and objects classes.
        font: Font for text withdrawal.
        wagon_mask (bool): Do you use the wagon mask to calculate the intersection.
        thrd_iou (float): Intersection over Union (IoU) threshold for determining the intersection of a person and a wagon.
        thrd_p_hw_bbox (float): Threshold value of the height to width ratio of Bbox person.
        alert_text (str): Warning text.
        text_up (int): Movement of text up from Bbox start point.
        color (tuple): Text color.
        fontscale (float): The size of the font.
        thickness (int): Text line thickness.

    Returns:
        img (np.ndarray): Image with warning text when a person is found under a wagon.
    """

    # Overtaken all detected human/housing objects in predictions
    for box_p in results_p['boxes_xyxy']:
        # Overtaking of all detected "wagon/locomotive/tank" sites in predictions
        for _, box_lwt, mask_lwt, cls_lwt, _, _ in zip(*results_lwt.values()):
            # Verification that the object found is a "wagon/tank""
            if cls_lwt != 1:
                # Calculation of the height and width of the Bbox person
                box_p_height = box_p[3] - box_p[1]
                box_p_width = box_p[2] - box_p[0]
                # Choice of method for calculation of intersection: mask or Bbox
                if wagon_mask:
                    # If a mask is used, calculate Intersection over Area (IoA))
                    IoA = intersection_polygon_rect_area(mask_lwt, box_p) / rect_area(box_p)
                else:
                    # If mask is not used, calculate Intersection over Area for Bboxes
                    IoA = intersection_over_area_bbox(box_p, box_lwt)
                # Checking the intersectional conditions and the bbox-size ratio
                if IoA > thrd_iou and box_p_width >= (box_p_height * thrd_p_hw_bbox):
                    # Put the warning text on the image.
                    cv2.putText(img, alert_text,
                                (int(box_p[0]), int(box_p[1]) - text_up),
                                font, fontscale, color, thickness)
    return img


# ========================================================================================#
# Task_6_1: Workers &apos; access under standing wagons                    #
# ========================================================================================#
def task_6_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    skip_interval: int,
    font: int,
    task_6_1_show: bool,
    time_absent_6_1: int,
    edge_threshold: int,
    thrd_ioa: float,
    alert_text_6_1: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
) -> np.ndarray:
    """
    Analyze the image to determine whether the workers are seated or under the wagons, 
    and annotate it when such cases are detected.

    Args:
        img (np.ndarray): Reference image for annotations.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        traker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        database (Database): A copy of the Database &apos; s Logistical Information Storage Class.
        skip_interval (int): Staff pass interval.
        font (int): Text font.
        task_6_1_show (bool): Do you need to display a warning for the task 6.1.
        time_absent_6_1 (int): Time during which a person is believed to have disappeared.
        edge_threshold (int): Threshold value for frame boundary determination.
        thrd_ioa (float): Threshold value for the intersection of the wagon and man.
        alert_text_6_1 (str): Warning text for task 6.1.
        text_up (int): Movement of text up from Bbox start point.
        color (tuple): Text color.
        fontscale (float): The size of the font.
        thickness (int): Text line thickness.

    Returns:
        img: Annotated image.
    """
    # Logistical activation
    database.triggered_6_1 = False

    # Getting the results of people and wagons tracking
    track_history_P = tracker_PJ.track_history
    last_update_time_p = tracker_PJ.last_update_time
    results_tracking_lwt = traker_LWT.results

    for track_id_p_abs, last_update_time in last_update_time_p.items():
        if (time() - last_update_time) > (time_absent_6_1 / skip_interval):
            # Checking that a person is missing and not out of frame.
            if track_history_P.get(f"person_{track_id_p_abs}_c_xy") is not None:
                last_known_position = np.array(list(track_history_P.get(f"person_{track_id_p_abs}_c_xy", [])))[-1]

                # Checking that a person is missing and not out of frame.
                if last_known_position is not None:
                    x, y = last_known_position[:2]
                    h, w = last_known_position[2:]
                    box_xyxyx_p = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                    frame_height, frame_width = img.shape[:2]

                    # Check that the last known position is not close to the edges of the frame
                    if (edge_threshold < x < frame_width - edge_threshold and
                            edge_threshold < y < frame_height - edge_threshold):

                        # Overtaken train track results
                        for _, _, mask_lwt, _, _, _ in zip(*results_tracking_lwt.values()):
                            # IoA calculation for each wagon
                            IoA = intersection_polygon_rect_area(mask_lwt, box_xyxyx_p) / rect_area(box_xyxyx_p)
                            # Verification of the conditions for the crossing of persons with wagons
                            if IoA > thrd_ioa:
                                # Shows the warning text in the image
                                if task_6_1_show:
                                    cv2.putText(img, alert_text_6_1, (int(x), int(y) - text_up), font,
                                                fontscale, color, thickness)
                                # To indicate that logic has been activated
                                database.triggered_6_1 = True
                                database.triggered_time_6_1 = datetime.now()
                                database.people_ids["6_1"] = int(track_id_p_abs)
    return img


# ========================================================================================#
# Task_6_2: Working on or under couplings                #
# Task_8.1.1: Identification of employees sitting in the observation zone               #
# (Joint implementation of logic for the tasks 6.2 and 8.1.1)                   #
# ========================================================================================#
def task_6_2_and_8_1_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    traker_LWT: TrackerLWT,
    database: DataBase,
    skip_interval: int,
    task_6_2_show: bool,
    task_8_1_1_show: bool,
    wagon_mask: bool,
    buffer_len_6_2: int,
    buffer_len_8_1_1: int,
    thrd_ioa: float,
    thrd_p_hw_6_2: float,
    thrd_p_hw_8_1_1: float,
    thrd_p_hw_8_1_1_down: float,
    show_p_hw: bool,
    alert_text_6_2: str,
    alert_text_8_1_1: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
    font,
) -> np.ndarray:
    """
    Analyze the image to determine whether the workers are seated or under the wagons,
    and annotate them when such cases are detected.

    Args:
        img (np.ndarray): Reference image for annotations.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        traker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        database (Database): A copy of the Database &apos; s Logistical Information Storage Class.
        skip_interval (int): Staff pass interval.
        task_6_2_show (bool): Do you need to display a warning for the task 6.2.
        task_8_1_1_show (bool): Do you need to display a warning for the task 8.1.1.
        wagon_mask (bool): Do you use the wagon mask to calculate intersections.
        buffer_len_6_2 (int): The length of the buffer to analyse the history of the 6.2 task.
        buffer_len_8_1_1 (int): The length of the buffer to analyse the history of the 8.1.1 task.
        thrd_ioa (float): IoA threshold for hazard determination.
        thrd_p_hw_6_2 (float): Threshold value for the determination of access under the wagon.
        thrd_p_hw_8_1_1 (float): Sitting threshold.
        thrd_p_hw_8_1_1_down (float): The upper exit threshold is false.
        show_p_hw (bool): Do you see the ratio of height to width of bbox.
        alert_text_6_2 (str): Warning text for task 6.2.
        alert_text_8_1_1 (str): Warning text for task 8.1.1.
        text_up (int): Upward displacement.
        color (tuple): Text color.
        fontscale (float): The size of the font.
        thickness (int): Text line thickness.
        font (int): Text font.

    Returns:
        img: Annotated image.
    """
    # Logistical activation
    database.triggered_6_2 = False
    database.triggered_8_1_1 = False

    # Getting the results of people and wagons tracking
    results_tracking_p = tracker_PJ.results_persons
    track_history_P = tracker_PJ.track_history
    results_tracking_lwt = traker_LWT.results

    # Overtaken people's track results
    for box_p, box_xyxyx_p, track_id_p in zip(results_tracking_p['boxes'], results_tracking_p['boxes_xyxy'],
                                              results_tracking_p['track_ids']):
        # History of HW Relationship Changes to the Track
        track_key_hw = f"p_{track_id_p}_hw"
        track_hw = list(track_history_P.get(track_key_hw, []))
        hw_ratio = box_p[2] / box_p[3]

        # Painting hw human bbox relationships
        if show_p_hw:
            cv2.putText(img, f'{hw_ratio:.2f} ID: {track_id_p}', (box_p[0], box_p[1]), font, fontscale, color,
                        thickness)

        # IoA storage list for each wagon
        IoA_list = []
        # Overtaken train track results
        for box_lwt, box_xyxyx_lwt, mask_lwt, cls_lwt, _, _ in zip(*results_tracking_lwt.values()):
            # Choice of method for calculation of intersection: mask or Bbox
            if wagon_mask:
                IoA = intersection_polygon_rect_area(mask_lwt, box_xyxyx_p) / rect_area(box_xyxyx_p)
            else:
                IoA = intersection_over_area_bbox(box_xyxyx_p, box_xyxyx_lwt)
            IoA_list.append(IoA)
            # Verification that the object found is a "wagon/tank""
            if cls_lwt != 1:
                # Verification of the conditions for the crossing of persons with wagons
                if IoA > thrd_ioa:
                    # Task 6.2: Intruding workers under standing wagons
                    # Verification of the conditions of the stratum of changes hw of human relations
                    if len(track_hw) >= buffer_len_6_2 and all(h > thrd_p_hw_6_2 for h in track_hw[-buffer_len_6_2:]):
                        if task_6_2_show:
                            # Figure of warning text 6_2
                            cv2.putText(img, alert_text_6_2, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - text_up), font,
                                        fontscale, color, thickness)
                        # Pointing out that logic worked.
                        database.triggered_6_2 = True
                        database.triggered_time_6_2 = datetime.now()
                        database.people_ids["6_2"] = int(track_id_p)

        # Task 8.1.1: Identification of employees sitting in the observation zone
        # Check that IoA < thrd_ioa for all wagons
        if not any(IoA_list) > thrd_ioa:
            if len(track_hw) >= (buffer_len_8_1_1 / skip_interval) and all(
                    (thrd_p_hw_8_1_1 < h < thrd_p_hw_8_1_1_down) for h in track_hw[-(int(buffer_len_8_1_1 / skip_interval)):]):
                if task_8_1_1_show:
                    cv2.putText(img, alert_text_8_1_1, (int(box_xyxyx_p[0]), int(box_xyxyx_p[1]) - text_up), font,
                                fontscale, color, thickness)
                # Pointing out that logic worked.
                database.triggered_8_1_1 = True
                database.triggered_time_8_1_1 = datetime.now()
                database.people_ids["8_1_1"] = int(track_id_p)
    return img


# ======================================================================================================#
# Task_7 Identification of intersections of W-tracks in a prohibited area or in a dangerous trajectory       #
# ======================================================================================================#
def task_7(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    results_r: dict,
    database: DataBase,
    skip_interval: int,
    db_enabled: bool,
    task_1_2_enabled: bool,
    task_7_1_enabled: bool,
    font: int,
    move_velocity_p: float,
    triggered_1_2: dict,
    person_ids_1_2: dict,
    feet_ratio: float,
    thrd_ioa_p_r: float,
    color_danger: Tuple[int],
    color_safe: Tuple[int],
    start_track_7: int,
    direction_calc_buffer: int,
    thrd_angle_7_2_1: float,
    show_task_7_2_2: bool,
    show_task_7_2_1: bool,
    text_up_7: int,
    alert_text_7_1: str,
    alert_text_7_2_1: str,
    alert_text_7_2_2: str,
    color_alert: Tuple,
    fontscale: float,
    thickness: int,
    filter_rails: bool,
    r_p_filter_scale: int = 10,
    show_debug: bool = False,
) -> np.ndarray:
    """
    It performs tracking analysis tasks and identifies the crossings of the railway routes in the prohibited zone or on a dangerous trajectory.

    Args:
        img (np.ndarray): Image of the results of the analysis.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        results_r (dict): Results of rail segmentation.
        database (Database): A copy of the database class to store logic response information.
        skip_interval (int): Staff pass interval.
        db_enabled (bool): Flag to include the database.
        task_1_2_enabled (bool): Flag to include task 1_2.
        task_7_1_enabled (bool): Flag to include task 7_1.
        font (int): The font for the text in the image.
        triggered_1_2 (bool): Logic response flag 1_2.
        person_id_1_2 (int): Human Identifier for task 1_2.
        move_velocity_p (float): Human threshold speed.
        thrd_ioa_p_r (float): Human rail crossing threshold.
        color_danger (Tuple[int]): Color for dangerous areas display.
        color_safe (Tuple[int]): Color for the display of safe areas.
        start_track_7 (int): Initial number of people monitored for task 7.
        direction_calc_buffer (int): Buffer for the calculation of the direction of traffic.
        thrd_angle_7_2_1 (float): The angle threshold for task 7_2_1.
        show_task_7_2_2 (bool): Task flag 7_2_2.
        show_task_7_2_1 (bool): Task flag 7_2_1.
        text_up_7 (int): Y-axis offset for task 7.
        alert_text_7_1 (str): Warning text for task 7_1.
        alert_text_7_2_1 (str): Warning text for task 7_2_1.
        alert_text_7_2_2 (str): Warning text for task 7_2_2.
        color_alert (Tuple): Color for warning text.
        fontscale (float): The size of the font for the text.
        thickness (int): Text thickness.
        filter_rails (bool): Lamp filtering flag.
        r_p_filter_scale (int, optional): The filtering rate of the rail contour, default 10.
        show_debug (bool): Whether to display debugging information on frame or not, by default False.

    Returns:
        img (np.ndarray): Image with warnings.
    """
    # Logistical activation
    database.triggered_7_1 = False
    database.triggered_7_2_2 = False
    database.triggered_7_2_1 = False

    # Getting the results of people and wagons tracking
    results_tracking_p = tracker_PJ.results_persons
    track_history_P = tracker_PJ.track_history

    # We find all the Bboxes people's legs.    
    persons_feet = get_persons_feet(results_tracking_p, feet_ratio)

    # We're going through all the predictions of people and wagons to calculate the logic
    for box_xyxy_p, person_feet, id_p in zip(results_tracking_p['boxes_xyxy'], persons_feet,
                                             results_tracking_p['track_ids']):
        # Application of human foot area to the image
        if show_debug:
            img = plot_bbox(img, person_feet, (255, 128, 0), thickness)
        for mask_r in results_r['masks']:
            # Calculation of the area of the crossing of persons and rails
            IoA_p_r = intersection_over_area_mask(person_feet, mask_r)

            # Checking the threshold for a person crossing a rail
            if IoA_p_r > thrd_ioa_p_r:
                if task_7_1_enabled and db_enabled and task_1_2_enabled:
                    # Checking that 1.2 logic worked and the id man crossing the rail matches the id man who's stepping on the rail
                    if triggered_1_2 is not None and person_ids_1_2 is not None:
                        if any(triggered_1_2.values()) and int(id_p) in person_ids_1_2.values():
                            # Display of the warning text
                            cv2.putText(img, alert_text_7_1, (int(box_xyxy_p[0]), int(box_xyxy_p[1] + (text_up_7 + 30))),
                                        font,
                                        fontscale, color_danger, thickness)
                            # Pointing out that logic worked.
                            database.triggered_7_1 = True
                            database.triggered_time_7_1 = datetime.now()
                            database.people_ids['7_1'] = int(id_p)

                # task 7_2
                track_key_p_vel_ema = f"person_{id_p}_ema"
                track_p_ema_list = list(track_history_P.get(track_key_p_vel_ema, []))
                if len(track_p_ema_list) > start_track_7:
                    track_p_ema_list = track_p_ema_list[-start_track_7:]
                    # task 7_2_2 - man standing on rails
                    if all(v < move_velocity_p for v in track_p_ema_list):
                        if show_task_7_2_2:
                            # Display of the warning text
                            cv2.putText(img, alert_text_7_2_2, (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - text_up_7),
                                        font, fontscale, color_alert, thickness)
                        # The indication that logic 7_2_2 has worked
                        database.triggered_7_2_2 = True
                        database.triggered_time_7_2_2 = datetime.now()
                        database.people_ids['7_2_2'] = int(id_p)
                    else:
                        # task 7_2_1 - rail traffic
                        # History of human displacement
                        track_key_p_c_xy = f"person_{id_p}_c_xy"
                        track_p_c_xy_list = np.array(list(track_history_P.get(track_key_p_c_xy, [])))[:, :2]

                        # History of the human movement
                        track_key_p_direction = f"person_{id_p}_dir_ema"
                        track_p_direction_list = list(track_history_P.get(track_key_p_direction, []))
                        p_angle_2x = track_p_direction_list[-1]

                        # Human path diagram
                        if show_debug:
                            img = plot_line_from_point(img, track_p_c_xy_list[-1], p_angle_2x, color_safe,
                                                       thickness * 2, 400)

                        if filter_rails:
                            # Filtering of rail contours in a radius from man
                            # The filter_contour_points function filters the rail contour points according to their distance
                            # To the last point of the human track.
                            # The radius of filtering is defined as the height of human feet multiplied by the factor r_p_filter_scale.
                            filtered_contour_r = filter_contour_points(mask_r, track_p_c_xy_list[-1], (
                                    int(person_feet[3]) - int(person_feet[1])) * r_p_filter_scale)
                            # Drawing the contour of filtered rails
                            if show_debug:
                                img = plot_contour(img, filtered_contour_r, color=[255, 128, 0], thickness=2)

                            # Approximation of the rail line
                            r_angle_2x, x_range, y_range = approximate_rail_line(filtered_contour_r, img.shape[0])

                        else:
                            r_angle_2x, x_range, y_range = approximate_rail_line(mask_r, img.shape[0])

                        # Graph of the annexed rail line
                        if show_debug:
                            img = plot_approximated_rail_line(img, x_range, y_range, color_danger, thickness * 2)

                        # Verification of calculated angles
                        if len(track_p_direction_list) > (direction_calc_buffer / skip_interval) and r_angle_2x is not None:
                            track_p_dir_last = np.array(list(track_p_direction_list)[-int(direction_calc_buffer / skip_interval):],
                                                        dtype=np.float32)

                            # Display the angles in the image
                            # cv2.putText(img, f'p:{track_p_dir_last[-1]:.2f} r:{r_angle_2x:.2f}', (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - (text_up_7+20)), font, fontscale, color_alert, thickness)

                            # Verification of the condition of parallel movement of human beings and rails
                            # If the difference between the angles is less than the threshold value thrd_angle_7_2_1, then it shall be considered,
                            # It's true that man is moving along the tracks.
                            condition_met_percentage = np.mean(
                                (abs(track_p_dir_last - r_angle_2x) < thrd_angle_7_2_1).astype(int))
                            if condition_met_percentage > 0.6:
                                # Warning management 7_2_1
                                if show_task_7_2_1:
                                    cv2.putText(img, alert_text_7_2_1,
                                                (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - text_up_7), font, fontscale,
                                                color_alert, thickness)
                                # The indication that logic 7_2_1 has worked
                                database.triggered_7_2_1 = True
                                database.triggered_time_7_2_1 = datetime.now()
                                database.people_ids['7_2_1'] = int(id_p)

    return img


# ====================================================================================================#
# Task_9 Definition of foreign objects in the designated area of the station                          #
# ====================================================================================================#
def task_9(
    img: np.ndarray,
    inferencer_world: InferencerWorld,
    world_model_params: dict,
    database: DataBase,
    task_9_show: bool,
    visualize_inference_params: dict,
    font: int
):
    """
        Performs foreign object detectives in the station &apos; s areas.

        Args:
            img (np.ndarray): Image of the results of the analysis.
            inferencer_world (InferencerWorld): copy of the Class InferencerWorld for the detection of foreign objects.
            world_model_params (dict): YOLO WORLD model parameters.
            database (DataBase): A copy of the database class to store logic response information.
            task_9_show (bool): The flag for the display of task 9.
            visualize_inference_params (dict): parameters for model inference display.
            font (int): The font for the text in the image.

        Returns:
            img (np.ndarray): Annotated image.
        """
    # Logistical activation
    database.triggered_9 = False

    # Definition of foreign objects on frame
    results_world, names_world = inferencer_world.predict_on_image_frame(img, "World",
                                                                         **world_model_params)
    # If at least one foreign object is found on the frame
    if results_world["boxes_xyxy"].any():
        # Pointing out logic 9 worked.
        database.triggered_9 = True
        database.triggered_time_9 = datetime.now()

        if task_9_show:
            # Visualization
            img = plot_results(results_world, names_world, img, 'world', font,
                               **visualize_inference_params)

    return img


# ====================================================================================================#
# Task_10_1 Definition of non-maintenance                             #
# ====================================================================================================#
def task_10_1(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    database: DataBase,
    task_10_1_show: bool,
    alert_text_10_1: str,
    thrd_phw_10_1: float,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
    font: int,
) -> np.ndarray:
    """
        The definition of a person's non-standard position works when a person is found to be lying.

        Args:
            img (np.ndarray): Image of the results of the analysis.
            tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
            database (Database): A copy of the database class to store logic response information.
            task_10_1_show (bool): The display flag.
            alert_text_10_1 (str): Warning text.
            thrd_phw_10_1 (float): The threshold for the ratio of height to width bbox.
            text_up (int): Upshift of text.
            color (tuple): Text color.
            fontscale (float): Text size.
            thickness (int): Text thickness.
            font: Text font.

        Returns:
            img (np.ndarray): Annotated image.
        """
    # Logistical activation
    database.triggered_10_1 = False

    # Results of human tracking
    results_tracking_p = tracker_PJ.results_persons

    # Definition of the non-standard position of a person on the frame
    for box_p, track_id_p in zip(results_tracking_p['boxes'], results_tracking_p['track_ids']):
        if box_p[3] / box_p[2] <= thrd_phw_10_1:
            if task_10_1_show:
                cv2.putText(img, alert_text_10_1, (int(box_p[0]), int(box_p[1]) - text_up), font,
                            fontscale, color, thickness)
            # Pointing out that logic worked.
            database.triggered_10_1 = True
            database.triggered_time_10_1 = datetime.now()
            database.people_ids["10_1"] = int(track_id_p)

    return img


# ======================================================================================================#
# Task_10_2 Existence of deviant behaviour)                       #
# ======================================================================================================#
def task_10_2(
    img: np.ndarray,
    tracker_PJ: TrackerPJ,
    tracker_LWT: TrackerLWT,
    database: DataBase,
    deviant_history: dict,
    skip_interval: int,
    move_velocity_p: float,
    task_10_2_show: bool,
    buffer_len_10_2: int,
    buffer_len_results_10_2: int,
    deviant_thrlds: dict,
    alert_text_10_2: str,
    text_up: int,
    color: tuple,
    fontscale: float,
    thickness: int,
    font: int,
    show_debug: bool = False,
) -> np.ndarray:
    """
    Analyses and annotates the image to determine signs of deviant walking in employees.

    Args:
        img (np.ndarray): Image for analysis.
        tracker_PJ (TrackerPJ): A copy of the tracker class for people and vests.
        tracker_LWT (TrackerLWT): Tracker class copy for wagons.
        database (Database): A copy of the database class to store logic response information.
        deviant_history (dict): History of deviant behaviour parameters.
        skip_interval (int): Staff pass interval.
        move_velocity_p (float): Human threshold speed.
        task_10_2_show (bool): The display flag.
        buffer_len_10_2 (int): The length of the buffer for analysis.
        buffer_len_results_10_2 (int): The length of the buffer to store the analysis.
        deviant_thrlds (dict): Threshold values for deviant walking analysis.
        alert_text_10_2 (str): Warning text.
        text_up (int): Upshift of text.
        color (tuple): Text color.
        fontscale (float): Text size.
        thickness (int): Text thickness.
        font: Text font.
        show_debug (bool): Whether to display debugging information on frame or not, by default False.

    Returns:
        img (np.ndarray): Annotated image.
    """
    # Initialization of the flag of logic
    database.triggered_10_2 = False

    # Getting the results of people and wagons tracking
    results_tracking_p = tracker_PJ.results_persons
    results_tracking_lwt = tracker_LWT.results
    track_history_P = tracker_PJ.track_history

    # Overtaken people's track results
    for box_p, box_xyxy_p, track_id_p in zip(results_tracking_p['boxes'], results_tracking_p['boxes_xyxy'], results_tracking_p['track_ids']):
        # Verification of a sufficiently large bbox condition
        if box_p[3] > deviant_thrlds["p_height"]:
            # Checking for a bbox crossing from a bbox wagon
            if any(intersection_polygon_rect_area(mask_lwt, box_xyxy_p) / rect_area(box_xyxy_p) > deviant_thrlds["intersection_threshold"] 
                for _, _, mask_lwt, _, _, _ in zip(*results_tracking_lwt.values())):
                continue

            # History of human coordinates for the track
            track_p_coords_list = np.array(list(track_history_P.get(f"person_{track_id_p}_c_xy", [])))[:, :2]
            track_p_ema_list = np.array(list(track_history_P.get(f"person_{track_id_p}_ema", [])))
            track_p_direction_list = np.array(list(track_history_P.get(f"person_{track_id_p}_dir_ema", [])))

            # History of HW Relationship Changes to the Track
            track_key_hw = f"p_{track_id_p}_hw"
            track_hw = list(track_history_P.get(track_key_hw, []))

            # Checking the adequacy of history size for analysis
            if len(track_p_ema_list) >= buffer_len_10_2:
                try:
                    # Use only the last buffer_len_10_2 values
                    recent_coords = track_p_coords_list[-int(buffer_len_10_2 / skip_interval):]
                    recent_ema = track_p_ema_list[-int(buffer_len_10_2 / skip_interval):]
                    recent_directions = track_p_direction_list[-int(buffer_len_10_2 / skip_interval):]

                    # Compute current speed
                    current_velocity = recent_ema[-1]

                    # Traffic trajectory analysis
                    distance_short = norm(track_p_coords_list[-1] - track_p_coords_list[-int(buffer_len_10_2 / skip_interval)])
                    straightness = calculate_straightness(recent_coords)

                    # Speed analysis
                    velocity_std = np.std(recent_ema)
                    velocity_changes = np.diff(recent_ema)
                    velocity_change_std = np.std(velocity_changes)

                    # Direction analysis
                    direction_changes = np.diff(recent_directions)
                    direction_change_std = np.std(direction_changes)

                    # Calculation of z-scores
                    recent_straightness = [calculate_straightness(recent_coords[:i+2]) for i in range(len(recent_coords)-1)]
                    z_straightness = calculate_z_score(straightness, recent_straightness)
                    z_velocity_std = calculate_z_score(velocity_std, [np.std(recent_ema[i:i+2]) for i in range(len(recent_ema)-1)])
                    z_velocity_change_std = calculate_z_score(velocity_change_std, [np.std(velocity_changes[i:i+2]) for i in range(len(velocity_changes)-1)])
                    z_direction_change_std = calculate_z_score(direction_change_std, [np.std(direction_changes[i:i+2]) for i in range(len(direction_changes)-1)])

                    # Calculation of deviant behaviour
                    sitting_buffer = int(deviant_thrlds["sitting_buffer"] / skip_interval)
                    if len(track_hw) >= sitting_buffer and all((h > deviant_thrlds['sitting']) for h in track_hw[-sitting_buffer:]):
                        deviant_score = 0  # If a person sits, we don't think behavior is deviant.
                    elif current_velocity > move_velocity_p and distance_short >= (int(buffer_len_10_2 / skip_interval) * deviant_thrlds['distance']):
                        deviant_score = ((abs(z_straightness) + 
                                        abs(z_velocity_std) * deviant_thrlds['velocity_weight'] + 
                                        abs(z_direction_change_std)) / 3) / (box_p[3] / deviant_thrlds['height_factor'])
                    else:
                        deviant_score = 0  # If a person stops, we don't think behavior is deviant

                    # Recording of hw bbox people in track_deviant
                    track_key_deviant = f"p_{track_id_p}_deviant"  # bbox width/altitude key
                    track_deviant = deviant_history.get(track_key_deviant, deque(maxlen=int(buffer_len_results_10_2)))
                    track_deviant.append(deviant_score)  # height / width bbox
                    deviant_history[track_key_deviant] = track_deviant
                    track_deviant_list = list(track_deviant)

                    # Checking for exceeding the threshold of deviant behaviour
                    if len(track_deviant_list) >= (buffer_len_results_10_2 / skip_interval) and \
                        np.mean([int(dev > deviant_thrlds['deviant_score']) for dev in track_deviant_list[-int(buffer_len_results_10_2/skip_interval):]]) > deviant_thrlds['deviant_threshold']:
                        if task_10_2_show:
                            cv2.putText(img, alert_text_10_2, 
                                        (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - text_up), 
                                        font, fontscale, color, thickness)
                        database.triggered_10_2 = True
                        database.triggered_time_10_2 = datetime.now()
                        database.people_ids['10_2'] = int(track_id_p)

                    if show_debug:
                        print(np.mean([int(dev > deviant_thrlds['deviant_score']) for dev in track_deviant_list]))
                        debug_text = f"H: {box_p[3]}, Score: {deviant_score:.2f}, {z_straightness:.2f}, {z_velocity_std:.2f}, {z_velocity_change_std:.2f}, {z_direction_change_std:.2f}"
                        cv2.putText(img, debug_text,
                                    (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - (text_up + 30)),
                                    font, fontscale * 0.8, (0, 171, 255), thickness)

                except Exception as e:
                    if show_debug:
                        error_text = f"Error: {str(e)}"
                        cv2.putText(img, error_text,
                                    (int(box_xyxy_p[0]), int(box_xyxy_p[1]) - (text_up + 30)),
                                    font, fontscale * 0.6, (0, 0, 255), thickness)
                        print(error_text)

    return img


def calculate_z_score(value, values):
    """
    Calculates z for the set value relative to the set of values.

    Args:
        value: Value for which z-evaluation is required.
        values: Mass of values for which z-estimation is calculated.

    Returns:
        z_score: value of z-evaluation.
    """
    if len(values) < 2:
        return 0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0  # Return 0 if the standard deviation is 0
    return (value - mean) / std


def calculate_straightness(coords):
    """
    Calculates the straight line of the trajectory. Range straightness: (0 to 1)).

    Args:
        coords: Massive of trajectory coordinates.
    """
    if len(coords) < 2:
        return 1.0
    total_distance = np.sum(norm(np.diff(coords, axis=0), axis=1))
    direct_distance = norm(coords[-1] - coords[0])
    return direct_distance / (total_distance + 1e-6)
