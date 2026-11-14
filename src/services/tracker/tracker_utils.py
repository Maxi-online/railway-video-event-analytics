from collections import deque
from typing import Dict, List, Deque

import cv2
import numpy as np
from src.utils.geometry_utils import fit_line, intersection_rect_area


def get_track_person_hw(
        track_history: Dict[str, List],
        track_id: int,
        track_history_len: int,
        box: List[int]
) -> None:
    """
    Writes the ratio of width to height (spect ratio) of the object to the history of the tracking for further analysis.

    Input parameters:
        track_history (dict): A dictionary to store the history of each object,
        track_id (int): Unique Object Identifier,
        track_history_len (int): Maximum number of records stored in history,
        box (list): List with coordinates of object [x_center, y_center, width, light]
    """
    # Recording of hw bbox people in track_history
    track_key_hw = f"p_{track_id}_hw"  # bbox width/altitude key
    track_hw = track_history.get(track_key_hw, deque(maxlen=track_history_len))
    track_hw.append((box[2] / box[3]))  # Bbox width/height
    track_history[track_key_hw] = track_hw


def get_track_velocity_ema(
        img,
        box: list,
        cls: int,
        track_id: int,
        track_history: dict,
        track_history_len: int,
        class_names: dict,
        new_frame_time: float,
        prev_frame_time: float,
        velocity_ema_alpha: float,
        show_velocity: bool,
        color_velocity: List,
        thick_velocity: int,
        fontscale_velocity: float,
        font: int,
        correction: dict,
        velocity_ema_start: int = 8,
):
    """
    Processing tracking results and updating information about the speed and exponential sliding average speed of the object.

    Input parameters:
        img: Image showing speed information,
        box (list): List with coordinates of object [x_center, y_center, width, light],
        cls (int): Object class identifier,
        track_id (int): The unique track identifier of an object,
        track_history (dict): A dictionary to store the history of each object,
        track_history_len (int): Maximum number of records stored in history,
        class_names (dict): Vocabulary with object class names,
        new_frame_time (float): Time mark for current frame,
        prev_frame_time (float): Time mark for previous frame,
        velocity_ema_alpha (float): Coefficient of smoothing for exponentially sliding average,
        show_velocity (bool): Image speed flag,
        color_velocity (List): Speed text color,
        thick_velocity (int): Speed text thickness,
        fontscale_velocity (float): The size of the font for the speed text,
        font (int): Speed font,
        correction (dict): Vocabulary with speed correction parameters,
        velocity_ema_start (int, optional): Initial value for EMA, default 8.

    Return value:
        Tuple: Updated dictionary track_history and image img.
    """
    if not isinstance(track_history, dict):
        raise TypeError('track_history must be a dictionary')
    # Recorded history of 10 coordinates and width heights of the centre of the Bbox track object
    x, y, w, h = box
    track_key_c_xywh = f"{class_names[cls]}_{track_id}_c_xy"
    track_values_c_xywh = track_history.get(track_key_c_xywh, deque(maxlen=track_history_len))
    track_values_c_xywh.append((int(x), int(y), int(w), int(h)))  # x, y center point, width, height
    track_history[track_key_c_xywh] = track_values_c_xywh

    # Calculation and recording of the story of 10 speed values of the tracking object
    velocity = velocity_calc(track_history, track_key_c_xywh, new_frame_time, prev_frame_time)
    track_key_velocity = f"{class_names[cls]}_{track_id}_velocity"
    track_values_velocity = track_history.get(track_key_velocity, deque(maxlen=track_history_len))
    track_values_velocity.append(velocity)
    track_history[track_key_velocity] = track_values_velocity

    # Calculation and recording of the history of 10 values of the exponential average speed of the tracking object
    track_key_ema = f"{class_names[cls]}_{track_id}_ema"
    track_values_ema = track_history.get(track_key_ema, deque(maxlen=track_history_len))

    ema = update_ema_velocity(track_values_c_xywh,
                              track_values_velocity,
                              track_values_ema,
                              correction=correction,
                              alpha=velocity_ema_alpha,
                              velocity_ema_start=velocity_ema_start)

    track_values_ema.append(ema)
    track_history[track_key_ema] = track_values_ema

    # Showing the speed of the object in the image
    if show_velocity:
        if ema is not None:
            cv2.putText(
                img,
                f'{ema:.1f} ID: {track_id}',
                (x, y),
                font,
                fontscale_velocity,
                color_velocity,
                thick_velocity
            )
        if velocity is not None:
            cv2.putText(
                img,
                f'{velocity:.1f} ID: {track_id}',
                (x, y - 30),
                font,
                fontscale_velocity,
                (0, 255, 255),
                thick_velocity
            )


def velocity_calc(track_history: dict, track_key_c_xy: str, new_frame_time: float, prev_frame_time: float):
    """
    Calculates the speed at which an object moves by changing its position between two consecutive personnel.

    Input parameters:
        track_history (dict): A dictionary with an object track history,
        track_key_c_xy (str): Key to access the position data of the object centre,
        new_frame_time (float): Time mark for current frame,
        prev_frame_time (float): Time mark for previous frame

    Output parameters:
        velocity: Speed of movement of object.
    """
    track_data = track_history.get(track_key_c_xy, [])
    track_data = np.array(list(track_data), dtype=np.float32)
    if len(track_data) < 2:
        return 0

    # Calculation of speed based on the last two points
    displacement = track_data[-1, :2] - track_data[-2, :2]

    # #Adjusting the movement of the centre of the object to change its size
    width_correction = (track_data[-2, 2] - track_data[-1, 2]) / 2
    height_correction = (track_data[-2, 3] - track_data[-1, 3]) / 2
    displacement_corrected = displacement - np.array([width_correction, height_correction])

    # Calculation of the corrected distance between the two points
    distance = np.linalg.norm(displacement_corrected)

    return distance


def intersection_over_union_c(
    boxA: List[float],
    boxB: List[float],
) -> float:
    """The function calculates IoU: the degree of intersection of two Bboxes.

    Input parameters:
        boxA (List[float]): The coordinates of Bbox 1st object. [x1, y1, x2, y2]],
        boxB (List[float]): The coordinates of the Bbox 2nd object. [a1, b1, a2, b2]]

    Output parameters:
        IoU (float): The extent of intersection of two Bboxes
    """
    # Conversion of Bbox coordinates to format [x1, y1, x2, y2]
    def convert_to_corners(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]

    boxA = convert_to_corners(boxA)
    boxB = convert_to_corners(boxB)

    # Calculation of the area of the intersection rectangle
    interArea = intersection_rect_area(boxA, boxB)
    if interArea == 0:
        return 0
    # Calculation of the area of 1st Bbox
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    # Calculation of 2nd Bbox area
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # Calculation of IoU
    IoU = interArea / float(boxAArea + boxBArea - interArea)

    return IoU


def update_ema_velocity(track_values_c_xywh: List | Deque,
                        track_values_velocity: List | Deque,
                        track_values_ema: List | Deque,
                        correction: dict,
                        alpha: float = 0.5,
                        velocity_ema_start: int = 8):
    """
    Updates the exponential moving average (EMA) speed.

    Input parameters:
        track_values_c_xywh (np.ndarray): The parameters of the centre of the object,
        track_values_velocity (List): Speed values of the object,
        track_values_ema (List): The EMA values of the object,
        alpha (float): Coefficient of smoothing for EMA,
        correction (dict): Vocabulary with speed correction parameters,
        velocity_ema_start (int): A boufer to start calculating EMA.

    Output parameters:
        velocity_ema (float): Updated EMA. flat
    """
    # Transform the c_xywh value list to the NumPy array with the data type float16
    track_values_c_xywh = np.array(list(track_values_c_xywh), dtype=np.float16)
    # We extract only the center coordinates (x, y) of the object from track_values_c_xywh
    track_data_xy = track_values_c_xywh[:, :2]

    if correction['enabled']:
        len_track = len(track_values_velocity) if len(track_values_velocity) < int(correction['velocity_corr_history_len']) \
            else int(correction['velocity_corr_history_len'])    
        # Check whether data are sufficient to calculate EMA
        if len(track_values_velocity) < velocity_ema_start:
            return None
        else:
            # If data are insufficient, return the initial EMA value
            ema_first = np.mean(
                track_values_velocity) if track_values_velocity else 0  # Return the initial value if the data are insufficient

            IoU_list = []
            distance_list = []

            # Check whether an object is stationary
            displacement = track_data_xy[-1] - track_data_xy[-len_track]
            # Adjusting the movement of the centre of the object to change its size
            width_correction = (track_values_c_xywh[-len_track, 2] - track_values_c_xywh[-1, 2]) / 2
            height_correction = (track_values_c_xywh[-len_track, 3] - track_values_c_xywh[-1, 3]) / 2
            displacement_corrected = displacement - np.array([width_correction, height_correction])
            # Calculate the distance to the new position with adjustment
            distance = np.linalg.norm(displacement_corrected)

            # Speed variability analysis
            velocities = np.array(list(track_values_velocity), dtype=np.float16)
            velocity_differences = np.diff(velocities[-len_track:])
            # Definition of sharp speed changes where absolute difference value >=4
            abrupt_change = np.abs(velocity_differences) >= correction['abrupt_change_diff']
            # Count the number of such sudden changes
            abrupt_change_sum = np.sum(abrupt_change)
            # We extract the speeds that have been recognized by sharp changes
            abrupt_velocity_differences = velocity_differences[abrupt_change]
            # Analyzing multiple changes in the direction of speed
            multidirectional = np.sum(np.diff(np.sign(abrupt_velocity_differences)) != 0)

            # Analysis of directional variability
            direction_changes = 0
            previous_direction = None
            # Verification of the number of sharp changes in direction
            for i in range(1, len_track):
                # Calculation of the directional vector
                direction = track_data_xy[i] - track_data_xy[i - 1]
                # Direction-shift check
                if previous_direction is not None and np.dot(direction, previous_direction) < 1:
                    # Calculation of the number of directional changes
                    direction_changes += 1
                previous_direction = direction

                # Calculation of Intersection over Area (IoA) for the last bbox and previous
                IoU = intersection_over_union_c(track_values_c_xywh[-(i + 1)], track_values_c_xywh[-1])
                IoU_list.append(IoU)

            IoU_list = np.array(IoU_list)
            # Proportion of staff with IoU above the threshold
            IoU_small_share = np.mean(IoU_list[5:] > correction['IoU_thrd'])

            distance_list = np.array(distance_list)

            # The requirements of this Regulation shall be as follows:
            # If the object was not moving at a significant distance
            if distance < (len_track * correction['distance_thrd_hard']):
                return 0
            # If the object was moving at a moderate distance but often changed direction and was stable in more than half of the cases
            elif (abrupt_change_sum > int(len_track / correction['abrupt_change_sum']) or 
                  IoU_small_share >= correction['IoU_small_share'] or 
                  multidirectional >= int(len_track / correction['multidirectional'])
                  ) and distance < (len_track *  correction['distance_thrd_soft']):
                return 0
            
            # Calculation and updating of EMA
            # If no previous EMA value exists, use the initial value
            if track_values_ema[-1] is None:
                ema = ema_first
            else:
                ema = alpha * track_values_velocity[-1] + (1 - alpha) * track_values_ema[-1]
        return ema

    # EMA without correction
    else:
        if len(track_values_velocity) < 2:
            return track_values_velocity[
                0] if track_values_velocity else 0  # Return the initial value if the data are insufficient
        # Calculation of EMA
        return alpha * track_values_velocity[-1] + (1 - alpha) * track_values_ema[-1]


def get_track_direction_ema(
        cls: int,
        track_id: int,
        track_history: dict,
        track_history_len: int,
        class_names: dict,
        alpha: float = 0.3,
) -> None:
    """
    Records the history of the direction of the object based on the last points in the history of the tracking.

    Input parameters:
        cls (int): Object class,
        track_id (int): Unique Object Identifier,
        track_history (dict): A dictionary to store the history of each object,
        track_history_len (int): Maximum number of records stored in history,
        class_names (dict): Vocabulary with object class names,
        alpha (float): Coefficient of smoothing for EMA,
    """
    track_data_xyhw = track_history.get(f"{class_names[cls]}_{track_id}_c_xy", [])
    if len(track_data_xyhw) < 4:
        return None

    # Calculation of the direction of the object on the basis of the latest points
    points_array = np.array(list(track_data_xyhw)[-4:], dtype=np.float32)
    points_array_xy = points_array[:, :2]
    angle_2_x, _ = fit_line(points_array_xy)
    track_key_direction = f"{class_names[cls]}_{track_id}_direction"
    track_values_direction = track_history.get(track_key_direction, deque(maxlen=track_history_len))
    track_values_direction.append(angle_2_x)
    track_history[track_key_direction] = track_values_direction

    # Calculation and recording of the history of 10 values of exponential average
    track_key_dir_ema = f"{class_names[cls]}_{track_id}_dir_ema"
    track_values_ema = track_history.get(track_key_dir_ema, deque(maxlen=track_history_len))
    if len(track_values_direction) < 2:
        ema = track_values_direction[
            -1] if track_values_direction else 0  # Return the initial value if the data are insufficient
    else:
        ema = alpha * track_values_direction[-1] + (1 - alpha) * track_values_ema[-1]
    
    track_values_ema.append(ema)
    track_history[track_key_dir_ema] = track_values_ema
