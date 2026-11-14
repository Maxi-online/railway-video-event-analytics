from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from src.services.tracker.tracker import TrackerLWT
from src.utils.draw_utils import plot_contour
from src.utils.geometry_utils import (
    fit_line, intersection_2polygons_area,
    euclidean_distance, closest_point_on_contour,
    intersection_over_area_mask
)


def approximate_rail_line(
    polygon: np.ndarray,
    img_width: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Approximize the rail line along the range and return the angular factor (k) and the points of the line.

    Args:
        polygon (np.ndarray): List of test site points [(x1, y1), (x2, y2)), ...].
        img_width (int): Image width (to determine the range x)).

    Returns:
        angle_2_x (float): The angle of the line direction relative to the x axis.
        x_range (np.ndarray): Massive x-coordinate of the amplified line.
        y_range (np.ndarray): Mass y-coordinate of the amplified line.
    """
    if len(polygon) > 2:
        points = np.array(polygon, dtype=np.float32)
        angle_2_x, poly_coeffs = fit_line(points)

        poly_func = np.poly1d(poly_coeffs)
        x_range = np.linspace(0, img_width - 1, img_width)
        y_range = poly_func(x_range)

        # Limit points within the image
        valid_idx = np.where((y_range >= 0) & (y_range < img_width))
        x_range = x_range[valid_idx]
        y_range = y_range[valid_idx]

        return angle_2_x, x_range, y_range


#==========================================================#
# Support functions for the logic of task_4_2 #
#==========================================================#

def get_moving_wagons_rails(
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    speed_threshold_lwt: float,
    thrld_dist_w_r: float,
    show_debug: bool,
    img: np.ndarray,
) -> Dict[int, int]:
    """
    The function defines moving wagons and their intersection with rails.
    
    Args:
        tracker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        results_r (Dict[str, np.ndarray]): Results of rail segmentation.
        speed_threshold_lwt (float): Threshold value of the speed of the wagon.
        thrld_dist_w_r (float): Proportion of the height of the wagon for determining the intersection with the rails.
        show_debug (bool): Debugging information flag.
        img (np.ndarray): Image for debugging information.

    Returns:
        wagons_rails_link (Dict[int, int]): dictionary linking rail and wagon indices.
    """
    # Railway-wagon link dictionary
    wagons_rails_link = {}
    
    # We're going through all the tracking results for the wagons.
    for index_lwt, (_, _, mask_lwt, cls_lwt, _, track_id_lwt) in enumerate(zip(*tracker_LWT.results.values())):
        # Check if the car's moving.
        if not is_wagon_moving(tracker_LWT, cls_lwt, track_id_lwt, speed_threshold_lwt):
            continue

        # Getting a bbox-oriented and wagon height
        o_bbox, height_lwt_or = get_obbox_and_wagon_height(mask_lwt)
        # Calculation of the displacement vector of the wagon
        displacement_vector = get_displacement_vector(tracker_LWT, cls_lwt, track_id_lwt)
        
        # Definition of intersections and cornering points of the wagon
        indexs_r_list, corner_points_list = get_intersecting_rails(mask_lwt, results_r, displacement_vector, o_bbox, height_lwt_or, thrld_dist_w_r)
        
        # If show_debug flag is installed, display debugging information
        if show_debug:
            draw_debug_info(img, corner_points_list, indexs_r_list, results_r)

        # Comparison of rail and wagon indices
        for index_r in indexs_r_list:
            wagons_rails_link[index_r] = index_lwt

    return wagons_rails_link


def get_wagons_rails(
    tracker_LWT: TrackerLWT,
    results_r: Dict[str, np.ndarray],
    thrld_dist_w_r: float,
    show_debug: bool,
    img: np.ndarray,
) -> Tuple[Dict, Dict]:
    """
    The function defines the wagons and their intersection with the rails.

    Args:
        tracker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        results_r (Dict[str, np.ndarray]): Results of rail segmentation.
        thrld_dist_w_r (float): Proportion of the height of the wagon for determining the intersection with the rails.
        show_debug (bool): Debugging information flag.
        img (np.ndarray): Image for debugging information.

    Returns:
        wagons_rails_link (Dict[int, int]): dictionary linking rail and wagon indices.
    """
    # Railway-wagon link dictionary
    wagons_rails_link = {}

    # Vocabulary to store the corner points of the wagon for each rail by its index
    indexR_cornerPoints = {}

    # We're going through all the tracking results for the wagons.
    for index_lwt, (_, _, mask_lwt, cls_lwt, _, track_id_lwt) in enumerate(zip(*tracker_LWT.results.values())):

        # Getting a bbox-oriented and wagon height
        o_bbox, height_lwt_or = get_obbox_and_wagon_height(mask_lwt)
        # Calculation of the displacement vector of the wagon
        displacement_vector = get_displacement_vector(tracker_LWT, cls_lwt, track_id_lwt)

        # Definition of intersections and cornering points of the wagon
        indexs_r_list, corner_points_list = get_intersecting_rails(mask_lwt, results_r, displacement_vector, o_bbox,
                                                                   height_lwt_or, thrld_dist_w_r)

        # If show_debug flag is installed, display debugging information
        if show_debug:
            draw_debug_info(img, corner_points_list, indexs_r_list, results_r)

        # Comparison of rail and wagon indices
        for index_r in indexs_r_list:
            wagons_rails_link[index_r] = index_lwt

        # Comparison of rail and corner points of the wagon
        for i in range(len(indexs_r_list)):
            # Rail index
            index_r = indexs_r_list[i]
            # Co-ordinates of the corner points of the rail wagon
            corner_points = corner_points_list[i]
            indexR_cornerPoints[index_r] = corner_points

    return wagons_rails_link, indexR_cornerPoints


def is_wagon_moving(
    tracker_LWT: TrackerLWT,
    cls_lwt: int,
    track_id_lwt: int,
    move_velocity_lwt: float,
) -> bool:
    """
    The function determines whether the wagon is moving on the basis of its current speed and set threshold.

    Args:
        tracker_LWT (TrackerLWT): A copy of the tracker class for wagons, locomotives and tanks.
        cls_lwt (int): Wagon/locomotive/tank class.
        track_id_lwt (int): Wagon track identifier.
        move_velocity_lwt (float): Threshold value of the speed of the wagon.

    Returns:
        bool: Returns True if the wagon is moving and False is otherwise.
    """
    # Form a key to access the train speed data in track history
    track_key_lwt_vel_ema = f"{tracker_LWT.class_names[cls_lwt]}_{track_id_lwt}_ema"
    
    # Take a list of the speed of the wagon from the history of tracking
    track_lwt_ema_list = list(tracker_LWT.track_history.get(track_key_lwt_vel_ema, []))
    
    # See if there's at least one speed record on the list
    if len(track_lwt_ema_list) > 0:
        # Take the last recorded speed.
        current_lwt_vel = track_lwt_ema_list[-1]
        # If speed exists and exceeds the threshold, let's return True
        return current_lwt_vel and current_lwt_vel > move_velocity_lwt
    else:
        # If there is no speed or less than the threshold value, let's return False
        return False


def get_obbox_and_wagon_height(mask_lwt: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    The function is found to be a oriented limiting rectangle (OBB) and a height of the masked wagon.

    Args:
        mask_lwt (np.ndarray): The mask of the wagon as a binary array.

    Returns:
        box_oriented_lwt (np.ndarray): Co-ordinates of the angles of the orientation rectangle containing the wagon.
        min(w_o, h_o) (float): Minimum side of the orientation rectangle representing the height of the wagon.
    """
    # Create a oriented restrictive rectangle around the car mask
    rect = cv2.minAreaRect(mask_lwt)
    # Extraction of the rectangular parameters
    (_, _), (w_o, h_o), _ = rect
    # Transforming the rectangle to the coordinates of its angles
    box_oriented_lwt = cv2.boxPoints(rect)
    # Transforming angles to a whole type
    box_oriented_lwt = np.int8(box_oriented_lwt)
    # Return the coordinates of the angles of the directional rectangle and the minimum side as the height of the wagon
    return box_oriented_lwt, min(w_o, h_o)


def get_displacement_vector(
    tracker_LWT: TrackerLWT,
    cls_lwt: int,
    track_id_lwt: int,
) -> np.ndarray:
    """
    Function to calculate the displacement vector of the wagon based on the history of its track.

    Args:
        tracker_LWT (TrackerLWT): Tracker class copy for wagons.
        cls_lwt (int): Class of wagon, locomotive or tank.
        track_id_lwt (int): Wagon track identifier.

    Returns:
        np.ndarray: Wagon displacement vector.
    """
    # Form a key to access historical data on the centre of the wagon
    track_key_lwt_center_xy = f"{tracker_LWT.class_names[cls_lwt]}_{track_id_lwt}_c_xy"
    
    # Get a list of the coordinates of the center of the wagon from the tracking history
    track_lwt_c_xy_list = np.array(list(tracker_LWT.track_history.get(track_key_lwt_center_xy, [])))
    
    # If the tracking history is shorter than the predetermined value of track_history_len
    if len(track_lwt_c_xy_list) < tracker_LWT.track_history_len:
        # Take the first point as the starting point.
        start_point = track_lwt_c_xy_list[0]
    else:
        # Otherwise, take the point removed on track_history_len back
        start_point = track_lwt_c_xy_list[-tracker_LWT.track_history_len]
    
    # Return the displacement vector as the difference between the initial and the last point
    return start_point - track_lwt_c_xy_list[-1]


def get_intersecting_rails(
    mask_lwt: np.ndarray,
    results_r: Dict[str, np.ndarray],
    displacement_vector: np.ndarray,
    o_bbox: np.ndarray,
    height_lwt_or: float,
    thrld_dist_w_r: float,
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Function for the determination of intersecting rails with a masked wagon and the position of the wagon.

    Args:
        mask_lwt (np.ndarray): The mask of the wagon as a bin array representing the wagon.
        results_r (Dict[str, np.ndarray]): A dictionary with rail segmentation results containing rail masks.
        displacement_vector (np.ndarray): Wagon displacement vector based on its movement.
        o_bbox (np.ndarray): Co-ordinates of the angles of the orientation limiting rectangle (OBB) of the wagon.
        height_lwt_or (float): The height of the wagon as defined by the minimum side of the orientation rectangle.
        thrld_dist_w_r (float): Pipe value for determining the crossings of the wagon and railways depending on their distance.

    Returns:
        Tuple[List[int], List[np.ndarray]: List of index of rail masks crossing with the wagon, and list of corner points of the wagon.
    """
    indexs_r_list = []  # List for storage of rail indices intersecting with the wagon
    corner_points_list = []  # List for storage of corner points of the wagon

    # Overtaking all rail masks and their indices
    for index_r, r_mask in enumerate(results_r['masks']):
        # Calculation of the intersection area and centre of the rail and wagon intersections
        intesect_area, intesect_centroid = intersection_2polygons_area(r_mask, mask_lwt)
        # If the intersection area is greater than zero, add rails to the intersection list
        if intesect_area > 0:
            # Obtaining the corner points of the wagon based on the mask and the displacement vector
            corner_points = get_corner_points(mask_lwt, o_bbox, displacement_vector)
            corner_points_list.append(corner_points)
            # Check whether the corner points of the wagon are within the distance thrld_dist_w_r from the centre of the intersection
            if any(euclidean_distance(intesect_centroid, point) < height_lwt_or * thrld_dist_w_r for point in corner_points):
                indexs_r_list.append(index_r)  # Adding a rail index to the list of intersecting countries

    return indexs_r_list, corner_points_list


def get_corner_points(
    mask_lwt: np.ndarray,
    o_bbox: np.ndarray,
    displacement_vector: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Function for determining the cornering points of a wagon near the rails based on it
    Current displacement and directional bbox.

    Args:
        mask_lwt (np.ndarray): The mask of the wagon as a binary array.
        o_bbox (np.ndarray): orientation restricted rectangle (OBB) of the wagon.
        displacement_vector (np.ndarray): Wagon displacement vector.

    Returns:
        List[Tuple[int,int]]: List of coordinates of corner points of the wagon near the rails.
    """
    # If the wagon moves left down or right up
    if (displacement_vector[1] < 0 < displacement_vector[0]) or (displacement_vector[0] < 0 < displacement_vector[1]):
        # Definition of the lower left and upper right corner points of the wagon
        lower_left_box = max(o_bbox, key=lambda point: (point[1], -point[0]))
        upper_right_box = max(o_bbox, key=lambda point: (point[0], point[1]))
        return [
            # To improve accuracy, the nearest points on the wagon contour to the selected angular points shall be determined
            closest_point_on_contour(mask_lwt, lower_left_box),
            closest_point_on_contour(mask_lwt, upper_right_box),
        ]
    # If the wagon moves to the right down or to the left up
    elif (displacement_vector[0] > 0 and displacement_vector[1] > 0) or (displacement_vector[0] < 0 and displacement_vector[1] < 0):
        # Definition of the lower right and upper left corner points of the wagon
        lower_right_box = max(o_bbox, key=lambda point: (point[1], point[0]))
        upper_left_corner = min(mask_lwt, key=lambda point: (point[0], -point[1]))
        return [
            # To improve accuracy, the nearest points on the wagon contour to the selected angular points shall be determined
            closest_point_on_contour(mask_lwt, lower_right_box),
            closest_point_on_contour(mask_lwt, upper_left_corner),
        ]
    # If the wagon moves left or right
    elif displacement_vector[1] == 0:
        return [
            # Definition of the extreme right and extreme left point of the wagon
            max(mask_lwt, key=lambda point: (point[0], point[1])),
            min(mask_lwt, key=lambda point: (point[0], -point[1])),
        ]
    # If the wagon moves up or down
    else:
        return [
            # Determination of the upper and lower point of the wagon
            max(mask_lwt, key=lambda point: point[1]),
            min(mask_lwt, key=lambda point: point[1]),
        ]


def draw_debug_info(
    img: np.ndarray,
    corner_points_list: List[np.ndarray],
    indexs_r_list: List[int],
    results_r: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    The function displays debugging information related to the movement of wagons and their intersection with rails on the frame.

    Args:
        img (np.ndarray): Image to which debugging information will be applied.
        corner_points_list (List[np.ndarray]): List of corners of all wagons crossing rails.
        indexs_r_list (List[int]): List of rail indices intersecting with wagons.
        results_r (Dict[str, np.ndarray]): rail segmentation results containing rail masks.

    Returns:
        img (np.ndarray): Image with debugged information.
    """
    # Displaying the corner points of the wagons in the image
    for points_pair in corner_points_list:
        for point in points_pair:
            # Draw a circle at each corner point
            cv2.circle(img, tuple(point), 10, (255, 255, 0), 5)

    # Image of intersecting rails in the image
    for index_r in indexs_r_list:
        # Drawing the lines on the image
        img = plot_contour(img, results_r['masks'][index_r], color=[0, 255, 255], thickness=3)
    
    # Return the image with debugging information
    return img


def get_persons_feet(
    results_tracking_p: Dict[str, np.ndarray],
    feet_ratio: float,
) -> np.ndarray:
    """
    Function to generate human foot coordinates based on track results.

    Args:
        results_tracking_p (Dict[str, np.ndarray]): Tracking results containing coordinates of limiting frame (bbox) people.
        feet_ratio (float): Coefficient used to calculate the foot position relative to the lower bbox.

    Returns:
        np.ndarray: Massive of human leg coordinates. If the track results are not available, the empty array returns.
    """
    # Checking to see if there's any track results.
    if len(results_tracking_p["boxes_xyxy"]) > 0:
        # Copy a set of bbox people's coordinates.
        persons_feet = np.array(results_tracking_p['boxes_xyxy']).copy()
        # Change the upper limit of bbox to get the foot coordinates
        persons_feet[:, 1] = persons_feet[:, 3] * feet_ratio
        return persons_feet
    # If the track results are missing, return the empty array
    return np.array([])


def is_person_crossing_rails(
    box_feet_p: np.ndarray,
    wagons_rails_link: Dict[int, int],
    results_r: Dict[str, np.ndarray],
    threshold_ioa: float,
) -> Tuple[Any, bool]:
    """
    Function to determine whether a person &apos; s feet cross rails under moving wagons.

    Args:
        box_feet_p (np.ndarray): Human leg coordinates as [x1, y1, x2, y2].
        wagons_rails_link (Dict[int, int]): dictionary linking rail and wagon indices.
        results_r (Dict[str, np.ndarray]): rail segmentation results containing rail masks.
        threshold_ioa (float): IoA (Intersection over Area) porous value to determine the crossings of human legs and rails.

    Returns:
        bool: Returns True if a man's legs cross the rails, and False otherwise.
    """
    # We're going to take all the rail indices associated with moving wagons
    for index_r in wagons_rails_link.keys():
        # Get a rail mask on the index.
        mask_r = results_r["masks"][index_r]
        # Checking whether a person's legs overlap with rails using IoA
        if intersection_over_area_mask(box_feet_p, mask_r) > threshold_ioa:
            # If there's a crossing, we'll return True.
            return index_r, True
    # If there are no intersections, let's return False.
    return None, False
