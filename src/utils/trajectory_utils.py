from typing import List, Tuple
import numpy as np
import cv2


def calculate_person_trajectory(
    img: np.ndarray,
    track_p_c_xy_list: List[Tuple]
) -> Tuple:
    """The function calculates the trajectory of the human path.

    Args:
        img (np.ndarray): Reference image.
        track_p_c_xy_list (List[Tuple]): list of coordinates of the centre of the Bbox people.

    Returns:
        pt1, pt2 (Tuple): Coordinates of the human path. Example: (10, 20), (30, 40)).
        k (float): Human track inclination factor.

    Raises:
        ValueError: If the number of points in track_p_c_xy_list is less than 2.
    """
    if len(track_p_c_xy_list) < 2:
        raise ValueError("track_p_c_xy_list must contain at least two points")

    # Definition of coordinates (x1, y1), (x2, y2)
    x1, y1 = track_p_c_xy_list[-2]
    x2, y2 = track_p_c_xy_list[-1]

    # Calculation of the straight line and intersection coefficients with the line Oy
    k, b = np.polyfit([x1, x2], [y1, y2], 1)

    # Calculation of endpoints based on image dimensions
    if x2 > x1:
        x_end = img.shape[1]  # Right edge of the image
    else:
        x_end = 0  # Left edge of the image

    y_end = k * x_end + b  # Relevant y value at selected border x

    # Collect coordinates at points for convenient use
    pt1, pt2 = (int(x1), int(y1)), (int(x_end), int(y_end))

    return pt1, pt2, k


def calculate_object_trajectory(
    img: np.ndarray,
    track_c_xy_list: List[Tuple],
    radius: float,
) -> Tuple:
    """The function calculates the trajectory towards the movement of the object.

    Args:
        img (np.ndarray): Reference image.
        track_c_xy_list (List[Tuple]): List of coordinates of the centre of the Bbox object.
        radius (float): The radius of the circle to determine the final trajectory point of the object).

    Returns:
        pt1, pt2 (Tuple): The trajectory points of the object. Example: (10, 20), (30, 40)).
        k (float): object &apos; s trajectory inclination factor.

    Raises:
        ValueError: If the number of points in track_c_xy_list is less than 5.
    """
    if len(track_c_xy_list) < 5:
        raise ValueError("The track_p_c_xy_list must contain at least two points")

    # Definition of coordinates (x1, y1), (x2, y2)
    x1, y1 = track_c_xy_list[-5]
    x2, y2 = track_c_xy_list[-1]

    # Calculation of direction vector from (x1, y1) to (x2, y2)
    direction_vector = np.array([x2 - x1, y2 - y1])
    direction_norm = np.linalg.norm(direction_vector)

    # Normalization of direction vector
    if direction_norm != 0:
        direction_vector = direction_vector / direction_norm

    # Calculation of the final point in the direction of motion and radius
    x_end = x2 + direction_vector[0] * radius
    y_end = y2 + direction_vector[1] * radius

    # Limiting the end point within the image
    x_end = np.clip(x_end, 0, img.shape[1] - 1)
    y_end = np.clip(y_end, 0, img.shape[0] - 1)

    # Collect coordinates at points for convenient use
    pt1, pt2 = (int(x2), int(y2)), (int(x_end), int(y_end))

    # Calculation of the object &apos; s trajectory inclination factor
    k = (y_end - y2) / (x_end - x2)

    return pt1, pt2, k


def calculate_wagon_trajectory(
    mask: np.ndarray,
    box: List[float],
) -> Tuple:
    """The function calculates the trajectory of the wagon on the rail.

    Args:
        mask (np.ndarray): The mask(/polygon) of the rail on which the wagon moves.
        box (List[float]): Bbox rails.

    Returns:
        pt1, pt2 (Tuple): Co-ordinates of the track of the wagon on the rail. Example: (10, 20), (30, 40)).
        k (float): trajectories of the wagon.
    """
    # Determination of the points of the parallel straight range (masks))
    line_rail = cv2.fitLine(mask, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    x1, y1 = int(line_rail[0] * 100 + line_rail[2]), int(line_rail[1] * 100 + line_rail[3])
    x2, y2 = int(-line_rail[0] * 100 + line_rail[2]), int(-line_rail[1] * 100 + line_rail[3])
    x_min = box[0]
    x_max = box[2]

    # Selection of line coefficients for the construction of a straight line not only between 1st point and 2nd but longer
    coefs = np.polyfit([x1, x2], [y1, y2], 1)

    # Direct model
    model = np.poly1d(coefs)

    # Data generation for direct function
    x = np.linspace(x_min, x_max, 2)

    # Calculation of function values
    y = model(x)

    # Collect coordinates at points for convenient use
    pt1, pt2 = (int(x[0]), int(y[0])), (int(x[-1]), int(y[-1]))

    # trajectories of the wagon on the rail
    k = coefs[0]

    return pt1, pt2, k
