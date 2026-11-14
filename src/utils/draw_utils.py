import warnings
from typing import List, Tuple

import cv2
import numpy as np

# Ignoring Warning Numpy
warnings.simplefilter('ignore', np.RankWarning)


def plot_bbox(image: np.ndarray, bbox: List[float], color: Tuple[int], thickness: int=1) -> np.ndarray:
    """
    The function builds Bbox around a detected object on a frame.

    Input parameters:
        frame (np.ndarray): The frame on which the BBox object will be delivered,
        bbox (np.ndarray): Bbox coordinates in format [x0, y0, x1, y1],
        color (Tuple[int]): Bbox color,
        thickness (int): Bbox line thickness.

    Output parameters:
        image_bbox (np.ndarray): Image with BBox object.
    """
    image_bbox = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                               color=color, thickness=thickness)   
    return image_bbox


def plot_binary_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int], alpha: float=0.5) -> np.ndarray:
    """
    The function puts the mask on the original image and returns the result as np.ndarray.

    Input parameters:
        frame (np.ndarray): The frame on which the mask will be applied,
        mask (np.ndarray): Segmentation mask,
        color (Tuple[int]): Mask colour,
        alpha (float): The transparency of the mask colour.
    Output parameters:
        image_combined (np.ndarray): Image with mask on site.
    """
    if mask is not None:
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    else:
        return image
    return image_combined


def plot_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int],
    thickness: int,
) -> np.ndarray:
    """
    The function puts the mask contour on the original image and returns the result as np.ndarray.
    
    Args:
        image (np.ndarray): The frame on which the mask will be drawn.
        mask (np.ndarray): Segmentation mask.
        color (Tuple[int]): Mask contour color.
        thickness (int): Thickness of the outline of the mask.

    Returns:
        frame (np.ndarray): Image with mask contour on object.
    """

    if mask.any():
        image_contour = cv2.drawContours(image, [mask], -1, color, thickness=thickness)
    else:
        # Return of the original image if no mask
        return image

    return image_contour


def plot_results(
    results: dict,
    class_names: dict,
    frame: np.ndarray,
    task: str,
    font: int,
    show_mask: bool,
    show_class: bool,
    colors_classes: dict,
    thickness: int,
    alpha: float,
    fontscale: float,
    is_tracking: bool = False,
) -> np.ndarray:
    """The function uses Bboxes and Masks to frame and returns the result as np.ndarray.
    Supports both ordinary detective and tracking, as well as special case task="world".

    Args:
        results (dict): Model prediction results.
        class_names (dict): A dictionary with class names.
        frame (np.ndarray): Visualization frame.
        task (str): Type of task ("PJ", "LWT", "R", "world").
        font (int): Text font (cv2.FONT_...).
        show_mask (bool): Draw the mask or only the contour.
        show_class (bool): Show class and probability.
        colors_classes (dict): Colors for classes.
        thickness (int): Line thickness.
        alpha (float): Transparent mask (0-1).
        fontscale (float): Text size.
        is_tracking (bool): True if results contain tracking data (addition fields)).

    Returns:
        np.ndarray: A set of BBoxes and Masks.
    """
    for items in zip(*results.values()):
        if is_tracking:
            _, bbox, mask, cls, prob, _ = items
        else:
            bbox, mask, cls, prob = items

        label = class_names[int(cls)]

        # Select color according to task
        color_key = "foreign_objects" if task == "world" else label
        color = colors_classes[color_key]

        # BBox or Mask/Contour drawing
        if task == "PJ":
            frame = plot_bbox(frame, bbox, color=color, thickness=thickness)
        elif task != "world" or is_tracking:  # For a "world" without a tracking masks don't draw
            if show_mask:
                frame = plot_binary_mask(frame, mask, color=color, alpha=alpha)
            else:
                frame = plot_contour(frame, mask, color=color, thickness=thickness)

        # Text drawing (class + probability)
        if show_class and (task != "world" or is_tracking):
            cv2.putText(
                frame,
                f"{label} {prob:.2f}",
                (int(bbox[0]), int(bbox[1])),
                fontFace=font,
                fontScale=fontscale,
                color=color,
                thickness=thickness,
            )

    return frame


def plot_results_unet(
    results: dict,
    class_names: dict,
    img: np.ndarray,
    show_mask: bool,
    show_class: bool,
    color: list,
    thickness: int,
    fontscale: float,
    alpha: float,
    font: int,
) -> np.ndarray:
    """
    A function for drawing the contours of objects in the image.

    Args:
        results (dict): Contains contours ('contours') limiting the scope ('boxes') and ID of classes ('cls')'),
        class_names (dict): Vocabulary with object class names,
        img (np.ndarray): Reference image,
        show_mask (bool): If True, display the colour mask on the image,
        show_class (bool): If True, display the class name on the image,
        color (Tuple[int]): Color of the shape of objects in the image,
        thickness (int): Thickness of the contour of objects in the image,
        fontscale (float): The extent of the contour of objects in the image,
        alpha (float): Transparent colour mask,
        font (int): Text display font.

    Returns:
        (np.ndarray): Image with sketches and possibly object classes.
    """
    if results['masks'] is not None:
        for contour in results['masks']:
            if contour.any():
                if show_mask:
                    img = plot_binary_mask(img, contour, color=color, alpha=alpha)
                else:
                    img = cv2.drawContours(img, [contour], -1, color, thickness)
                # Name of class
                if show_class:
                    for contour, cls in zip(results['masks'],results['cls']):
                        label = class_names[int(cls)]
                        min_y_index = np.argmin(contour[:, 1])
                        min_y_x = contour[min_y_index, 0]
                        min_y_y = contour[min_y_index, 1]
                        label = class_names[int(cls)]
                        cv2.putText(img, label, (min_y_x, min_y_y),
                                    fontFace=font, fontScale=fontscale, color=color,
                                    thickness=thickness)
        return img


def plot_fps(
    frame: np.ndarray,
    frame_fps: float,
    avg_fps: float,
    color: Tuple[int],
    font,
):
    """The function gives the number of frames per second (FPS) at the top left corner of the frame.

    Args:
        frame (np.ndarray): Reference image.
        frame_fps (float): Current number of frames per second.
        avg_fps (float): Average number of frames per second.
        color (Tuple[int]): FPS text display color.
        font: Text font.
    """
    cv2.putText(frame, f"Frame FPS: {frame_fps:.1f}", (7, 70), fontFace=font,
                fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA)
    if avg_fps > 0:
        cv2.putText(frame, f"Average FPS: {avg_fps:.1f}", (7, 120), fontFace=font,
                    fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA)
    return


def plot_object_trajectory(
    img: np.ndarray,
    pt1: Tuple[int],
    pt2: Tuple[int],
    color: Tuple[int],
    thickness: int,
):
    """The function uses an object &apos; s trajectory on the frame.

    Args:
        img (np.ndarray): Reference image.
        pt1 (Tuple[int]): Co-ordinates of the starting point of the trajectory of the object.
        pt2 (Tuple[int]): Coordinates of the end of the course of an object.
        color (Tuple[int]): BGR track color.
        thickness (int): Thickness of the trajectory line of an object.
    """
    cv2.line(img, pt1, pt2, color, thickness)
    return


def plot_danger_zone(
    img: np.ndarray,
    danger_zone_cords: np.ndarray,
    color: Tuple[int],
    thickness: int
) -> np.ndarray:
    """The function is applied to the frame of a dangerous area (calibre or polygon)).

    Args:
        img (np.ndarray): Reference image.
        danger_zone_cords (np.ndarray): Coordinates of the dangerous area.
        color (Tuple[int]): Color of the dangerous area.
        thickness (int): Thickness of the line of the dangerous area figure.

    Returns:
        img (np.ndarray): A frame with a dangerous area.
    """
    # If a dangerous area is a rectangle
    if danger_zone_cords.ndim == 1:
        img = cv2.rectangle(img, (danger_zone_cords[0], danger_zone_cords[1]), (danger_zone_cords[2], danger_zone_cords[3]), color, thickness)
    else:
        # If the hazardous zone is a polygon
        img = cv2.polylines(img, [danger_zone_cords], True, color, thickness)

    return img


def plot_line_from_point(
    img: np.ndarray,
    center_point: Tuple[int, int],
    angle: float,
    color: Tuple[int, int, int],
    thickness: int,
    radius: int = 400,
) -> np.ndarray:
    """
    A function to draw a line from a given point with a given radius and angle relative to the x axis.

    Args:
        img (np.ndarray): Reference image.
        center_point (Tuple[int, int]): Initial point of the line (x, y)).
        angle (float): An angle relative to the x axis in degrees.
        color (Tuple[int, int, int]): Line color (BGR).
        thickness (int): Line thickness.
        radius (int): Line radius.

    Returns:
        img (np.ndarray): Image with line drawn.
    """
    # Conversion of angle from degrees to radians
    angle_rad = np.deg2rad(angle)

    # Calculation of half the radius
    half_radius = radius / 2

    # Calculation of the initial and final points of the line using half of the radius and angle
    start_point = (
        int(center_point[0] - half_radius * np.cos(angle_rad)),
        int(center_point[1] - half_radius * np.sin(angle_rad))
    )

    end_point = (
        int(center_point[0] + half_radius * np.cos(angle_rad)),
        int(center_point[1] + half_radius * np.sin(angle_rad))
    )

    # Limiting the coordinates of the initial and final points within the image
    start_point = (
        np.clip(start_point[0], 0, img.shape[1] - 1),
        np.clip(start_point[1], 0, img.shape[0] - 1)
    )

    end_point = (
        np.clip(end_point[0], 0, img.shape[1] - 1),
        np.clip(end_point[1], 0, img.shape[0] - 1)
    )

    # Drawing the line in the image
    img = cv2.line(img, start_point, end_point, color, thickness)

    return img


def plot_approximated_rail_line(
    img: np.ndarray,
    x_range: np.ndarray,
    y_range: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    """
    Draws an ancupunctured rail line on the image.

    Args:
        img (np.ndarray): Reference image.
        x_range (np.ndarray): X-coordinates of the line.
        y_range (np.ndarray): Y-coordinates of the line.
        color (Tuple[int, int, int]): BGR line color.
        thickness (int): Line thickness.

    Returns:
        img (np.ndarray): Image with line drawn.
    """
    if len(x_range) < 2 or len(y_range) < 2:
        return img

    for i in range(len(x_range) - 1):
        img = cv2.line(img, (x_range[i], y_range[i]), (x_range[i + 1], y_range[i + 1]), color, thickness)

    return img
