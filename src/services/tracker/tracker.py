from collections import defaultdict
from time import time
from typing import List, Dict, Optional

import cv2
import numpy as np
from ultralytics.utils.ops import scale_image

# === our utils ===
from src.utils.preprocess import preprocess_frame
from src.utils.nms_utils import soft_nms

from src.services.tracker.tracker_utils import (
    get_track_person_hw, get_track_velocity_ema, get_track_direction_ema
)
from src.utils.geometry_utils import transform_mask_2_max_polygon


class Tracker:
    def __init__(self, model):
        # Definition of the model
        self.model = model

        # Initialization of the time of appearance
        self.new_frame_time = time()
        self.prev_frame_time = 0

        # Initialization of Tracking History
        self.track_history = defaultdict(lambda: [])

        # A dictionary to store the time of the last update of each track
        self.last_update_time = {}

        # Parameter to determine how long a track may be inactive before removal
        self.track_timeout = 120 # in seconds

    def predict_on_image_track(
        self,
        img: np.ndarray,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        tracker: str = "botsort.yaml",
    ) -> Dict:
        """
        The function performs image processing using the YOLO model and monitors the movement of objects.

        Args:
            img (np.ndarray): np.ndarray image for analysis.
            iou (float): The threshold for calculating the intersection of objects ( default 0.5)).
            conf (float): The threshold of confidence in an object's detective (default 0.5)).
            imgsz (int): Image size for processing ( default 640)).
            verbose (bool): Inclusion of detailed output mode (false default)).
            tracker (str): Tracker configuration used ( default "botsort.yaml"").

        Returns:
            Dict: A dictionary with image processing results.
        """
        # -----Pre-processing ----
        img_proc = preprocess_frame(img)

        # -----Inference & Track -----
        result = self.model.track(img_proc, iou=iou, conf=conf, imgsz=imgsz,
                                  verbose=verbose, tracker=tracker, persist=True)[0]

        # Getting the results of a detective
        cls = result.boxes.cls.cpu().numpy()
        self.class_names = result.names
        probs = result.boxes.conf.cpu().numpy()
        # Bbox coordinates for tracking are returned as [xc, yc, w, h] - coordinates of the centre of Bbox,
        # Its breadth and height. Done to improve the accuracy of the calculation of the speed of movement of the object
        boxes = result.boxes.xywh.int().cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        track_ids = []  # default

        # ----- Soft-NMS to remove duplicates without losing Recall -----
        if len(boxes_xyxy) > 0:
            keep = soft_nms(boxes_xyxy.copy(), probs.copy(), cls.copy(), iou_thr=0.5, sigma=0.5)
            boxes = boxes[keep]
            boxes_xyxy = boxes_xyxy[keep]
            cls = cls[keep]
            probs = probs[keep]
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().numpy()[keep]
            else:
                track_ids = []
        # If there's no boxing, we're missing the keep.

        # Retaining results
        results = {'boxes': boxes, 'boxes_xyxy': boxes_xyxy, 'masks': [], 'cls': cls, 'probs': probs,
                   'track_ids': track_ids}

        try:
            # Acquisition of object masks
            masks = result.masks.data.cpu().numpy()
            masks = np.moveaxis(masks, 0, -1)
            masks = scale_image(masks, result.orig_shape)
            masks = np.moveaxis(masks, -1, 0)
            masks = [transform_mask_2_max_polygon(mask) for mask in masks]
            results['masks'] = masks
        except:
            # Processing of cases where objects masks are not available
            results['masks'] = [None] * len(results['boxes'])

        # Updating time tags for tracking
        self.prev_frame_time = self.new_frame_time
        self.new_frame_time = time()

        # Update the time of last update for each track
        current_time = time()
        for track_id in track_ids:
            self.last_update_time[track_id] = current_time

        # Check and remove obsolete tracks
        self.remove_stale_tracks()

        return results

    def remove_stale_tracks(self):
        'Removes tracks that have not been updated for some time.'
        current_time = time()
        stale_tracks = []
        for track_id, last_update in self.last_update_time.items():
            if current_time - last_update > self.track_timeout:
                stale_tracks.append(track_id)

        for track_id in stale_tracks:
            del self.last_update_time[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
            # print(f"Track with ID {track_id} was removed due to lack of updates.")

    def set_track_timeout(self, timeout):
        'Sets waiting times to remove inactive tracks.'
        self.track_timeout = timeout


class TrackerPJ(Tracker):
    def __init__(self, model):
        super().__init__(model)
        # Initialization of dictionaries with results
        self.results_persons = dict()
        self.results_jackets = dict()

        # Vocabulary to store data on the human foot and the moving wagon
        self.feet_wagon_intersections = {}


    def predict_on_image_track(
        self,
        img: np.ndarray,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        tracker: str = "botsort.yaml",
        track_history_len: int = 10,
        velocity_ema_start: int = 8,
        velocity_ema_alpha: float = 0.9,
        dir_ema_alpha: float = 0.5,
        show_velocity: bool = False,
        color_velocity: List = (0, 171, 255),
        thick_velocity: int = 2,
        fontscale_velocity: float = 1.2,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
    ) -> np.ndarray:
        """
        The function detects objects in the image and monitors their movement.

        Args:
            img (np.ndarray): Incoming image.
            iou (float, optional): Cut-off threshold for suppression of not maximums. By default 0.5.
            conf (float, optional): The confidence threshold to filter predictions. By default 0.5.
            imgsz (int, optional): Image size for scaling. By default 640.
            verbose (bool, optional): Flag to output detailed information. False default.
            tracker (str, optional): Path to tracker configuration file. By default "botsort.yaml".
            track_history_len (int, optional): The length of track history. By default 10.
            velocity_ema_start (int, optional): The frame number from which the speed calculation begins. By default 8.
            velocity_ema_alpha (float, optional): Coefficient of smoothing of exponentially sliding average for speed. By default 0.9.
            dir_ema_alpha (float, optional): Coefficient of smoothing of exponentially sliding mean for direction. By default 0.5.
            show_velocity (bool, optional): Flag to display speed in the image. False default.
            color_velocity (List, optional): Color for image speed display. By default (0, 171, 255)).
            thick_velocity (int, optional): The thickness of the speed line in the image. By default 2.
            fontscale_velocity (float, optional): The size of the font to display the speed in the image. By default 1.2.
            font (int, optional): Speed display font. By default cv2.FONT_HERSHEY_SIMPLEX.

        Returns:
            img (np.ndarray): Image with monitored objects.
        """
        results = super().predict_on_image_track(
            img, iou, conf, imgsz, verbose, tracker
            )

        self.results_persons = {'boxes': [], 'boxes_xyxy': [], 'masks': [], 'cls': [], 'probs': [], 'track_ids': []}
        self.results_jackets = {'boxes': [], 'boxes_xyxy': [], 'masks': [], 'cls': [], 'probs': [], 'track_ids': []}

        for box, box_xyxyx, mask, cls, prob, track_id in zip(*results.values()):
            if cls == 0:
                self.results_persons['boxes'].append(box)
                self.results_persons['boxes_xyxy'].append(box_xyxyx)
                self.results_persons['masks'].append(mask)
                self.results_persons['cls'].append(cls)
                self.results_persons['probs'].append(prob)
                self.results_persons['track_ids'].append(track_id)

                get_track_person_hw(self.track_history, track_id, track_history_len, box)
                get_track_velocity_ema(img, box, cls, track_id,
                                       self.track_history, track_history_len, self.class_names,
                                       self.new_frame_time, self.prev_frame_time, velocity_ema_alpha,
                                       show_velocity, color_velocity, thick_velocity,
                                       fontscale_velocity, font, correction={'enabled': False},
                                       velocity_ema_start=velocity_ema_start)
                get_track_direction_ema(cls, track_id, self.track_history,
                                        track_history_len,
                                        self.class_names, dir_ema_alpha)
            else:
                self.results_jackets['boxes'].append(box)
                self.results_jackets['boxes_xyxy'].append(box_xyxyx)
                self.results_jackets['masks'].append(mask)
                self.results_jackets['cls'].append(cls)
                self.results_jackets['probs'].append(prob)
                self.results_jackets['track_ids'].append(track_id)

        return img


class TrackerLWT(Tracker):
    def __init__(self, model):
        super().__init__(model)
        # Initialization of the dictionary with results
        self.results = dict()
        self.velocity_ema_start = None

        # Trick IDs history length
        self.track_history_len = None


    def predict_on_image_track(
        self,
        img: np.ndarray,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        tracker: str = "botsort.yaml",
        velocity_correction: Optional[dict] = None,
        track_history_len: int = 10,
        velocity_ema_start: int = 8,
        velocity_ema_alpha: float = 0.9,
        dir_ema_alpha: float = 0.5,
        show_velocity: bool = False,
        color_velocity: List = (0, 171, 255),
        thick_velocity: int = 2,
        fontscale_velocity: float = 1.2,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
        """
        Project objects in the image and track their movement.

        Args:
            img (np.ndarray): Incoming image.
            iou (float, optional): Cut-off threshold for suppression of not maximums. By default 0.5.
            conf (float, optional): The confidence threshold to filter predictions. By default 0.5.
            imgsz (int, optional): Image size for scaling. By default 640.
            verbose (bool, optional): Flag to output detailed information. False default.
            tracker (str, optional): Path to tracker configuration file. By default "botsort.yaml".
            velocity_correction (dict, optional): Speed correction parameters. default {'enable': False}.
            track_history_len (int, optional): The length of track history. By default 10.
            velocity_ema_start (int, optional): The frame number from which the speed calculation begins. By default 8.
            velocity_ema_alpha (float, optional): Coefficient of smoothing of exponentially sliding average for speed. By default 0.9.
            dir_ema_alpha (float, optional): Coefficient of smoothing of exponentially sliding mean for direction. By default 0.5.
            show_velocity (bool, optional): Flag to display speed in the image. False default.
            color_velocity (List, optional): Color for image speed display. By default (0, 171, 255)).
            thick_velocity (int, optional): The thickness of the speed line in the image. By default 2.
            fontscale_velocity (float, optional): The size of the font to display the speed in the image. By default 1.2.
            font (int, optional): Speed display font. By default cv2.FONT_HERSHEY_SIMPLEX.

        Returns:
            img (np.ndarray): Image with monitored objects.
        """
        if velocity_correction is None:
            velocity_correction = {'enable': False}

        self.results = super().predict_on_image_track(
            img, iou, conf, imgsz, verbose, tracker
            )

        self.track_history_len = track_history_len

        for box, _, _, cls, _, track_id in zip(*self.results.values()):
            get_track_velocity_ema(img, box, cls, track_id,
                                self.track_history, track_history_len, self.class_names,
                                self.new_frame_time, self.prev_frame_time, velocity_ema_alpha,
                                show_velocity, color_velocity, thick_velocity,
                                fontscale_velocity, font, correction=velocity_correction, velocity_ema_start=velocity_ema_start)
            get_track_direction_ema(cls, track_id, self.track_history,
                                    track_history_len,
                                    self.class_names, dir_ema_alpha)

        return img
