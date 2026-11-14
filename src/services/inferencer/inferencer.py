from typing import Tuple, List

import cv2
import numpy as np
from ultralytics.utils.ops import scale_image

from src.utils.geometry_utils import transform_mask_2_max_polygon
from src.services.inferencer.inferencer_utils import (
    create_binary_mask, sigmoid
)


class Inferencer:
    def __init__(self, model):
        # Definition of models
        self.model = model


    @staticmethod
    def onnx_preprocessing(
        image: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Preparation of an image for the release of the ONNX Runtime model.

        Args:
            image (np.ndarray): Image in numpy format.
            image_size (Tuple[int, int]): The desired image size after a change of size.

        Returns:
            (np.ndarray): Processed image ready for model output.
        """
        # Change in image size
        image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)
        # Normalization of the image
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0  # Normalization of Imagenet
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0   # Normalization of Imagenet
        denominator = np.reciprocal(std, dtype=np.float32)
        image = image.astype(np.float32)
        image -= mean
        image *= denominator
        # Translating image channels for compatibility with ONNX
        image = image.transpose((2, 0, 1))[None]
        return image


    def _predict_on_image_unet(
        self,
        img: np.ndarray,
        area_ratio: float,
        threshold: float = 0.4,
        imgsz_unet: Tuple[int, int] = (384, 384),
    ) -> Tuple:
        """
        Feature for predicting the object mask on the image using the UNet model and extract
        Sounding boxes according to the designated floor space.

        Args:
            img (np.ndarray): np.ndarray - Reference image in numpy narray format.
            area_ratio (float): boundary filtering threshold relative to the area of the image.
            threshold (float), optional (default=0.4): Threshold to binarize the predicted mask.
            imgsz_unet (Tuple[int, int]), optional (default=(384, 384)): Target image size for prediction.

        Returns:
            mask_R (np.ndarray): The visualized mask of the object.
            result (dict): Contains bounding boxes, contours, and object classes.
            class_names (dict): A dictionary with class names.
        """
        # Pre-image for the ONNX model
        onnx_input = self.onnx_preprocessing(img, image_size=imgsz_unet)
        onnx_input_name = self.model.get_inputs()[0].name
        ort_inputs = {onnx_input_name: onnx_input}
        # Getting Model Promise
        results_R = self.model.run(None, ort_inputs)[0]
        prob_mask_R = sigmoid(results_R[0])[0]
        # Obtaining a binary mask and adding to the size of the original image
        mask_R = create_binary_mask(prob_mask_R, desired_size=(img.shape[1], img.shape[0]), threshold=threshold)

        mask_uint8 = np.uint8(mask_R * 255)

        # Removing the containers from the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculation of the area threshold for circuit filtering
        image_area = img.shape[1] * img.shape[0]
        area_threshold = image_area * area_ratio

        # Add contours to the list if their area is larger than the threshold
        contours_threshold = [contour.reshape(-1, 2) for contour in contours if
                              cv2.contourArea(contour) > area_threshold]

        boxes = []
        for contour in contours_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))

        # YOLO Results Formation
        results_final = {
            'boxes_xyxy': boxes,
            'masks': contours_threshold,
            'cls': [0] * len(boxes),
            'probs': [0.7] * len(boxes),
        }

        class_names = {0: 'rail'}

        return results_final, class_names


    def predict_on_image_frame(
        self,
        img: np.ndarray,
        task: str,
        iou: float = 0.5,
        conf: float = 0.5,
        imgsz: int = 640,
        verbose: bool = False,
        area_ratio: float = 0.002
    ) -> Tuple:
        """The function performs the YOLO inference in the image and returns the results in the form of [boxes, masks, cls],
        probs]. Real-size masks relative to the image.

        Args:
            img (np.ndarray): Reference image.
            task (str): The task name of the detective with which the model works (PJ, LWT, R)).
            iou (float): Value Intersection Over Union. float=0.5.
            conf (float): The value of confidence in the accuracy of the predicted class. float=0.5.
            imgsz (int): The size of the input image.
            verbose (bool): Run model predictive logs or not.
            area_ratio (float): The area threshold for filtering rail contours.

        Returns:
            results (dict): Results of object detectives.
            class_names (dict): Class names.
        """
        result = self.model(img, iou=iou, conf=conf, imgsz=imgsz, verbose=verbose)[0]

        # detection
        cls = result.boxes.cls.cpu().numpy()
        class_names = result.names
        probs = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        results = {'boxes_xyxy': boxes, 'masks': [], 'cls': cls, 'probs': probs}

        # segmentation
        try:
            masks = result.masks.data.cpu().numpy()  # masks, (N, H, W)
            masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
            # rescale masks to original image
            masks = scale_image(masks, result.orig_shape)
            masks = np.moveaxis(masks, -1, 0)  # masks, (N, H, W)
            # == sync, corrected by elderman == @elder_man
            if task == 'R':
                # Calculation of the area threshold for circuit filtering
                image_area = img.shape[1] * img.shape[0]
                area_threshold = image_area * area_ratio
                masks_countour_list = []
                for mask in masks:
                    # Add contours to the list if their area is larger than the threshold
                    mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                    contours_list = [contour.reshape(-1, 2) for contour in mask_contours if
                                     cv2.contourArea(contour) > area_threshold]
                    for contour in contours_list:
                        masks_countour_list.append(contour)
                # Adding mask coordinates to the dictionary
                results['masks'] = masks_countour_list
            else:
                masks = [transform_mask_2_max_polygon(mask) for mask in masks]
                # Adding mask coordinates to the dictionary
                results['masks'] = masks
        except:
            # Recording the value of None for masks in the dictionary
            results['masks'] = [None] * len(results['boxes_xyxy'])

        return results, class_names


    def predict_rails(
        self,
        img: np.ndarray,
        unet_enabled: bool,
        r_unet_params: dict,
        rail_model_params: dict,
    ):
        """
        The function detects rails using the YOLO or Internet model. Processing the results and providing them correctly.

        Args:
            img (np.ndarray): Incoming image.
            unet_enabled (bool): Use the Unet or YOLO model.
            r_unet_params (dict): Options for the Unet model.
            rail_model_params (dict): Parameters for the YOLO model.

        Returns:
            results_R (dict): Results of rail detection.
            names_R (dict): Class names.
        """
        if unet_enabled:
            results_R, names_R = self._predict_on_image_unet(img, **r_unet_params)
        else:
            results_R, names_R = self.predict_on_image_frame(img, task='R', **rail_model_params)

        return results_R, names_R


class InferencerWorld (Inferencer):
    def __init__(self, model, text_prompts: List):
        super().__init__(model)

        # Castomic class search rooms
        self.text_prompts = text_prompts

        if len(self.text_prompts) != 0:
            self.model.set_classes(self.text_prompts)
