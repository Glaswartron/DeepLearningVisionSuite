'''
This module contains utility functions and structures for the deep learning suite.
'''

import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2.functional as F
import torchvision.transforms.v2 as transforms

import lightning.pytorch as pl

from Models.Classifier import Classifier
from Models.ObjectDetector import ObjectDetector
from Datasets import *


STR_TO_OPTIMIZER = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSProp": optim.RMSprop,
    "AdamW": optim.AdamW
}

STR_TO_LR_SCHEDULER = {
    # "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "StepLR": optim.lr_scheduler.StepLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR
}


def load_model_from_checkpoint(model_class, hyperparameters, checkpoint_path, 
                               classification_multiview=False,
                               classification_multiclass=False,
                               num_classes=-1,
                               strict=True):
    '''
    Loads a model (classifier or object detector) with the given model hyperparameters from a checkpoint file and initializes it.  
    Uses the `from_hyperparameters` implementation of the model class internally.

    Args:
        model_class: class
            The class of the model to load.
        hyperparameters: dict
            The hyperparameters to initialize the model with.
        checkpoint_path: str
            The path to the checkpoint file.
        classification_multiview: bool
            Whether the model is a multi-view classifier.
        classification_multiclass: bool
            Whether the model/dataset is multi-class.
        num_classes: int
            The number of classes for object detectors or multi-class classifiers.
        strict: bool
            Whether to use strict loading in the models load_state_dict method.
    
    Returns:
        model: Classifier or ObjectDetector
            The loaded model.
    '''
    if issubclass(model_class, Classifier):
        model = model_class.from_hyperparameters(hyperparameters, multiview=classification_multiview,
                                                 multiclass=classification_multiclass, num_classes=num_classes)
    elif issubclass(model_class, ObjectDetector):
        model = model_class.from_hyperparameters(hyperparameters, num_classes)
    checkpoint = torch.load(checkpoint_path)
    weights = {k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(weights, strict=strict)
    return model


def get_ckpt_and_params_paths_from_dir(dir_path):
    '''
    Given a directory path, this function returns the path to the best checkpoint and the best hyperparameters file.

    Args:
        dir_path: str
            The path to the directory containing the checkpoint and hyperparameters files.

    Returns:
        ckpt_path: str
            The path to the best checkpoint file.  
        params_path: str
            The path to the best hyperparameters file.
    '''
    all_files = os.listdir(dir_path)

    ckpt_files = list(filter(lambda file_name: file_name.startswith("best_") and file_name.endswith(".ckpt"), all_files))
    if len(ckpt_files) == 0:
        ckpt_files = list(filter(lambda file_name: file_name.endswith(".ckpt"), all_files))

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint found in directory: {dir_path}")
    elif len(ckpt_files) > 1:
        raise ValueError(f"Multiple possible checkpoints found in directory: {dir_path}")
    
    ckpt_file = ckpt_files[0]
    ckpt_path = os.path.join(dir_path, ckpt_file)

    hyperparameters_file_list = list(filter(
        lambda file_name: (file_name.startswith("best_hyperparameters") and file_name.endswith(".csv")) 
        or file_name == "hyperparameters.csv", all_files
    ))
        
    if len(hyperparameters_file_list) == 0:
        raise FileNotFoundError(f"No best hyperparameters found in directory: {dir_path}")
    elif len(hyperparameters_file_list) > 1:
        raise ValueError(f"Multiple best hyperparameters found in directory: {dir_path}")
    
    hyperparameters_file = hyperparameters_file_list[0]
    params_path = os.path.join(dir_path, hyperparameters_file)

    return ckpt_path, params_path


def save_classification_test_results(results, test_results_path):
    # Save classification report
    with open(os.path.join(test_results_path, "classification_report_test.txt"), "w") as f:
        f.write(results["classification_report_test"])
    if "classification_report_val" in results:
        with open(os.path.join(test_results_path, "classification_report_val.txt"), "w") as f:
            f.write(results["classification_report_val"])

    # Save confusion matrix
    conf_matrix_df = pd.DataFrame(results["confusion_matrix_test"])
    conf_matrix_df.to_csv(os.path.join(test_results_path, "confusion_matrix_test.csv"), index=False)
    if "confusion_matrix_val" in results:
        conf_matrix_df = pd.DataFrame(results["confusion_matrix_val"])
        conf_matrix_df.to_csv(os.path.join(test_results_path, "confusion_matrix_val.csv"), index=False)

    # Save file names of misclassified images
    set_types = ["test"]
    if "misclassified_file_names_false_positive_val" in results:
        # If validation set results are present, save them as well
        set_types.append("val")
    for set_type in set_types:
        if f"false_positives_{set_type}" in results:
            with open(os.path.join(test_results_path, f"misclassified_file_names_false_positive_{set_type}.txt"), "w") as f:
                f.write("\n".join([str(e) for e in results[f"false_positives_{set_type}"]]))
            with open(os.path.join(test_results_path, f"misclassified_file_names_false_negative_{set_type}.txt"), "w") as f:
                f.write("\n".join([str(e) for e in results[f"false_negatives_{set_type}"]]))
        if f"false_positives_raw_predictions_{set_type}" in results:
            with open(os.path.join(test_results_path, f"misclassified_raw_predictions_false_positive_{set_type}.txt"), "w") as f:
                f.write("\n".join([str(x) for x in results[f"false_positives_raw_predictions_{set_type}"]]))
            with open(os.path.join(test_results_path, f"misclassified_raw_predictions_false_negative_{set_type}.txt"), "w") as f:
                f.write("\n".join([str(x) for x in results[f"false_negatives_raw_predictions_{set_type}"]]))


def save_object_detection_test_results(results, test_results_path):
    # Save mAP
    with open(os.path.join(test_results_path, "mAP_test.txt"), "w") as f:
        f.write(str(results["mAP_test"]))
    if "mAP_val" in results:
        with open(os.path.join(test_results_path, "mAP_val.txt"), "w") as f:
            f.write(str(results["mAP_val"]))


def parse_data_augmentations_yaml_dict(yaml_dict):
    transform_list = []

    # Check and add random flip
    if "random_flip" in yaml_dict:
        flip_params = yaml_dict["random_flip"]
        if "random_horizontal_flip_prob" in flip_params:
            transform_list.append(transforms.RandomHorizontalFlip(p=flip_params["random_horizontal_flip_prob"]))
        if "random_vertical_flip_prob" in flip_params:
            transform_list.append(transforms.RandomVerticalFlip(p=flip_params["random_vertical_flip_prob"]))

    # Check and add color jitter
    if "color_jitter" in yaml_dict:
        jitter_params = yaml_dict["color_jitter"]
        transform_list.append(transforms.ColorJitter(
            brightness=jitter_params.get("color_jitter_brightness", 0),
            contrast=jitter_params.get("color_jitter_contrast", 0),
            saturation=jitter_params.get("color_jitter_saturation", 0),
            hue=jitter_params.get("color_jitter_hue", 0)
        ))

    # Check and add random affine
    if "random_affine" in yaml_dict:
        affine_params = yaml_dict["random_affine"]
        transform_list.append(transforms.RandomAffine(
            degrees=affine_params.get("random_affine_degrees", 0),
            translate=(
                affine_params.get("random_affine_translate_width", 0),
                affine_params.get("random_affine_translate_height", 0)
            ),
            scale=(
                affine_params.get("random_affine_scale_min", 1),
                affine_params.get("random_affine_scale_max", 1)
            ),
            shear=affine_params.get("random_affine_shear", 0)
        ))

    # Check and add random perspective
    if "random_perspective" in yaml_dict:
        perspective_params = yaml_dict["random_perspective"]
        transform_list.append(transforms.RandomPerspective(
            distortion_scale=perspective_params.get("random_perspective_distortion_scale", 0.5),
            p=perspective_params.get("random_perspective_p", 0.5)
        ))

    # Check and add random sharpness adjustment
    if "random_sharpness_adjustment" in yaml_dict:
        sharpness_params = yaml_dict["random_sharpness_adjustment"]
        transform_list.append(transforms.RandomAdjustSharpness(
            sharpness_params.get("random_adjust_sharpness_factor", 1),
            p=sharpness_params.get("random_adjust_sharpness_p", 0.5)
        ))

    # Check and add Gaussian blur
    if "gaussian_blur" in yaml_dict:
        blur_params = yaml_dict["gaussian_blur"]
        transform_list.append(transforms.GaussianBlur(
            kernel_size=blur_params.get("gaussian_blur_kernel_size", 3),
            sigma=(
                blur_params.get("gaussian_blur_sigma_min", 0.1),
                blur_params.get("gaussian_blur_sigma_max", 2.0)
            )
        ))

    # Check and add random histogram equalization
    if "random_histogram_equalization" in yaml_dict:
        equalize_params = yaml_dict["random_histogram_equalization"]
        transform_list.append(transforms.RandomEqualize(
            p=equalize_params.get("random_equalize_p", 0.5)
        ))

    # Return composed transforms
    return transforms.Compose(transform_list)


def get_default_augmentations_classification():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.15),
        # (0.05, 0.05) is range in width and height to translate
        transforms.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.92, 1.08), shear=2.2),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.RandomAdjustSharpness(0.45, p=0.5),
        transforms.GaussianBlur(9, sigma=(0.05, 1.5)),
        transforms.RandomEqualize(p=0.5)
    ])


def get_default_augmentations_object_detection():
    # No transforms that would deform bounding boxes
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.15),
        transforms.RandomAdjustSharpness(0.45, p=0.5),
        transforms.GaussianBlur(9, sigma=(0.05, 1.5)),
        transforms.RandomEqualize(p=0.5)
    ])


class ScriptableNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # Convert mean and std to tensors for TorchScript compatibility
        self.mean = torch.tensor(mean).view(-1, 1, 1)  # Shape: [C, 1, 1]
        self.std = torch.tensor(std).view(-1, 1, 1)    # Shape: [C, 1, 1]

    def forward(self, x):
        # Ensure mean and std are on the same device as x
        if x.dim() == 3:  # Handle single image (C, H, W)
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        else:
            raise ValueError(f"Expected input with 3 dimensions (C, H, W), but got {x.dim()} dimensions.")


def classification_dataloader_collate_function(batch):
    """
    Custom collate function to handle variable-sized images and
    wrap images in a list and labels in a tensor if needed.

    Args:
        batch (list): A list of (image, label) tuples.

    Returns:
        tuple: A tuple containing a list of images and a tensor of labels.
    """
    images, labels = zip(*batch)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to a tensor
    return list(images), labels


def object_detection_dataloader_collate_function(batch):
    return tuple(zip(*batch))


def resize_boxes(boxes: torch.Tensor, original_size: tuple[int, int], new_size: tuple[int, int]):
    """
    Resize bounding boxes according to the new image size.

    Parameters:
    boxes (Tensor): Bounding boxes in the format [x_min, y_min, x_max, y_max]
    original_size (tuple): Original image size (height, width)
    new_size (tuple): New image size (height, width)

    Returns:
    Tensor: Resized bounding boxes
    """
    original_height, original_width = original_size
    new_height, new_width = new_size
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    resized_boxes = boxes.clone()
    resized_boxes[:, [0, 2]] *= scale_x
    resized_boxes[:, [1, 3]] *= scale_y

    return resized_boxes


def init_and_seed_everything():
    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # GPU setup
    torch.set_float32_matmul_precision('medium')


def extract_by_prefix(dictionary, prefix):
    '''
    Gets all key-value pairs from a dictionary where the key starts with a specified prefix.
    Removes the prefix from the keys. Prefix should be of the form "prefix\_".
    '''

    return {key.removeprefix(prefix): dictionary[key] for key in dictionary if key.startswith(prefix)}