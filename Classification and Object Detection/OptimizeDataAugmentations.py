import numpy as np
import pandas as pd

import os
import sys
import argparse
import logging
import random

import optuna

import torch
import torchvision
import torchvision.transforms as transforms

from Models.Classifier import Classifier
from Datasets import datasets
import Utility

import lightning.pytorch as pl

from TrainAndTestModel import train_model


# Possible improvement: Support using both GPUs?


def optimize_data_augmentations(classifier_class, hyperparameters_path, dataset, epochs, results_path, gpu,
                                n_trials,
                                use_grad_acc, grad_acc_partial_batch_size, dataloaders_num_workers):

    sqlite_path = os.path.join(results_path, "data_augmentation_optimization.db")
    # Possible improvement: Make resuming possible, save studies at higher level in the same database like HP optimizations
    study = optuna.create_study(direction="maximize",
        study_name="data_augmentation_optimization", storage=f"sqlite:///{sqlite_path}")
    specific_objective = lambda trial: optuna_objective(
        trial, classifier_class, hyperparameters_path, 
        dataset, epochs, gpu, 
        use_grad_acc, grad_acc_partial_batch_size, dataloaders_num_workers
    )

    study.optimize(specific_objective, n_trials=n_trials, n_jobs=1) # n_jobs=1 because of GPU usage

    best_params = study.best_params
    best_score = study.best_value

    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best F1-score: {best_score}")

    # Save best parameters to a csv file
    best_params_df = pd.DataFrame(best_params, index=[0])
    best_params_df.to_csv(os.path.join(results_path, "best_data_augmentation_params.csv"), index=False)

    # Save best classification report and confusion matrix to a txt file
    best_classification_report = study.best_trial.user_attrs["classification_report_val"]
    best_confusion_matrix = study.best_trial.user_attrs["confusion_matrix_val"]
    with open(os.path.join(results_path, "best_data_augmentation_validation_set_classification_report.txt"), "w") as f:
        f.write(best_classification_report)
    with open(os.path.join(results_path, "best_data_augmentation_validation_set_confusion_matrix.txt"), "w") as f:
        f.write(best_confusion_matrix)
    with open(os.path.join(results_path, "best_data_augmentation_test_set_classification_report.txt"), "w") as f:
        f.write(study.best_trial.user_attrs["classification_report_test"])
    with open(os.path.join(results_path, "best_data_augmentation_test_set_confusion_matrix.txt"), "w") as f:
        f.write(study.best_trial.user_attrs["confusion_matrix_test"])


def optuna_objective(trial, classifier_class, hyperparameters_path, dataset, epochs, gpu,
                     use_grad_acc, grad_acc_partial_batch_size, dataloaders_num_workers):
    
    # Random flips
    random_horizontal_flip_prob = trial.suggest_float("random_horizontal_flip_prob", 0.0, 1.0)
    random_vertical_flip_prob = trial.suggest_float("random_vertical_flip_prob", 0.0, 1.0)

    # Color jitter
    color_jitter_brightness = trial.suggest_float("color_jitter_brightness", 0.0, 0.5)
    color_jitter_contrast = trial.suggest_float("color_jitter_contrast", 0.0, 0.5)
    color_jitter_saturation = trial.suggest_float("color_jitter_saturation", 0.0, 0.5)
    color_jitter_hue = trial.suggest_float("color_jitter_hue", 0.0, 0.5)
    
    # Random affine transformation
    random_affine_degrees = trial.suggest_float("random_affine_degrees", 0.0, 30.0)
    random_affine_translate = trial.suggest_float("random_affine_translate", 0.0, 0.2)
    random_affine_scale = trial.suggest_float("random_affine_scale", 0.0, 0.2)
    random_affine_shear = trial.suggest_float("random_affine_shear", 0.0, 3.0)
    
    # Random perspective transformation
    random_perspective_distortion_scale = trial.suggest_float("random_perspective_distortion_scale", 0.0, 0.3)
    random_perspective_p = trial.suggest_float("random_perspective_p", 0.0, 1.0)
    
    # Random sharpness adjustment
    random_adjust_sharpness_factor = trial.suggest_float("random_adjust_sharpness_factor", 0.0, 1.0)
    random_adjust_sharpness_p = trial.suggest_float("random_adjust_sharpness_p", 0.0, 1.0)
    
    # Gaussian blur
    gaussian_blur_kernel_size = trial.suggest_categorical("gaussian_blur_kernel_size", [3, 5, 7, 9, 11, 13, 15, 17, 19])
    # gaussian_blur_sigma_min = trial.suggest_float("gaussian_blur_sigma_min", 0.0, 2.0)
    # gaussian_blur_sigma_max = trial.suggest_float("gaussian_blur_sigma_max", gaussian_blur_sigma_min, 2.0)
    
    # Random histogram equalization
    random_equalize_p = trial.suggest_float("random_equalize_p", 0.0, 1.0)

    # Build big augmentation transform
    data_augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(random_horizontal_flip_prob),
        transforms.RandomVerticalFlip(random_vertical_flip_prob),
        transforms.ColorJitter(
            brightness=color_jitter_brightness, 
            contrast=color_jitter_contrast, 
            saturation=color_jitter_saturation, 
            hue=color_jitter_hue
        ),
        transforms.RandomAffine(
            degrees=random_affine_degrees, 
            translate=(random_affine_translate, random_affine_translate), 
            scale=(1 - random_affine_scale, 1 + random_affine_scale), 
            shear=random_affine_shear
        ),
        transforms.RandomPerspective(
            distortion_scale=random_perspective_distortion_scale, 
            p=random_perspective_p
        ),
        transforms.RandomAdjustSharpness(
            random_adjust_sharpness_factor, 
            p=random_adjust_sharpness_p
        ),
        transforms.GaussianBlur(
            gaussian_blur_kernel_size, 
            sigma=(0.1, 2.0)
        ),
        transforms.RandomEqualize(p=random_equalize_p)
    ])

    # Train and test a classifier model with the specified data augmentation transform for the training data
    results = train_model(
        model_class=classifier_class, hyperparameters_path=hyperparameters_path,
        dataset=dataset, epochs=epochs, results_path=None, gpu=gpu,
        data_augmentation_transform=data_augmentation_transform,
        use_grad_acc=use_grad_acc, grad_acc_partial_batch_size=grad_acc_partial_batch_size,
        dataloaders_num_workers=dataloaders_num_workers
    )

    trial.set_user_attr("classification_report_val", results["classification_report_val"])
    trial.set_user_attr("confusion_matrix_val", str(results["confusion_matrix_val"]))
    trial.set_user_attr("classification_report_test", results["classification_report_test"])
    trial.set_user_attr("confusion_matrix_test", str(results["confusion_matrix_test"]))
    
    return results["f1_val"]


def main(args):
    classifier_class = Classifier.get_classifier_class_by_name(args.classifier)
    hyperparameters_path = args.hyperparameters
    grad_acc_partial_batch_size = args.grad_acc_partial_batch_size if args.use_grad_acc else None

    _log_arguments(args)

    # Copy hyperparameters to results_path for traceability
    hyperparameters_df = pd.read_csv(hyperparameters_path)
    hyperparameters_df.to_csv(os.path.join(args.results_path, "hyperparameters.csv"), index=False)

    optimize_data_augmentations(
        classifier_class=classifier_class,
        hyperparameters_path=hyperparameters_path,
        dataset=args.dataset,
        n_trials=args.optimization_trials,
        epochs=args.epochs,
        results_path=args.results_path,
        gpu=args.gpu,
        use_grad_acc=args.use_grad_acc,
        grad_acc_partial_batch_size=grad_acc_partial_batch_size,
        dataloaders_num_workers=args.dataloaders_num_workers
    )


def _log_arguments(args):
    logging.info(f"Classifier: {args.classifier}")
    logging.info(f"Hyperparameters path: {args.hyperparameters}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Optimization trials: {args.optimization_trials}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Results path: {args.results_path}")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Use gradient accumulation: {args.use_grad_acc}")
    logging.info(f"Gradient accumulation partial batch size: {args.grad_acc_partial_batch_size}")
    logging.info(f"Dataloaders num workers: {args.dataloaders_num_workers}")


def setup():
    # Reproducibility
    Utility.init_and_seed_everything()

    # Command line arguments and config

    argument_parser = argparse.ArgumentParser(description="Train a classifier for image classifiation with specified fixed hyperparameters on a specified dataset and output the results.")
    
    argument_parser.add_argument("--classifier", type=str, required=True, choices=Classifier.get_all_available_classifier_names(), help="The type of classifier model to train.")
    argument_parser.add_argument("--hyperparameters", type=str, required=True, help="Path to the hyperparameters csv file.")
    argument_parser.add_argument("--dataset", type=str, required=True, choices=datasets.keys(), help="The dataset to use for training.")
    argument_parser.add_argument("--optimization_trials", type=int, required=True, help="Number of different parameter combinations to try.")
    argument_parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train the model for.")
    argument_parser.add_argument("--results_path", type=str, required=True, help="Path to the directory where the results should be saved.")
    argument_parser.add_argument("--gpu", type=int, required=True, help="GPU to use for training. Usually 0 or 1.")
    argument_parser.add_argument("--dataloaders_num_workers", type=int, required=False, default=8, help="Number of workers for the dataloaders.")
    argument_parser.add_argument("--use_grad_acc", action="store_true", help="Flag. Whether to use gradient accumulation.")
    argument_parser.add_argument("--grad_acc_partial_batch_size", required=False, type=int, default=4, help="Partial batch size for gradient accumulation.")
    argument_parser.add_argument("--logging_level", type=str, default="INFO", help="Logging level for the script (DEBUG, INFO, WARNING, ...)")
    args = argument_parser.parse_args()

    # Check and create directories

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Set up logging
    logging_level = args.logging_level
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.getLevelName(logging_level), # Possible improvement: getLevelName is deprecated
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # Log to both file and console (does not affect PyTorch and Optuna logging, so you only get the "full" output in the console)
            logging.FileHandler(os.path.join(args.results_path, f"log.txt")),
            logging.StreamHandler(sys.stdout)
        ])

    # Log environment information
    logging.info(f"Versions: torch={torch.__version__}, torchvision={torchvision.__version__}, pytorch-lightning={pl.__version__}, numpy={np.__version__}, pandas={pd.__version__}")
    logging.info(f"Cuda available: {torch.cuda.is_available()}")

    return args


if __name__ == '__main__':
    args = setup()
    main(args)