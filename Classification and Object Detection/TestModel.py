'''
This script trains and tests a model for binary/multiclass image classifiation or object detection
with specified hyperparameters on a specified dataset and outputs the results.

Usage:
    python TestClassifier.py --model <model> --checkpoint <checkpoint> --hyperparameters <hyperparameters> --results_path <results_path> --dataset <dataset> --gpu <gpu> [--no_strict_ckpt_loading]
    python TestClassifier.py --model <model> --run_path <run_path> --dataset <dataset> --gpu <gpu> [--no_strict_ckpt_loading]
'''

import numpy as np
import pandas as pd

import os
import logging
import argparse

import Utility

import torch
from torch.utils.data import DataLoader
import torchvision

import lightning.pytorch as pl

import torchmetrics

from Models.ImageModel import ImageModel
from Models.Classifier import Classifier
from Models.ObjectDetector import ObjectDetector
from Models.MaskRCNNObjectDetector import MaskRCNNObjectDetector
from LightningImageClassifier import LightningImageClassifier
from LightningObjectDetector import LightningObjectDetector
from Datasets import datasets, make_datasets

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


# Possible improvement: Indicate that this expects a Lightning module
def test_classifier(classifier, classification_multiview, trainer,
                    test_dataset=None, validation_dataset=None,
                    scores_only=False, dataloaders_num_workers=8, dataloaders_persistent_workers=True):
    
    # Create dataloaders for test and validation set
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, 
                                 batch_size=1, # Always use batch size 1 for val and test
                                 shuffle=False,
                                 num_workers=dataloaders_num_workers,
                                 persistent_workers=dataloaders_persistent_workers,
                                 collate_fn=Utility.classification_dataloader_collate_function)
    else:
        test_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(validation_dataset, 
                                       batch_size=1, # Always use batch size 1 for val and test
                                       shuffle=False,
                                       num_workers=dataloaders_num_workers,
                                       persistent_workers=dataloaders_persistent_workers,
                                       collate_fn=Utility.classification_dataloader_collate_function)
    else:
        validation_loader = None

    # Test the classifier on the test set and get predictions from the test set
    if test_loader is not None:

        logging.info("Running test on test set")
        metrics = trainer.test(classifier, dataloaders=test_loader)
        
        predictions_test = np.array(classifier.test_predictions)
        labels_test = np.array(classifier.test_labels)
        loss_test = metrics[0]["test_loss"] 
        if not scores_only and not test_dataset.multiclass:
            # Certainty / Probability of label 1 directly after sigmoid
            raw_predictions_test = np.array(classifier.test_predictions_raw)

        # Calculate test set metrics
        average = "binary" if not test_dataset.multiclass else "macro"
        f1_test = f1_score(labels_test, predictions_test, average=average)
        accuracy_test = accuracy_score(labels_test, predictions_test)
        if not scores_only:
            class_report_test = classification_report(labels_test, predictions_test)
            conf_matrix_test = confusion_matrix(labels_test, predictions_test)

    # Test the classifier on the validation set
    if validation_loader is not None:
        # Note that test is used, not validate (were "testing on the validation set")

        classifier.test_predictions = []
        classifier.test_labels = []
        classifier.test_predictions_raw = []

        logging.info("Running test on validation set")
        metrics = trainer.test(classifier, dataloaders=validation_loader)
        
        predictions_val = np.array(classifier.test_predictions)
        labels_val = np.array(classifier.test_labels)
        loss_val = metrics[0]["test_loss"] 
        if not scores_only and not validation_dataset.multiclass:
            raw_predictions_val = np.array(classifier.test_predictions_raw)

        # Calculate validation set metrics
        average = "binary" if not validation_dataset.multiclass else "macro"
        f1_val = f1_score(labels_val, predictions_val, average=average)
        accuracy_val = accuracy_score(labels_val, predictions_val)
        if not scores_only:
            class_report_val = classification_report(labels_val, predictions_val)
            conf_matrix_val = confusion_matrix(labels_val, predictions_val)

    # Get misclassified files and raw predictions for binary classification
    if not scores_only and not test_dataset.multiclass:
        if test_loader is not None:
            false_positives_output_test, false_negatives_output_test,\
            false_positive_raw_predictions_output_test, false_negative_raw_predictions_output_test =\
                _get_misclassified_files_and_raw_predictions(predictions_test, labels_test, test_dataset, raw_predictions_test, multiview=classification_multiview)
        if validation_loader is not None:
            false_positives_output_val, false_negatives_output_val,\
            false_positive_raw_predictions_output_val, false_negative_raw_predictions_output_val =\
                _get_misclassified_files_and_raw_predictions(predictions_val, labels_val, validation_dataset, raw_predictions_val, multiview=classification_multiview)

    results = {}
    if test_loader is not None:
        results["f1_test"] = f1_test
        results["accuracy_test"] = accuracy_test
        results["loss_test"] = loss_test
        if not scores_only:
            results["classification_report_test"] = class_report_test
            results["confusion_matrix_test"] = conf_matrix_test
            if not test_dataset.multiclass:
                results["false_positives_test"] = false_positives_output_test
                results["false_negatives_test"] = false_negatives_output_test
                results["false_positives_raw_predictions_test"] = false_positive_raw_predictions_output_test
                results["false_negatives_raw_predictions_test"] = false_negative_raw_predictions_output_test
    if validation_loader is not None:
        results["f1_val"] = f1_val
        results["accuracy_val"] = accuracy_val
        results["loss_val"] = loss_val
        if not scores_only:
            results["classification_report_val"] = class_report_val
            results["confusion_matrix_val"] = conf_matrix_val
            if not test_dataset.multiclass:
                results["false_positives_val"] = false_positives_output_val
                results["false_negatives_val"] = false_negatives_output_val
                results["false_positives_raw_predictions_val"] = false_positive_raw_predictions_output_val
                results["false_negatives_raw_predictions_val"] = false_negative_raw_predictions_output_val

    return results


def test_object_detector(object_detector, trainer,
                         test_dataset=None, validation_dataset=None,
                         dataloaders_num_workers=8, dataloaders_persistent_workers=True):

    # Create dataloaders for test and validation set
    collate_function = Utility.object_detection_dataloader_collate_function
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, 
                                 batch_size=1, # Always use batch size 1 for val and test
                                 shuffle=False,
                                 num_workers=dataloaders_num_workers,
                                 persistent_workers=dataloaders_persistent_workers,
                                 collate_fn=collate_function)
    else:
        test_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(validation_dataset, 
                                       batch_size=1, # Always use batch size 1 for val and test
                                       shuffle=False,
                                       num_workers=dataloaders_num_workers,
                                       persistent_workers=dataloaders_persistent_workers,
                                       collate_fn=collate_function)
    else:
        validation_loader = None

    results = {}

    # Test the object detector on the test set and get predictions from the test set
    if test_loader is not None:
        metrics = trainer.test(object_detector, dataloaders=test_loader)
        predictions_test = object_detector.test_predictions.copy()
        targets_test = object_detector.test_targets.copy()
        results["loss_test"] = metrics[0]["test_loss"]

        # Calculate test set mAP
        mAP_metric = torchmetrics.detection.mean_ap.MeanAveragePrecision()
        mAP_metric.update(predictions_test, targets_test)
        results["mAP_test"] = mAP_metric.compute()["map"]

    # Test the object detector on the validation set
    if validation_loader is not None:
        # Note that test is used, not validate
        object_detector.test_predictions = []
        object_detector.test_targets = []
        metrics = trainer.test(object_detector, dataloaders=validation_loader)
        predictions_val = object_detector.test_predictions.copy()
        targets_val = object_detector.test_targets.copy()
        results["loss_val"] = metrics[0]["test_loss"]

        # Calculate validation set mAP
        mAP_metric = torchmetrics.detection.mean_ap.MeanAveragePrecision()
        mAP_metric.update(predictions_val, targets_val)
        results["mAP_val"] = mAP_metric.compute()["map"]
        
    return results


def _get_misclassified_files_and_raw_predictions(predictions, labels, dataset, raw_predictions, multiview):
    # Get misclassified indices
    misclassified_indices = np.where(predictions != labels)[0]
    # Get file names of misclassified images
    if not multiview:
        misclassified_file_names = dataset.file_names[misclassified_indices]
        misclassified_true_labels = labels[misclassified_indices] # The true labels of the misclassified images
        misclassified_raw_predictions = raw_predictions[misclassified_indices]
        false_positives_output = misclassified_file_names[misclassified_true_labels == 0]
        false_negatives_output = misclassified_file_names[misclassified_true_labels == 1]
        false_positive_raw_predictions_output = misclassified_raw_predictions[misclassified_true_labels == 0]
        false_negative_raw_predictions_output = misclassified_raw_predictions[misclassified_true_labels == 1]
        return false_positives_output, false_negatives_output, false_positive_raw_predictions_output, false_negative_raw_predictions_output
    else:
        # Returns lists of lists
        file_names_lists = [[view[0] for view in sample] for sample in dataset.data]
        misclassified_file_names_lists = [file_names_lists[i] for i in misclassified_indices]
        misclassified_true_labels = labels[misclassified_indices] # The true labels of the misclassified images
        misclassified_raw_predictions = raw_predictions[misclassified_indices]
        false_positives_indices = np.where(misclassified_true_labels == 0)[0]
        false_negatives_indices = np.where(misclassified_true_labels == 1)[0]
        false_positives_output = [misclassified_file_names_lists[i] for i in false_positives_indices]
        false_negatives_output = [misclassified_file_names_lists[i] for i in false_negatives_indices]
        false_positive_raw_predictions_output = [misclassified_raw_predictions[i] for i in false_positives_indices]
        false_negative_raw_predictions_output = [misclassified_raw_predictions[i] for i in false_negatives_indices]
        return false_positives_output, false_negatives_output, false_positive_raw_predictions_output, false_negative_raw_predictions_output


def main(args, results_path):
    # Extract some constants from args for easier access
    model_class = ImageModel.get_model_class_by_name(args.model)
    classification_multiview = args.classification_multiview
    if args.run_path is None:
        checkpoint_path = args.checkpoint
        hyperparameters_path = args.hyperparameters
    else:
        checkpoint_path, hyperparameters_path = Utility.get_ckpt_and_params_paths_from_dir(args.run_path)

    # Load hyperparameters
    hyperparameters = pd.read_csv(hyperparameters_path).to_dict(orient="records")[0]

    # For backwards compatibility with older hyperparameter CSV files
    if issubclass(model_class, MaskRCNNObjectDetector) and "trainable_backbone_layers" not in hyperparameters:
        hyperparameters["trainable_backbone_layers"] = 3

    # Create datasets based on specific type of problem
    datasets = make_datasets(model_class, args.dataset, classification_multiview, combine_train_val=False,
                             data_augmentation_transform=None, train=False, val=True, test=True)
    test_dataset = datasets["test"]
    validation_dataset = datasets["val"]

    if issubclass(model_class, ObjectDetector) or test_dataset.multiclass:
        num_classes = test_dataset.num_classes
        logging.info(f"Inferring number of classes for ObjectDetector or Multiclass Classifier from test dataset: {num_classes}")
    else:
        num_classes = -1

    multiclass = test_dataset.multiclass if not issubclass(model_class, ObjectDetector) else False
        
    # Load model from checkpoint
    logging.info("Loading model")
    model = Utility.load_model_from_checkpoint(model_class=model_class, 
                                               hyperparameters=hyperparameters,
                                               checkpoint_path=checkpoint_path,
                                               classification_multiview=classification_multiview, 
                                               classification_multiclass=multiclass,
                                               num_classes=num_classes,
                                               strict=args.strict_ckpt_loading)

    '''
    Set up Lightning module
    Note that batch_size and learning_rate are not used in the test method, so they can be set to any value
    '''
    if issubclass(model_class, Classifier):
        model_lightning_module = LightningImageClassifier.load_from_checkpoint(
            checkpoint_path, model=model, 
            multiclass=test_dataset.multiclass, num_classes=test_dataset.num_classes,
            loss_criterion=torch.nn.BCEWithLogitsLoss() if not test_dataset.multiclass else torch.nn.CrossEntropyLoss(),
            batch_size=1, learning_rate=0, strict=args.strict_ckpt_loading
        )
    elif issubclass(model_class, ObjectDetector):
        model_lightning_module = LightningObjectDetector.load_from_checkpoint(
            checkpoint_path, model=model, batch_size=1, learning_rate=0, strict=args.strict_ckpt_loading
        )

    trainer = pl.Trainer(accelerator="gpu", devices=[args.gpu])

    logging.info("Testing model")

    # Test the model based on its type and get results
    if issubclass(model_class, Classifier):
        results = test_classifier(model_lightning_module, classification_multiview, trainer, 
                                  test_dataset=test_dataset, validation_dataset=validation_dataset)
        logging.info("Saving results")
        Utility.save_classification_test_results(results, results_path)
    elif issubclass(model_class, ObjectDetector):
        results = test_object_detector(model_lightning_module, trainer,
                                       test_dataset=test_dataset, validation_dataset=validation_dataset)
        logging.info("Saving results")
        Utility.save_object_detection_test_results(results, results_path)

    logging.info("Done")


def setup(args):

    # Reproducibility
    Utility.init_and_seed_everything()

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.getLevelName(args.logging_level),
        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log environment information
    logging.info(f"Versions: torch={torch.__version__}, torchvision={torchvision.__version__}, pytorch-lightning={pl.__version__}, numpy={np.__version__}, pandas={pd.__version__}")
    logging.info(f"Cuda available: {torch.cuda.is_available()}")
    
    # Check and create directories
    if args.run_path is None:
        results_path = args.results_path
    else:
        # If run_path is provided, results_path does not have to be set (but can be)
        results_path = args.run_path if args.results_path is None else args.results_path
    if not os.path.exists(results_path):
        logging.info(f"Results directory does not exist. Creating it at {results_path}")
        os.makedirs(results_path)
    if not os.path.exists(os.path.join(results_path, "Test results")):
        logging.info(f"Test results directory does not exist. Creating it at {os.path.join(results_path, 'Test results')}")
        os.makedirs(os.path.join(results_path, "Test results"))
        results_path = os.path.join(results_path, "Test results")
    elif len(os.listdir(os.path.join(results_path, "Test results"))) > 0:
        # In case there already is a Test results directory, append a number or the next higher number to the directory name
        # If user causes integer overflow, user probably has bigger problems
        num = 1
        while os.path.exists(os.path.join(results_path, f"Test results {num}")):
            num += 1
        logging.warning(f"Test results directory is not empty. Creating new directory at {os.path.join(results_path, f'Test results {num}')}")
        os.makedirs(os.path.join(results_path, f"Test results {num}"))
        results_path = os.path.join(results_path, f"Test results {num}")
    else:
        results_path = os.path.join(results_path, "Test results")

    return results_path


def parse_args():
    available_models = Classifier.get_all_available_classifier_names() + ObjectDetector.get_all_available_object_detector_names()
    parser = argparse.ArgumentParser(description="Test a checkpoint for image classification or object detection.")
    parser.add_argument("--model", type=str, required=True, choices=available_models, help="The model to use for testing.")
    parser.add_argument("--classification_multiview", action="store_true", help="Flag. Whether you are a multiview model on a multiview dataset.")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the classifier checkpoint to test. Does not have to be provided if run_path is provided.")
    # Possible improvement: Consider making hyperparameters completely optional for models that dont have model hyperparameters
    parser.add_argument("--hyperparameters", type=str, required=False, help="Path to the hyperparameters csv file containing the model hyperparameters for the model. Does not have to be provided if run_path is provided.")
    parser.add_argument("--run_path", type=str, required=False, help="Path to the directory containing the results of the hyperparameter optimization run. If provided, checkpoint and hyperparameters do not have to be provided.")
    parser.add_argument("--results_path", type=str, required=False, help="Path to the directory where the results should be saved. Does not have to be provided if run_path is provided.")
    parser.add_argument("--dataset", type=str, required=True, choices=datasets.keys(), help="The dataset to use for training and testing.")
    parser.add_argument("--gpu", type=int, required=True, help="The GPU to use for testing.")
    parser.add_argument("--no_strict_ckpt_loading", default=True, required=False, action="store_false", dest="strict_ckpt_loading", help="Flag. Disable strict loading for checkpoint.")
    parser.add_argument("--logging_level", type=str, default="INFO", required=False, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="The logging level to use.")

    args = parser.parse_args()

    # Enforce that either checkpoint, hyperparameters and results_path are provided or run_path is provided
    if args.run_path is None and (args.checkpoint is None or args.hyperparameters is None or args.results_path is None):
        parser.error("If run_path is not provided, checkpoint, hyperparameters and results_path have to be provided.")
    if args.run_path is not None and (args.checkpoint is not None or args.hyperparameters is not None or args.results_path is not None):
        parser.error("If run_path is provided, checkpoint, hyperparameters and results_path have to be omitted.")

    return args


if __name__ == "__main__":
    args = parse_args()
    results_path = setup(args)
    main(args, results_path)