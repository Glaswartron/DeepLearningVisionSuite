"""
This module / script contains the train_model function, which trains a model with fixed hyperparameters.
Can be used for both binary/multiclass image classification and object detection with any of the implemented models.
When used as a script it executes the function, evaluates the model on the test and validation sets
and saves the results using the following command line arguments:
- model: The type of model to train.
- classification_multiview: Flag. Whether to use a multiview model and dataset for classification.
- hyperparameters: Path to the hyperparameters csv file.
- dataset: The dataset to use for training.
- epochs: Number of epochs to train the model for.
- results_path: Path to the directory where the results should be saved.
- data_augmentations: Path to a data augmentations yaml file. If None is provided, will use default augmentations.
- combine_train_val: Flag. Whether to  combine train and validation set for training. By default, trains only on train set.
- gpu: GPU to use for training. Usually 0 or 1.
- early_stopping_patience: Number of epochs without improvement after which training is stopped.
- dataloaders_num_workers: Number of workers for the dataloaders.
- dataloaders_persistent_workers: Flag. Whether to use persistent workers in the data loaders.
- use_grad_acc: Flag. Whether to use gradient accumulation.
- grad_acc_partial_batch_size: Partial batch size for gradient accumulation.
- logging_level: Logging level for the script (DEBUG, INFO, WARNING, ...).
"""


import numpy as np
import pandas as pd

import os
import sys
import logging
import argparse
import yaml

from Models.ImageModel import ImageModel
from Models.Classifier import Classifier
from Models.ObjectDetector import ObjectDetector
from LightningImageClassifier import LightningImageClassifier
from LightningObjectDetector import LightningObjectDetector
from TestModel import test_classifier, test_object_detector
from Datasets import make_datasets, datasets
from Models.MaskRCNNObjectDetector import MaskRCNNObjectDetector

import Utility as Utility

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

import lightning.pytorch as pl


def train_model(
    model_class, classification_multiview,
    hyperparameters, epochs, gpu,
    train_dataset,
    dataloaders_num_workers, dataloaders_persistent_workers, use_grad_acc, grad_acc_partial_batch_size=None,
    test_dataset=None, validation_dataset=None, lightning_callbacks=[],
    batch_size=None, early_stopping_patience=None,
    tensorboard_logger=None):
    """
        Trains a model with given hyperparameters.
        Args:
            model_class (class): The model class to be used.
            classification_multiview (bool): Whether to use a multiview model and dataset for classification models.
            hyperparameters (dict): Dict containing hyperparameters.
            epochs (int): Number of epochs to train the model for.
            gpu (int): GPU device ID to be used for training.
            train_dataset (torch.utils.data.Dataset): Training dataset.
            dataloaders_num_workers (int): Number of workers for data loading.
            dataloaders_persistent_workers (bool): Whether to use persistent workers in the data loaders.
            use_grad_acc (bool): Whether to use gradient accumulation.
            grad_acc_partial_batch_size (int): Partial batch size for gradient accumulation.
            test_dataset (torch.utils.data.Dataset): Test dataset. Either this or validation dataset has to be provided. Defaults to None.
            validation_dataset (torch.utils.data.Dataset): Validation dataset. Either this or test dataset has to be provided. Defaults to None.
            lightning_callbacks (list): List of additional PyTorch Lightning callbacks to be used during training.
                                        Early stopping is already being set via early_stopping_patience. Defaults to [].
            scores_only (bool): Whether to only return numerical scores (F1 or mAP, accuracy and loss) instead of more detailed results. Defaults to False.
            batch_size (int): Batch size to be used for training. If not provided, it has to be specified in the hyperparameters. Defaults to None.
            early_stopping_patience (int): Number of epochs without improvement after which training is stopped.
                                        If not provided, it can be set in the hyperparameters.
                                        If not found there either, no early stopping is done. Defaults to None.
            tensorboard_logger (pl.loggers.TensorBoardLogger): TensorBoard logger to be used for logging. Defaults to None.
        Returns:
            model_lightning_module (pl.LightningModule): The trained PyTorch Lightning module.
            trainer (pl.Trainer): The PyTorch Lightning trainer used for training.
    """

    if test_dataset is None and validation_dataset is None:
        raise ValueError("Either test_dataset or validation_dataset has to be provided")

    ''' Dataloaders and sampling '''

    if not batch_size and "batch_size" in hyperparameters:
        batch_size = hyperparameters["batch_size"]
    elif not batch_size and "batch_size" not in hyperparameters:
        raise ValueError("batch_size not specified in hyperparameters or as argument")

    # sampler_replacement only present in later versions of HPO
    replacement = hyperparameters["sampler_replacement"] if "sampler_replacement" in hyperparameters else True

    # Gradient accumulation
    if not use_grad_acc:
        accumulate_grad_batches = 1 # Used further down in Trainer
        actual_batch_size = batch_size
    else:
        if grad_acc_partial_batch_size < batch_size:
            # e.g. batch_size = 32, grad_acc_partial_batch_size = 4 -> accumulate_grad_batches = 8, actual_batch_size = 4
            accumulate_grad_batches = batch_size // grad_acc_partial_batch_size
            actual_batch_size = grad_acc_partial_batch_size
        else:
            accumulate_grad_batches = 1
            actual_batch_size = batch_size

    # Oversampling
    if issubclass(model_class, Classifier):
        if not train_dataset.multiclass:
            num_class_0 = len([label for label in train_dataset.labels if label == 0])
            num_class_1 = len([label for label in train_dataset.labels if label == 1])
            total_samples = num_class_0 + num_class_1
            class_weights = [total_samples / num_class_0, total_samples / num_class_1] # Inverse of class frequencies
            weights = [class_weights[label] for label in train_dataset.labels] # With labels 0 and 1 for each sample
            sampler = WeightedRandomSampler(weights, total_samples, replacement=replacement)
        else:
            # Possible improvement: This should also work correctly for the not-multiclass case, so it can just be used always
            nums_classes = [len([l for l in train_dataset.labels if l == class_val])
                            for class_val in sorted(list(set(train_dataset.labels)))]
            total_samples = len(train_dataset.labels) # Alternatively sum(nums_classes)
            class_weights = [total_samples / num_class for num_class in nums_classes] # Inverse of class frequencies
            weights = [class_weights[l] for l in train_dataset.labels]
            sampler = WeightedRandomSampler(weights, total_samples, replacement=replacement)
    else:
        sampler = None

    # Determine if dropping the last batch is necessary (if the dataset size is not divisible by the batch size)
    drop_last_train = len(train_dataset) % actual_batch_size != 0

    # Possible improvement: Move dataloaders (back) into lightning modules? Maybe wont work everywhere bc sometimes
    #                       you want to control what dataloaders you use by passing them to test/validate?
    # Possible improvement: Consider using PL DataModules for this (combine dataset and dataloader)
    collate_function = Utility.classification_dataloader_collate_function if not issubclass(model_class, ObjectDetector) else Utility.object_detection_dataloader_collate_function
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size,
                              sampler=sampler,
                              num_workers=dataloaders_num_workers,
                              persistent_workers=dataloaders_persistent_workers,
                              drop_last=drop_last_train,
                              collate_fn=collate_function)
    
    # Create validation dataloader
    if validation_dataset is not None:
        validation_loader = DataLoader(validation_dataset, 
                                       batch_size=1, # Always use batch size 1 for val and test
                                       shuffle=False,
                                       num_workers=dataloaders_num_workers,
                                       persistent_workers=dataloaders_persistent_workers,
                                       collate_fn=collate_function)
    else:
        validation_loader = None
    
    ''' Model training '''

    # trainable_backbone_layers for Mask R-CNN only present in later versions of HPO
    if issubclass(model_class, MaskRCNNObjectDetector):
        if "trainable_backbone_layers" not in hyperparameters:
            hyperparameters["trainable_backbone_layers"] = 3

    # Build model using model hyperparameters
    logging.info("Creating model using hyperparameters")
    if issubclass(model_class, Classifier):
        model = model_class.from_hyperparameters(hyperparameters, multiview=classification_multiview,
                                                 multiclass=train_dataset.multiclass, num_classes=train_dataset.num_classes)
    elif issubclass(model_class, ObjectDetector):
        num_classes = train_dataset.num_classes
        logging.info(f"Inferring number of classes for ObjectDetector from training set: {num_classes}")
        model = model_class.from_hyperparameters(hyperparameters, num_classes)

    optimizer_params = Utility.extract_by_prefix(hyperparameters, "optimizer_")
    optimizer_params.pop("use_weight_decay") # This is just a "meta-parameter"

    learning_rate_scheduler = Utility.STR_TO_LR_SCHEDULER[hyperparameters["learning_rate_scheduler"]]
    learning_rate_scheduler_params = Utility.extract_by_prefix(hyperparameters, "lr_scheduler_")
    # For CosineAnnealingLR, T_max is not saved but has to be set "manually" to the specific number of epochs
    # Note that T_max was actually being optimized in earlier versions of the HPO
    if learning_rate_scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
        learning_rate_scheduler_params["T_max"] = epochs

    # Create Lightning module using hyperparameters
    # Possible improvement: Try using weighted loss function (set weights using class_weights),
    #                       works for both binary and multi-class classification
    if issubclass(model_class, Classifier):
        if not train_dataset.multiclass:
            loss_criterion = torch.nn.BCEWithLogitsLoss()
            num_classes = -1
        else:
            # Multi-class loss
            # loss_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
            loss_criterion = torch.nn.CrossEntropyLoss()
            num_classes = train_dataset.num_classes
            logging.info(f"Doing multiclass classification with inferred number of classes: {num_classes}")

        model_lightning_module = LightningImageClassifier(
            model=model,
            learning_rate=hyperparameters["learning_rate"],
            multiclass=train_dataset.multiclass,
            num_classes=num_classes,
            loss_criterion=loss_criterion,
            optimizer=Utility.STR_TO_OPTIMIZER[hyperparameters["optimizer"]],
            optimizer_params=optimizer_params,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_params=learning_rate_scheduler_params
        )
    elif issubclass(model_class, ObjectDetector):
        model_lightning_module = LightningObjectDetector(
            model=model,
            learning_rate=hyperparameters["learning_rate"],
            optimizer=Utility.STR_TO_OPTIMIZER[hyperparameters["optimizer"]],
            optimizer_params=optimizer_params,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_params=learning_rate_scheduler_params
        )

    # Possible improvement: non-strict loading for SwinTransformerV2

    callbacks = []

    # Early stopping callback
    if early_stopping_patience or "early_stopping_patience" in hyperparameters:
        if "early_stopping_patience" in hyperparameters:
            early_stopping_patience = hyperparameters["early_stopping_patience"]
        early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience)
        callbacks.append(early_stopping_callback)

    # Add additional callbacks (checkpoint saving, Optuna pruning, etc.)
    callbacks.extend(lightning_callbacks)

    # Train the model
    logging.info(f"Training model")
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=tensorboard_logger,
        accelerator="gpu",
        devices=[gpu],
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        num_sanity_val_steps=0
    ) 
    trainer.fit(model_lightning_module, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    return model_lightning_module, trainer


def main(args):
    model_class = ImageModel.get_model_class_by_name(args.model)

    _log_arguments(args)

    hyperparameters = pd.read_csv(args.hyperparameters)

    # Copy hyperparameters file to results directory for traceability
    logging.info("Copying hyperparameters file to results directory")
    hyperparameters.to_csv(os.path.join(args.results_path, "hyperparameters.csv"), index=False)

    hyperparameters = hyperparameters.to_dict(orient="records")[0]

    if args.data_augmentations is None:
        if issubclass(model_class, Classifier):
            data_augmentation_transform = Utility.get_default_augmentations_classification()
        elif issubclass(model_class, ObjectDetector):
            data_augmentation_transform = Utility.get_default_augmentations_object_detection()
    else:
        if args.data_augmentations != " ":
            with open(args.data_augmentations, "r") as f:
                augmentations_dict = yaml.load(f, Loader=yaml.FullLoader)
            data_augmentation_transform = Utility.parse_data_augmentations_yaml_dict(augmentations_dict)
        else:
            data_augmentation_transform = None

    datasets = make_datasets(model_class, args.dataset, args.classification_multiview,
                             args.combine_train_val,
                             data_augmentation_transform)

    # Log augmentation transforms to results directory for traceability. Transforms here match those in make_datasets.
    transforms_path = os.path.join(args.results_path, "transforms.txt")
    with open(transforms_path, "w") as f:
        f.write("Model transform:\n")
        f.write(str(model_class.get_transforms()))
        f.write("\nTrain augmentation transform:\n")
        f.write(str(data_augmentation_transform))

    # Checkpoint callback, saves the best model based on validation F1 score or mAP
    if not args.combine_train_val:
        if issubclass(model_class, Classifier):
            monitor = "val_f1"
            # Filename e.g. "best_checkpoint_epoch=10_val_f1=0.85.ckpt"
            # Possible improvement: Maybe not optimal because filename contains two dots
            filename = f"best_checkpoint_{{epoch}}_{{val_f1}}"
        elif issubclass(model_class, ObjectDetector):
            monitor = "val_mAP"
            filename = f"best_checkpoint_{{epoch}}_{{val_mAP}}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=monitor,
            mode="max",
            dirpath=args.results_path,
            filename=filename,
            save_top_k=1
        )
        callbacks = [checkpoint_callback]
    else:
        checkpoint_callback = None
        callbacks = []

    tensorboard_run_dir = os.path.join(args.results_path, f"TensorBoard")
    tensorboard_logger = pl.loggers.TensorBoardLogger(tensorboard_run_dir, name=f"{model_class.name}_fixed_hp_training")

    # Log hyperparameters to TensorBoard
    tensorboard_logger.log_hyperparams(hyperparameters)

    grad_acc_partial_batch_size = args.grad_acc_partial_batch_size if args.use_grad_acc else None

    model_lightning_module, trainer  = train_model(
        model_class=model_class,
        classification_multiview=args.classification_multiview,
        hyperparameters=hyperparameters,
        epochs=args.epochs,
        gpu=args.gpu,
        early_stopping_patience=args.early_stopping_patience,
        train_dataset=datasets["train"],
        validation_dataset=datasets["val"] if not args.combine_train_val else None,
        test_dataset=datasets["test"],
        lightning_callbacks=callbacks,
        dataloaders_num_workers=args.dataloaders_num_workers,
        dataloaders_persistent_workers=args.dataloaders_persistent_workers,
        use_grad_acc=args.use_grad_acc,
        grad_acc_partial_batch_size=grad_acc_partial_batch_size,
        tensorboard_logger=tensorboard_logger
    )

    # Save model checkpoint if not already saved by ModelCheckpoint callback
    # Done before evaluation just in case evaluation it fails in some version
    if checkpoint_callback is None:
        logging.info("Saving model checkpoint")
        model_checkpoint_path = os.path.join(args.results_path, "model.ckpt")
        trainer.save_checkpoint(model_checkpoint_path)

    ''' Model evaluation '''

    logging.info("Evaluating model")
    validation_dataset = datasets["val"] if "val" in datasets else None
    if issubclass(model_class, Classifier):
        results = test_classifier(
            classifier=model_lightning_module,
            classification_multiview=args.classification_multiview, 
            trainer=trainer, 
            test_dataset=datasets["test"], 
            validation_dataset=validation_dataset,
            scores_only=False,
            dataloaders_num_workers=args.dataloaders_num_workers,
            dataloaders_persistent_workers=args.dataloaders_persistent_workers
        )
    elif issubclass(model_class, ObjectDetector):
        results = test_object_detector(
            object_detector=model_lightning_module,
            trainer=trainer,
            test_dataset=datasets["test"],
            validation_dataset=validation_dataset,
            dataloaders_num_workers=args.dataloaders_num_workers,
            dataloaders_persistent_workers=args.dataloaders_persistent_workers
        )
    logging.info("Done")

    # Log metrics
    logging.info("Test set metrics:")
    if issubclass(model_class, Classifier):
        logging.info(results["classification_report_test"])
        logging.info(results["confusion_matrix_test"])
        logging.info(f"F1 score (test): {results['f1_test']}")
    elif issubclass(model_class, ObjectDetector):
        logging.info(f"mAP (test): {results['mAP_test']}")
    if not args.combine_train_val:
        logging.info("Validation set metrics:")
        if issubclass(model_class, Classifier):
            logging.info(results["classification_report_val"])
            logging.info(results["confusion_matrix_val"])
            logging.info(f"F1 score (val): {results['f1_val']}")
        elif issubclass(model_class, ObjectDetector):
            logging.info(f"mAP (val): {results['mAP_val']}")

    # Save metrics
    if issubclass(model_class, Classifier):
        Utility.save_classification_test_results(results, args.test_results_path)
    elif issubclass(model_class, ObjectDetector):
        Utility.save_object_detection_test_results(results, args.test_results_path)


def _log_arguments(args):
    logging.info(f"Classifier: {args.model}")
    logging.info(f"Classification multiview: {args.classification_multiview}")
    logging.info(f"Combine train and validation set: {args.combine_train_val}")
    logging.info(f"Hyperparameters: {args.hyperparameters}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Results path: {args.results_path}")
    if args.data_augmentations is not None:
        logging.info(f"Data augmentations path: {args.data_augmentations}")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Early stopping patience: {args.early_stopping_patience}")
    logging.info(f"Number of dataloaders workers: {args.dataloaders_num_workers}")
    logging.info(f"Use persistent dataloader workers: {args.dataloaders_persistent_workers}")
    logging.info(f"Use gradient accumulation: {args.use_grad_acc}")
    logging.info(f"Gradient accumulation partial batch size: {args.grad_acc_partial_batch_size}")
    logging.info(f"Logging level: {args.logging_level}")


def setup():
    # Reproducibility
    Utility.init_and_seed_everything()

    # Command line arguments and config

    available_models = Classifier.get_all_available_classifier_names() + ObjectDetector.get_all_available_object_detector_names()

    argument_parser = argparse.ArgumentParser(description="Train a model for image classifiation or object detection with specified fixed hyperparameters on a specified dataset and output the results.")
    
    argument_parser.add_argument("--model", type=str, required=True, choices=available_models, help="The type of model to train.")
    argument_parser.add_argument("--classification_multiview", action="store_true", help="Flag. Whether to use a multiview model and dataset for classification.")
    argument_parser.add_argument("--hyperparameters", type=str, required=True, help="Path to the hyperparameters csv file.")
    argument_parser.add_argument("--dataset", type=str, required=True, choices=datasets.keys(), help="The dataset to use for training.")
    argument_parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train the model for.")
    argument_parser.add_argument("--results_path", type=str, required=True, help="Path to the directory where the results should be saved.")
    argument_parser.add_argument("--data_augmentations", type=str, required=False, help="Path to a data augmentations yaml file. If None is provided, will use default augmentations.")
    argument_parser.add_argument("--combine_train_val", action="store_true", dest="combine_train_val", help="Flag. Whether to combine train and validation set for training and only train on the train set. By default, train and val kept seperate and validation is performed after each train epoch.")
    argument_parser.add_argument("--gpu", type=int, required=True, help="GPU to use for training. Usually 0 or 1.")
    argument_parser.add_argument("--early_stopping_patience", type=int, required=False, help="Number of epochs without improvement after which training is stopped.")
    argument_parser.add_argument("--dataloaders_num_workers", type=int, required=False, default=8, help="Number of workers for the dataloaders.")
    argument_parser.add_argument("--dataloaders_no_persistent_workers", action="store_false", dest="dataloaders_persistent_workers", help="Flag. Usually persistant workers are used for the dataloaders. This disables that.")
    argument_parser.add_argument("--use_grad_acc", action="store_true", help="Flag. Whether to use gradient accumulation.")
    argument_parser.add_argument("--grad_acc_partial_batch_size", required=False, type=int, default=4, help="Partial batch size for gradient accumulation.")
    argument_parser.add_argument("--logging_level", type=str, default="INFO", help="Logging level for the script (DEBUG, INFO, WARNING, ...)")
    args = argument_parser.parse_args()

    # Ensure grad_acc_partial_batch_size is set if use_grad_acc is set
    if args.use_grad_acc and args.grad_acc_partial_batch_size is None:
        raise ValueError("grad_acc_partial_batch_size must be set if use_grad_acc is set.")

    # Check and create results directory
    if not os.path.exists(args.results_path):
        # No logging available yet
        print(f"Results directory does not exist. Creating it at {args.results_path}")
        os.makedirs(args.results_path)

    # Set up logging
    logging_level = args.logging_level
    handlers = [logging.StreamHandler(sys.stdout)]
    # Log to both file and console (does not affect PyTorch and Optuna logging, so you only get the "full" output in the console)
    handlers.append(logging.FileHandler(os.path.join(args.results_path, f"log.txt")))
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.getLevelName(logging_level),
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

    # Possible improvement: Warning if a training folder at the results path already exists

    # Check and create test results directory
    # Adds "test_results_path" to args in addition to the normal results_path
    # Possible improvement: Merge with the same logic in TestClassifier.py into a utility function?
    if not os.path.exists(os.path.join(args.results_path, "Test results")):
        logging.info(f"Test results directory does not exist. Creating it at {os.path.join(args.results_path, 'Test results')}")
        os.makedirs(os.path.join(args.results_path, "Test results"))
        args.test_results_path = os.path.join(args.results_path, "Test results")
    elif len(os.listdir(os.path.join(args.results_path, "Test results"))) > 0:
        # In case there already is a Test results directory, append a number or the next higher number to the directory name
        # If user causes integer overflow, user probably has bigger problems
        num = 1
        while os.path.exists(os.path.join(args.results_path, f"Test results {num}")):
            num += 1
        logging.warning(f"Test results directory is not empty. Creating new directory at {os.path.join(args.results_path, f'Test results {num}')}")
        os.makedirs(os.path.join(args.results_path, f"Test results {num}"))
        args.test_results_path = os.path.join(args.results_path, f"Test results {num}")
    else:
        args.test_results_path = os.path.join(args.results_path, "Test results")

    # Log environment information
    logging.info(f"Versions: torch={torch.__version__}, torchvision={torchvision.__version__}, pytorch-lightning={pl.__version__}, numpy={np.__version__}, pandas={pd.__version__}")
    logging.info(f"Cuda available: {torch.cuda.is_available()}")

    return args


if __name__ == "__main__":
    args = setup() 
    main(args)