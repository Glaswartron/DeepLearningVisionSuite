'''
This script is responsible for optimizing a model - binary/multiclass image classifier or object detector - on an image dataset.
It runs hyperparameter optimization for the given model and dataset, and saves the best hyperparameters to a CSV file.
All parameters, paths, etc. have to be specified in the config file or as arguments.

Basic usage: python TrainAndOptimizeClassifier.py --config path/to/config.yaml --run_name "Run 1" --gpu 0 --allow_continue_run False

Arguments:  
- config is the path to the configuration file (format see the existing configs in the Configs/Hyperparameter optimization folder)
- run_name is a unique run identifier (string or int) for this HPO run. Set manually.
- gpu is the GPU to use for training. Usually 0 or 1.
- allow_continue_run is whether to allow continuing an existing run with the same name. Optional, default is False.
- use_grad_acc is whether to use gradient accumulation. Optional, default is False.
- grad_acc_partial_batch_size is the partial batch size for gradient accumulation. Optional, default is 4.
- dataloaders_num_workers is the number of workers for the dataloaders. Optional, default is 8.
- dataloaders_no_persistent_workers is a flag. Usually persistent workers are used for the dataloaders. This disables that. Optional, default is False.
- logging_level is the logging level for the script (DEBUG, INFO, WARNING, ...). Optional, default is INFO.

The script will create a subdirectory for the run in the results directory and save the best hyperparameters to a CSV file.
An Optuna SQLite database will be created or accessed in the provided results directory.
The script will also create a TensorBoard folder and a log file for the run and log the hyperparameter ranges and transforms that were used.
'''

import numpy as np
import pandas as pd

import os
import sys
import logging
import argparse
import yaml
import datetime

from Models.ImageModel import ImageModel
from Models.Classifier import Classifier
from Models.ObjectDetector import ObjectDetector
from HyperparameterOptimization import optimize_hyperparameters
from Datasets import make_datasets

import Utility as Utility

import torch
import torchvision
import torchvision.transforms.v2 as transforms

import lightning.pytorch as pl


def main(config, args):
    # Extract some constants from configuration for quick access
    if "model" in config:
        model_name = config["model"]
    elif "classifier" in config:
        # For compatibility with old configs where it was only called "classifier"
        model_name = config["classifier"]
    model_class = ImageModel.get_model_class_by_name(model_name)
    classification_multiview = config["classification_multiview"] if "classification_multiview" in config else False
    results_path = config["results_path"]

    # Log configuration
    logging.info(f"Run name: {args.run_name}")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Allow continue run: {args.allow_continue_run}")
    logging.info(f"Use gradient accumulation: {args.use_grad_acc}")
    logging.info(f"Gradient accumulation partial batch size: {args.grad_acc_partial_batch_size}")
    logging.info(f"Dataloaders num workers: {args.dataloaders_num_workers}")
    logging.info(f"Dataloaders persistent workers: {args.dataloaders_persistent_workers}")
    logging.info(f"Logging level: {args.logging_level}")
    logging.info("Configuration:")
    for key, value in config.items():
        if type(value) != dict:
            logging.info(f"{key}: {value}")
        else: # Nested dictionary
            logging.info(f"{key}:")
            for subkey, subvalue in value.items():
                logging.info(f"  {subkey}: {subvalue}")

    # For backwards compatibility with old configs and old run results in general
    if config["training"]["use_early_stopping"] and "early_stopping_patience" not in config["training"]:
        optimize_early_stopping_patience = True
        logging.info("Early stopping patience not found in config. Optimizing it instead.")
    else:
        optimize_early_stopping_patience = False
    if config["hpo"]["use_pruning"] and "pruning_n_startup_trials" not in config["hpo"]:
        config["hpo"]["pruning_n_startup_trials"] = 4
        logging.info("Pruning n_startup_trials not found in config. Setting it to default value of 4.")
    if config["hpo"]["use_pruning"] and "pruning_n_warmup_steps" not in config["hpo"]:
        config["hpo"]["pruning_n_warmup_steps"] = 5
        logging.info("Pruning n_warmup_steps not found in config. Setting it to default value of 5.")

    # Define data augmentation transforms for the training set
    if args.data_augmentations is None:
        if issubclass(model_class, Classifier):
            data_augmentation_transform = Utility.get_default_augmentations_classification()
        elif issubclass(model_class, ObjectDetector):
            data_augmentation_transform = Utility.get_default_augmentations_object_detection()
    else:
        with open(args.data_augmentations, "r") as f:
           augmentations_dict = yaml.load(f, Loader=yaml.FullLoader)
        data_augmentation_transform = Utility.parse_data_augmentations_yaml_dict(augmentations_dict)
    
    datasets = make_datasets(
        model_class=model_class,
        dataset=config["dataset"],
        classification_multiview=classification_multiview,
        combine_train_val=False,
        data_augmentation_transform=data_augmentation_transform,
        test=False
    )

    # Save transforms to run directory for traceability
    transforms_path = os.path.join(results_path, args.run_name, "transforms.txt")
    with open(transforms_path, "w") as f:
        f.write("Model transform:\n")
        f.write(str(model_class.get_transforms()))
        f.write("\nTrain augmentation transform:\n")
        f.write(str(data_augmentation_transform))

    # Hyperparameter optimization

    study = optimize_hyperparameters(
        model_class=model_class,
        classification_multiview=classification_multiview,
        n_trials=config["hpo"]["trials"],
        train_set=datasets["train"],
        validation_set=datasets["val"],
        results_path=results_path,
        run_name=args.run_name,
        dataset_name=config["dataset"],
        training_epochs=config["training"]["epochs"],
        use_early_stopping=config["training"]["use_early_stopping"],
        use_pruning=config["hpo"]["use_pruning"],
        dataloaders_num_workers=args.dataloaders_num_workers,
        dataloaders_persistent_workers=args.dataloaders_persistent_workers,
        gpu=args.gpu,
        use_grad_acc=args.use_grad_acc,
        grad_acc_partial_batch_size=args.grad_acc_partial_batch_size if args.use_grad_acc else None,
        allow_continue_run=args.allow_continue_run,
        optuna_sqlite_path=config["hpo"]["optuna_database_path"] if "optuna_database_path" in config["hpo"] else None,
        early_stopping_patience=config["training"]["early_stopping_patience"] if "early_stopping_patience" in config["training"] else None,
        batch_size=config["training"]["batch_size"] if "batch_size" in config["training"] else None, # Optional fixed batch size
        strict_ckpt_loading=config["training"]["strict_ckpt_loading"] if "strict_ckpt_loading" in config["training"] else True,
        pruning_n_startup_trials=config["hpo"]["pruning_n_startup_trials"] if "pruning_n_startup_trials" in config["hpo"] else None,
        pruning_n_warmup_steps=config["hpo"]["pruning_n_warmup_steps"] if "pruning_n_warmup_steps" in config["hpo"] else None,
        optimize_early_stopping_patience=optimize_early_stopping_patience
    )

    # study currently unused since all saving etc. happens continuously during the optimization


def setup():

    # Reproducibility
    Utility.init_and_seed_everything()

    # Command line arguments and config

    argument_parser = argparse.ArgumentParser(description="Hyperparameter-optimize a classifier for image classifiation on a specified dataset and output the results.")
    
    argument_parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    argument_parser.add_argument("--run_name", type=str, help="Unique run identifier (string) for this HPO run. Set manually.",
                                 required=True)
    argument_parser.add_argument("--gpu", type=int, required=True, help="GPU to use for training. Usually 0 or 1.")
    argument_parser.add_argument("--allow_continue_run", default=False, required=False, action="store_true", help="Flag. Whether to allow continuing an existing run with the same name.")
    argument_parser.add_argument("--data_augmentations", type=str, required=False, help="Path to a data augmentations yaml file. If None is provided, will use default augmentations.")
    argument_parser.add_argument("--use_grad_acc", action="store_true", help="Flag. Whether to use gradient accumulation.")
    argument_parser.add_argument("--grad_acc_partial_batch_size", required=False, type=int, default=4, help="Partial batch size for gradient accumulation.")
    argument_parser.add_argument("--dataloaders_num_workers", type=int, required=False, default=8, help="Number of workers for the dataloaders.")
    argument_parser.add_argument("--dataloaders_no_persistent_workers", action="store_false", dest="dataloaders_persistent_workers", help="Flag. Usually persistant workers are used for the dataloaders. This disables that.")
    argument_parser.add_argument("--logging_level", type=str, default="INFO", help="Logging level for the script (DEBUG, INFO, WARNING, ...)")
    args = argument_parser.parse_args()

    # Note that args.config is a path

    if not os.path.exists(args.config) or not args.config.endswith(".yaml"):
        raise ValueError("Invalid configuration file path. Please provide a valid YAML configuration file.")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    results_path = config["results_path"]
    
    # Check and create directories

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    run_dir_path = os.path.join(results_path, args.run_name)
    if not os.path.exists(run_dir_path):
        os.makedirs(run_dir_path)
    elif not args.allow_continue_run:
        # No real logging possible yet
        raise ValueError(f"Run directory {run_dir_path} already exists. Please choose a different run name or set \"allow_continue_run\" to true.")

    # Set up logging
    logging_level = args.logging_level
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.getLevelName(logging_level),
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # Log to both file and console (does not affect PyTorch, so you only get the "full" output in the console)
            # (Optuna logging is also getting redirected to the standard logger)
            logging.FileHandler(os.path.join(results_path, args.run_name, f"hpo_log.txt")),
            logging.StreamHandler(sys.stdout)
        ])

    # Save a copy of the config file to the run directory for traceability
    config_path = os.path.join(run_dir_path, "hpo_config.yaml")
    if args.allow_continue_run and os.path.exists(config_path):
        with open(config_path, "r") as f:
            old_config = yaml.load(f, Loader=yaml.FullLoader)
            # Compare dicts, save new config with a different name if they differ
            if old_config != config:
                logging.info("The configuration file has changed since the last run. Saving new config with a different name.")
                config_path = os.path.join(run_dir_path, f"hpo_config_{datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Log environment information
    logging.info(f"Versions: torch={torch.__version__}, torchvision={torchvision.__version__}, pytorch-lightning={pl.__version__}, numpy={np.__version__}, pandas={pd.__version__}")
    logging.info(f"Cuda available: {torch.cuda.is_available()}")

    return config, args


if __name__ == "__main__":
    config, args = setup() 
    main(config, args)