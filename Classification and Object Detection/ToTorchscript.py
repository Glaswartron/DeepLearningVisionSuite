'''
This script takes a model checkpoint, its model class and a model hyperparameters csv
and creates a TorchScript file for deployment from the checkpointed model.
'''

import numpy as np
import pandas as pd

import os
import argparse
import logging

import torch
import torch.nn as nn
import torchvision

import lightning.pytorch as pl

from Models.ImageModel import ImageModel
from Models.Classifier import Classifier
from Models.ObjectDetector import ObjectDetector
from Models.MaskRCNNObjectDetector import MaskRCNNObjectDetector
from LightningImageClassifier import LightningImageClassifier
from LightningObjectDetector import LightningObjectDetector
import Datasets
import Utility


def main(settings):
    # Extract some constants from config for easier access
    model_class = ImageModel.get_model_class_by_name(settings.model)
    if settings.run_path is None:
        checkpoint_path = settings.checkpoint
        hyperparameters_path = settings.hyperparameters
        results_path = settings.results_path
    else:
        checkpoint_path, hyperparameters_path = Utility.get_ckpt_and_params_paths_from_dir(settings.run_path)
        results_path = settings.run_path

    # Load hyperparameters
    hyperparameters = pd.read_csv(hyperparameters_path).to_dict(orient="records")[0]

    # Load classifier from checkpoint

    datasets = Datasets.make_datasets(model_class, settings.dataset, False, False, None, True, False, False)
    if issubclass(model_class, ObjectDetector) or datasets["train"].multiclass:
        num_classes = datasets["train"].num_classes
    else:
        num_classes = -1

    logging.info("Loading model")
    # model = Utility.load_model_from_checkpoint(model_class, False, num_classes, hyperparameters, checkpoint_path, strict=settings.strict_ckpt_loading)
    checkpoint = torch.load(checkpoint_path)
    if issubclass(model_class, Classifier):
        model = model_class.from_hyperparameters(hyperparameters, multiview=settings.classification_multiview,
                                                 multiclass=datasets["train"].multiclass, num_classes=num_classes)
        weights = {k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()}
    elif issubclass(model_class, ObjectDetector):
        model = model_class.from_hyperparameters(hyperparameters, num_classes)
        # model = model.model
        weights = {k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(weights)

    # Model has to be put on the same device as it will be used on for now
    # Possible improvement: Make device an argument
    device = torch.device("cuda")
    model.to(device)

    # Set up Lightning module
    logging.info("Loading checkpoint into Lightning module")
    # model_lightning_module = lightning_module_class.load_from_checkpoint(
    #     checkpoint_path, model=base_model, batch_size=1, learning_rate=0, strict=settings.strict_ckpt_loading
    # )
    if issubclass(model_class, Classifier):
        model_lightning_module = LightningImageClassifier(model, 0, 
                                                               datasets["train"].multiclass, datasets["train"].num_classes)
    elif issubclass(model_class, ObjectDetector):
        model_lightning_module = LightningObjectDetector(model, 0)

    model_lightning_module.to(device)
    model_lightning_module.eval() # See https://github.com/pytorch/pytorch/issues/34948, doesnt solve the problem though

    # Save model as TorchScript

    logging.info("Saving TorchScript")

    # example_input has to be a list of possible inputs (here only one input example for every variant),
    # so if the models take lists as input, the input has to be a list of lists
    if issubclass(model_class, Classifier):
        if not settings.classification_multiview:
            example_input = [[torch.randn(3, 839, 1847)]]
            method = "trace"
        else:
            example_input = [[[torch.randn(3, 839, 1847) for _ in range(6)]]]
            method = "trace"
    elif(model_class, ObjectDetector):
        example_input = [[torch.randn(3, 3040, 4056)]]
        method = "script"

    checkpoint_filename = os.path.basename(checkpoint_path).removesuffix(".ckpt") # Use the same name as the checkpoint file
    model_lightning_module.to_torchscript(os.path.join(results_path, checkpoint_filename) + ".pt",
                                          example_inputs=example_input,
                                          method=method) # method="trace" is the only one that works for DINOv2
    # torch.jit.save(ts, os.path.join(RESULTS_PATH, f"DINOv2Classifier.pt"))

    logging.info("Done")


def setup(settings):

    # Reproducibility
    Utility.init_and_seed_everything()

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log environment information
    logging.info(f"Versions: torch={torch.__version__}, torchvision={torchvision.__version__}, pytorch-lightning={pl.__version__}, numpy={np.__version__}, pandas={pd.__version__}")
    logging.info(f"Cuda available: {torch.cuda.is_available()}")
    
    # Check and create directories
    if settings.run_path is None:
        results_path = settings.results_path
    else:
        results_path = settings.run_path
    if not os.path.exists(results_path):
        logging.info(f"Results directory does not exist. Creating it at {results_path}")
        os.makedirs(results_path)


def parse_args():
    available_models = Classifier.get_all_available_classifier_names() + ObjectDetector.get_all_available_object_detector_names()
    parser = argparse.ArgumentParser(description="Create a TorchScript file from a model checkpoint.")
    parser.add_argument("--model", type=str, required=True, choices=available_models, help="The model to create a TorchScript file for.")
    parser.add_argument("--classification_multiview", action="store_true", help="Flag. Whether to use a multiview model and dataset for classification.")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the model checkpoint to convert to TorchScript. Does not have to be provided if run_path is provided.")
    # Possible improvement: Consider making hyperparameters completely optional for models that dont have model hyperparameters
    parser.add_argument("--hyperparameters", type=str, required=False, help="Path to the hyperparameters csv file containing the model hyperparameters for the model. Does not have to be provided if run_path is provided.")
    parser.add_argument("--dataset", type=str, required=False, help="Name of the dataset on which the model was trained. Only required for object detection to infer the number of classes.")
    parser.add_argument("--run_path", type=str, required=False, help="Path to the directory containing the results of the hyperparameter optimization run. If provided, checkpoint and hyperparameters do not have to be provided.")
    parser.add_argument("--results_path", type=str, required=False, help="Path to the directory where the TorchScript file will be saved. Does not have to be provided if run_path is provided.")
    parser.add_argument("--no_strict_ckpt_loading", default=True, required=False, action="store_false", dest="strict_ckpt_loading", help="Disable strict loading for checkpoint.")
    
    args = parser.parse_args()

    # Enforce that either checkpoint, hyperparameters and results_path are provided or run_path is provided
    if args.run_path is None and (args.checkpoint is None or args.hyperparameters is None or args.results_path is None):
        parser.error("If run_path is not provided, checkpoint, hyperparameters and results_path have to be provided.")
    if args.run_path is not None and (args.checkpoint is not None or args.hyperparameters is not None or args.results_path is not None):
        parser.error("If run_path is provided, checkpoint, hyperparameters and results_path have to be omitted.")

    return args


if __name__ == "__main__":
    settings = parse_args()
    setup(settings)
    main(settings)