'''
This script generates GradCAM visualizations for a set of images using a given classifier checkpoint.
Saves the visualizations in the same folder as the images.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import logging
import argparse
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

import lightning.pytorch as pl
from lightning.pytorch import seed_everything

from Models.Classifier import Classifier
import Utility as Utility

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def main(settings):
    # Extract some constants from config for easier access
    classifier = Classifier.get_classifier_class_by_name(settings.classifier)
    if settings.run_path is None:
        checkpoint_path = settings.checkpoint
        hyperparameters_path = settings.hyperparameters
    else:
        checkpoint_path, hyperparameters_path = Utility.get_ckpt_and_params_paths_from_dir(settings.run_path)

    device = f"cuda:{settings.gpu}"

    # Load hyperparameters
    hyperparameters = pd.read_csv(hyperparameters_path).to_dict(orient="records")[0]

    # Load classifier from checkpoint
    logging.info("Loading classifier")
    model = Utility.load_model_from_checkpoint(classifier, hyperparameters, checkpoint_path, strict=settings.strict_ckpt_loading)

    result_images = []

    # Possible improvement: Use batches instead of iterating over the images?
    for image_path in os.listdir(settings.images):
        # Load image as tensor
        input_image = Image.open(os.path.join(settings.images, image_path))
        preprocess = transforms.Compose([
            transforms.Resize((224, 224))
            # transforms.CenterCrop(224)
        ])
        input_image = preprocess(input_image)
        input_image = np.float32(input_image) / 255
        input_tensor = preprocess_image(input_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        model.eval()
        model.to(device)

        # Get prediction of model for image
        prediction = model(input_tensor.to(device))
        prediction = int(np.round(torch.sigmoid(prediction).squeeze().cpu().detach().numpy()))

        # Target layer as recommended for Vision Transformers, Possible improvement: other target layers for other models
        target_layers = [model.dino_model.blocks[-1].norm1]
        input = input_tensor

        # GradCAM

        grad_cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        score_cam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

        targets = None

        smooth = settings.smooth
        grayscale_grad_cam = grad_cam(input_tensor=input, targets=targets, aug_smooth=smooth)
        grayscale_grad_cam = grayscale_grad_cam[0, :]
        grayscale_score_cam = score_cam(input_tensor=input, targets=targets, aug_smooth=smooth)
        grayscale_score_cam = grayscale_score_cam[0, :]

        # Build the visualization

        visualization_grad_cam = show_cam_on_image(input_image, grayscale_grad_cam, use_rgb=True)
        visualization_score_cam = show_cam_on_image(input_image, grayscale_score_cam, use_rgb=True)
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].imshow(input_image)
        ax[0].title.set_text("Original image")
        ax[0].axis("off")
        ax[1].imshow(visualization_grad_cam)
        ax[1].title.set_text(f"Predicted label: {prediction}")
        ax[1].axis("off")
        ax[2].imshow(visualization_score_cam)
        ax[2].title.set_text(f"ScoreCAM")
        ax[2].axis("off")

        result_images.append(fig)

    # Save the visualizations in the same folder as the images
    for i, fig in enumerate(result_images):
        filename = f"GradCAM_{i}"
        if smooth:
            filename += "_smoothed"
        fig.savefig(os.path.join(settings.images, filename + ".png"))

    logging.info("Done")


def reshape_transform(tensor):
    result = tensor[:, 1:  , :].reshape(tensor.size(0), 224 // 14, 224 // 14, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def setup():
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GradCAMs for a set of images using a given classifier checkpoint.")
    parser.add_argument("--classifier", type=str, required=True, choices=Classifier.get_all_available_classifier_names(), help="The classifier model to use for generating GradCAM images.")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the classifier checkpoint to generate GradCAM images with. Does not have to be provided if run_path is provided.")
    # Possible improvement: Consider making hyperparameters completely optional for models that dont have model hyperparameters
    parser.add_argument("--hyperparameters", type=str, required=False, help="Path to the hyperparameters csv file containing the model hyperparameters for the classifier. Does not have to be provided if run_path is provided.")
    parser.add_argument("--run_path", type=str, required=False, help="Path to the directory containing the results of the hyperparameter optimization run. If provided, checkpoint and hyperparameters do not have to be provided.")
    parser.add_argument("--images", type=str, required=True, help="The folder of images to run GradCAM for.")
    parser.add_argument("--smooth", type=bool, default=False, required=False, help="Whether to apply smoothing to the GradCAM output.")
    parser.add_argument("--gpu", type=int, required=True, help="The GPU to use for infering the model.")
    parser.add_argument("--no_strict_ckpt_loading", default=True, required=False, action="store_false", dest="strict_ckpt_loading", help="Disable strict loading for checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    settings = parse_args()
    setup()
    main(settings)