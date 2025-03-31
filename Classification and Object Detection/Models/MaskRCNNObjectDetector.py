import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F

from typing import List, Dict, Optional
import logging

from Models.ObjectDetector import ObjectDetector # Derived from nn.Module
import Utility


class MaskRCNNObjectDetector(ObjectDetector):
    '''
    A custom object detection model built on top of Mask R-CNN.
    '''

    name = "MaskRCNN"

    def __init__(self, num_classes, trainable_backbone_layers, hidden_layer_size):
        super().__init__()

        logging.info("Loading Mask R-CNN model.")

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            trainable_backbone_layers=trainable_backbone_layers
        )

        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        # And replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer_size,
            num_classes
        )

    # Type annotations required for torchscripting the module
    def forward(self, images: list[torch.Tensor], targets:Optional[List[Dict[str, torch.Tensor]]]=None):
        if not torch.jit.is_scripting():
            if targets is None:
                images = [self._transforms(image) for image in images]
            else:
                images, targets = zip(*[self._transforms(i, t) for i, t in zip(images, targets)])
        else:
            # Resize transform has problems with scripting
            images = [F.resize(image, (800, 1067)) for image in images]

        if self.model.training:
            pred = self.model(images, targets)
        else:
            pred = self.model(images)

        # Transform the predictions back to the original image size
        if not torch.jit.is_scripting():
            # Return losses during training, detections during inference
            if not self.model.training:
                for i in range(len(pred)):
                    # Possible improvement: Infer original image size from input images (causes errors during scripting if done here dynamically)
                    pred[i]["boxes"] = Utility.resize_boxes(pred[i]["boxes"], (800, 1067), (3040, 4056))
        else:
            # Model returns a (losses, detections) tuple during scripting
            pred[1][0]["boxes"] = Utility.resize_boxes(pred[1][0]["boxes"], (800, 1067), (3040, 4056))
        # Possible improvement: Also resize masks if needed, currently returned masks are not valid anyways

        return pred

    @staticmethod
    def get_hyperparameters_optuna(trial):
        '''
        Takes an Optuna trial object and returns a dictionary of hyperparameters (constructur arguments) for the model.
        The hyperparameters are sampled from the trial object using the suggest methods.
        '''

        trainable_backbone_layers = trial.suggest_int("trainable_backbone_layers", 0, 5)
        hidden_layer_size = trial.suggest_int("hidden_layer_size", 64, 512)

        return {
            "trainable_backbone_layers": trainable_backbone_layers,
            "hidden_layer_size": hidden_layer_size,
        }

    @staticmethod
    def get_transforms():
        '''
        Returns the train, val and test transforms used for the Mask R-CNN model.
        '''
        return transforms.Resize((800, 1067))
    
    @classmethod
    def from_hyperparameters(cls, hyperparameters, num_classes, **kwargs):
        '''
        Creates a new instance of the model with the given model hyperparameters.
        The dictionary may contain additional keys that are not part of the model hyperparameters,
        this ensures that you can simply pass the entire result hyperparameters of an hpo run.
        '''
        trainable_backbone_layers = hyperparameters["trainable_backbone_layers"]
        hidden_layer_size = hyperparameters["hidden_layer_size"]
        return cls(num_classes, trainable_backbone_layers, hidden_layer_size)