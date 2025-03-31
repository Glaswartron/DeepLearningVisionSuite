'''
This module contains the EfficientNetV2-M classifier model for image classification.
It is a PyTorch module that uses the torchvision implementation of EfficientNetV2 and adds
a custom classifier on top of it to output a single value for binary classification.

This module uses the torchvision implementation of EfficientNetV2-M and its default weights.
'''

import torch
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import torchvision.transforms.v2 as transforms

import logging

from Models.Classifier import Classifier # Derived from nn.Module


# Possible improvement: Multiview


class EfficientNetV2Classifier(Classifier):
    '''
    A custom classifier model built on top of EfficientNetV2-M.
    '''

    name = "EfficientNetV2"
    supports_multiview = False

    def __init__(self, multiview, n_layers, layer_sizes, layer_dropouts, multiclass=False, num_classes=-1):
        super().__init__(multiview)

        # Load EfficientNetV2 model
        logging.info(f"Loading EfficientNetV2-M model.")
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

        # Create classifier layers, replacing the default classifier with a custom one
        num_features = self.model.classifier[1].in_features
        self.model.classifier = Classifier.make_classification_head(num_features, n_layers, layer_sizes, layer_dropouts,
                                                                    multiclass=multiclass, num_classes=num_classes)

    def forward(self, x):
        x = torch.stack([self._transforms(xi) for xi in x], dim=0)
        x = self.model(x)
        return x

    @staticmethod
    def get_hyperparameters_optuna(trial):
        '''
        Takes an Optuna trial object and returns a dictionary of hyperparameters (constructur arguments) for the model.
        The hyperparameters are sampled from the trial object using the suggest methods.
        '''

        n_layers = trial.suggest_int("n_layers", 1, 3)

        hyperparameters = {
            "n_layers": n_layers
        }

        if n_layers == 1:
            return hyperparameters

        for i in range(n_layers - 1):
                hyperparameters[f"layer_size_{i}"] = trial.suggest_int(f"layer_size_{i}", 64, 512)
                hyperparameters[f"dropout_{i}"] = trial.suggest_float(f"dropout_{i}", 0.05, 0.70)

        return hyperparameters

    @staticmethod
    def get_transforms():
        '''
        Returns the train, val and test transforms used for the EfficientNetV2 model.
        '''
        return transforms.Compose([
            # The decision was made to avoid center crop transforms since
            # they caused problems with TorchScript in the past and could
            # crop out important parts of the image
            transforms.Resize((480, 480)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @classmethod
    def from_hyperparameters(cls, hyperparameters, multiview, multiclass, num_classes=-1, **kwargs):
        '''
        Creates a new instance of the classifier model with the given model hyperparameters.
        `hyperparameters` has to be a dictionary with the keys matching those returned by `get_hyperparameters_optuna`.
        The dictionary may contain additional keys that are not part of the model hyperparameters,
        this ensures that you can simply pass the entire result hyperparameters of an hpo run.
        '''
        n_layers = hyperparameters["n_layers"]
        layer_sizes = []
        layer_dropouts = []
        for i in range(n_layers - 1):
            layer_sizes.append(hyperparameters[f"layer_size_{i}"])
            layer_dropouts.append(hyperparameters[f"dropout_{i}"])
        
        return cls(multiview, n_layers, layer_sizes, layer_dropouts, multiclass, num_classes)