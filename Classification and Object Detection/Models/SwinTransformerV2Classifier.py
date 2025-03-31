'''
This module contains the SwinTransformerV2-Base classifier model for image classification.
It is a PyTorch module that uses the mmpretrain (openmmlab) implementation of SwinTransformerV2.
'''

import torch
import torch.nn as nn

import mmpretrain

import logging

from Models.Classifier import Classifier # Derived from nn.Module
import PretrainedModelTransforms


# Possible improvement: Multiview

class SwinTransformerV2Classifier(Classifier):
    '''
    A custom classifier model built on top of SwinTransformerV2-Base.
    '''

    name = "SwinTransformerV2"
    supports_multiview = False

    def __init__(self, avg_pool_before_clf_output_size, n_layers, layer_sizes, layer_dropouts):
        super().__init__()

        # Load SwinTransformerV2 model
        logging.info(f"Loading SwinTransformerV2-Base model.")
        mmpretrain_model = mmpretrain.get_model("swinv2-base-w16_in21k-pre_3rdparty_in1k-256px", pretrained=True)
        self.model = mmpretrain_model.backbone

        # Add global average pooling layer before classifier ("neck" in mmpretrain implementation)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(avg_pool_before_clf_output_size, avg_pool_before_clf_output_size))
        
        feature_dim = (avg_pool_before_clf_output_size ** 2) * 1024

        # Create classifier layers ("head")
        self.classifier = Classifier.make_classification_head(feature_dim, n_layers, layer_sizes, layer_dropouts)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x[0])
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    @staticmethod
    def get_hyperparameters_optuna(trial):
        '''
        Takes an Optuna trial object and returns a dictionary of hyperparameters (constructur arguments) for the model.
        The hyperparameters are sampled from the trial object using the suggest methods.
        '''

        avg_pool_before_clf_output_size = trial.suggest_int("avg_pool_before_clf_output_size", 1, 3)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_sizes = []
        layer_dropouts = []

        if n_layers == 1:
            return {
                "avg_pool_before_clf_output_size": avg_pool_before_clf_output_size,
                "n_layers": n_layers,
                "layer_sizes": [],
                "layer_dropouts": []
            }

        for i in range(n_layers - 1):
                layer_sizes.append(trial.suggest_int(f"layer_size_{i}", 64, 512))
                layer_dropouts.append(trial.suggest_float(f"dropout_{i}", 0.05, 0.70))

        return {
            "avg_pool_before_clf_output_size": avg_pool_before_clf_output_size,
            "n_layers": n_layers,
            "layer_sizes": layer_sizes,
            "layer_dropouts": layer_dropouts
        }

    @staticmethod
    def get_transforms():
        '''
        Returns the train, val and test transforms used for the EfficientNetV2 model.
        '''
        return PretrainedModelTransforms.SWINTRANSFORMERV2_TRANSFORMS
    
    @classmethod
    def from_hyperparameters(cls, multiview, hyperparameters, multiclass, num_classes=-1, **kwargs):
        '''
        Creates a new instance of the classifier model with the given model hyperparameters.
        `hyperparameters` has to be a dictionary with the keys matching those returned by `get_hyperparameters_optuna`.
        The dictionary may contain additional keys that are not part of the model hyperparameters,
        this ensures that you can simply pass the entire result hyperparameters of an hpo run.
        '''
        if multiview:
            raise ValueError("SwinTransformerV2Classifier does currently not support multiview.")

        avg_pool_before_clf_output_size = hyperparameters["avg_pool_before_clf_output_size"]
        n_layers = hyperparameters["n_layers"]
        layer_sizes = []
        layer_dropouts = []
        for i in range(n_layers - 1):
            layer_sizes.append(hyperparameters[f"layer_size_{i}"])
            layer_dropouts.append(hyperparameters[f"dropout_{i}"])
        
        return cls(avg_pool_before_clf_output_size, n_layers, layer_sizes, layer_dropouts, multiclass=multiclass, num_classes=num_classes)