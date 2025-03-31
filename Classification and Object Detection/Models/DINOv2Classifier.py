'''
This module contains the DINOv2 classifier model for image classification.
It is a PyTorch module that uses Metas DINOv2 model as a fine-tunable feature extractor and adds a classifier head on top.
The module contains different DINOv2 models with different sizes (e.g. "vitb14" and "vits14")
and the base class for all DINOv2 classifier implementations.

DINOv2 is a self-supervised learning method for visual representation learning. 
It is based on the Vision Transformer (ViT) architecture.

For information about DinoV2 and how to use it see:
- https://github.com/facebookresearch/dinov2
- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md

For more concrete tutorials and examples see:
- https://github.com/facebookresearch/dinov2/pull/305
- https://kili-technology.com/data-labeling/computer-vision/dinov2-fine-tuning-tutorial-maximizing-accuracy-for-computer-vision-tasks
- https://purnasaigudikandula.medium.com/dinov2-image-classification-visualization-and-paper-review-745bee52c826
- https://blog.roboflow.com/how-to-classify-images-with-dinov2/
'''


import torch
import torchvision.transforms.v2 as transforms

from abc import abstractmethod

import logging

from Models.Classifier import Classifier # Derived from nn.Module
import Utility

# Possible improvement: Name could be mistaken for a vitb14 DINOv2 model (the subclass for which is just called DINOv2Classifier)
class BaseDINOv2Classifier(Classifier):
    '''
    A custom classifier model built on top of DINOv2.
    Uses DINOv2 model as a feature extractor and adds 
    a classifier head with linear layers and dropout.
    This is the base class for all DINOv2 models (with different sizes).
    '''

    ''' 
    Note: Since Classifer is an abstract class and this class contains/passes down 
          abstract methods, it is also an abstract class
    '''

    def __init__(self, multiview, n_layers, layer_sizes, layer_dropouts, multiclass=False, num_classes=-1):
        super().__init__(multiview)

        # Load DINOv2 model
        logging.info(f"Loading DINOv2 model with size {self.model_size}.")
        self.dino_model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{self.model_size}")

        # Create classifier layers
        num_features = self.dino_model.num_features if not self.multiview else self.dino_model.num_features * 6
        self.classifier = Classifier.make_classification_head(num_features, n_layers, layer_sizes, layer_dropouts, 
                                                              multiclass=multiclass, num_classes=num_classes)

    def forward(self, x):
        if not self.multiview:
            # x is a list of image tensors 
            x = torch.stack([self._transforms(xi) for xi in x], dim=0)
            x = self.dino_model(x)
            x = self.dino_model.norm(x) 
            x = self.classifier(x)
            return x
        else:
            # Pass each view through the DINOv2 model and concatenate the features for the classifier
            predictions = []
            for sample in x:
                view_embeddings = []
                for view in sample:
                    view = self._transforms(view)
                    view = view.unsqueeze(0)
                    view_embedding = self.dino_model(view)
                    view_embedding = self.dino_model.norm(view_embedding)
                    # view_embedding = self.norm(view_embedding)  # Normalize (another student does this with a LayerNorm(768))
                    view_embeddings.append(view_embedding)

                concatenated_embedding = torch.cat(view_embeddings, dim=1)
                predictions.append(self.classifier(concatenated_embedding))
            predictions = torch.cat(predictions, dim=0)
            return predictions

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
        Returns the transforms used for the DINOv2 model.
        '''
        return transforms.Compose([
            # The decision was made to avoid center crop transforms since
            # they caused problems with TorchScript in the past and could
            # crop out important parts of the image
            transforms.Resize((224, 224)),
            Utility.ScriptableNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
        return cls(multiview, n_layers, layer_sizes, layer_dropouts, multiclass=multiclass, num_classes=num_classes, **kwargs)
    
    @staticmethod
    @property
    @abstractmethod
    def model_size():
        '''
        The size/type of the DINOv2 model to use, e.g. "vitb14".
        '''
        pass


class DINOv2Classifier(BaseDINOv2Classifier):
    '''
    A custom classifier model built on top of DINOv2.
    Uses DINOv2 model as a feature extractor and adds 
    a classifier head with linear layers and dropout.
    This model uses the "vitb14" DINOv2 model.
    '''

    name = "DINOv2"
    model_size = "vitb14"
    supports_multiview = True

class DINOv2ClassifierSmall(BaseDINOv2Classifier):
    '''
    A custom classifier model built on top of DINOv2.
    Uses DINOv2 model as a feature extractor and adds 
    a classifier head with linear layers and dropout.
    This model uses the "vits14" DINOv2 model.
    '''

    name = "DINOv2Small"
    model_size = "vits14"
    supports_multiview = True

class DINOv2ClassifierLarge(BaseDINOv2Classifier):
    '''
    A custom classifier model built on top of DINOv2.
    Uses DINOv2 model as a feature extractor and adds 
    a classifier head with linear layers and dropout.
    This model uses the "vitl14" DINOv2 model.
    '''

    name = "DINOv2Large"
    model_size = "vitl14"
    supports_multiview = True