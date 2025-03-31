'''
This module contains the abstract Classifier base class for image classification models.
It provides a common interface for all classifier models so they can be used generically in the classification pipeline.
'''

from abc import abstractmethod
import importlib
import torch.nn as nn

from Models.ImageModel import ImageModel


class Classifier(ImageModel):

    def __init__(self, multiview, multiclass=False, num_classes=-1):
        super().__init__()
        if multiview and not self.supports_multiview:
            raise ValueError("The model you are trying to instantiate as a multiview model \
                              does not support multiview")
        self.multiview = multiview
        self.multiclass = multiclass
        self.num_classes = num_classes

    @staticmethod
    @property
    @abstractmethod
    def supports_multiview():
        '''
        Whether this classifier model supports multi-view data.
        '''
        pass

    # Note: Difficult to make clf head into its own module/class because that causes 
    # backwards compatibility issues with existing ckpts.
    @staticmethod
    def make_classification_head(num_features, n_layers, layer_sizes, layer_dropouts, multiclass=False, num_classes=-1):
        classifier = nn.Sequential()
        if n_layers != 1:
            for i in range(n_layers):
                if i == 0:
                    classifier.add_module(f"linear_{i}", nn.Linear(num_features, layer_sizes[i]))
                elif i == n_layers - 1:
                    # If this is the last layer, add output neurons and no dropout and ReLU
                    if not multiclass:
                        classifier.add_module(f"linear_{i}", nn.Linear(layer_sizes[i-1], 1))
                    else:
                        classifier.add_module(f"linear_{i}", nn.Linear(layer_sizes[i-1], num_classes))
                    break
                else:
                    classifier.add_module(f"linear_{i}", nn.Linear(layer_sizes[i-1], layer_sizes[i]))
                classifier.add_module(f"dropout_{i}", nn.Dropout(layer_dropouts[i]))
                classifier.add_module(f"relu_{i}", nn.ReLU())
        else:
            # 1 layer means no dropout and ReLU, just a single linear layer
            if not multiclass:
                classifier.add_module("linear_0", nn.Linear(num_features, 1))
            else:
                classifier.add_module("linear_0", nn.Linear(num_features, num_classes))
        return classifier
    
    @classmethod
    def get_all_available_classifier_names(cls):
        '''
        Returns a list of all available classifier classes.
        '''
        import Models.DINOv2Classifier
        import Models.EfficientNetV2Classifier
        if importlib.util.find_spec("mmpretrain") is not None and importlib.util.find_spec("mmcv") is not None:
            import Models.SwinTransformerV2Classifier # mmpretrain and mmcv have to be installed for this

        classes_to_check = set(cls.__subclasses__())
        all_classes = []
        while len(classes_to_check) > 0:
            current_class = classes_to_check.pop()
            all_classes.append(current_class)
            classes_to_check.update(current_class.__subclasses__())

        return list(filter(lambda name_attr: isinstance(name_attr, str), map(lambda cls: cls.name, all_classes)))