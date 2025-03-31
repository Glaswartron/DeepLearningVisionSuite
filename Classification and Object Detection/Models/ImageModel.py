from abc import ABC, abstractmethod
import importlib
import torch.nn as nn


class ImageModel(ABC, nn.Module):

    def __init__(self):
        super().__init__()
        # Create and cache the transforms once on initialization for performance reasons
        self._transforms = self.get_transforms()

    @staticmethod
    @property
    @abstractmethod
    def name():
        '''
        The string name of this model for referencing it e.g. in config files or args.
        '''
        pass

    @staticmethod
    @abstractmethod
    def get_hyperparameters_optuna(trial):
        '''
        Takes an Optuna `Trial` object and returns a dictionary of hyperparameters (constructur arguments) for the model.
        The hyperparameters are sampled from the trial object using the suggest methods.
        '''
        pass

    # Possible improvement: Consider making this not static anymore to allow changing the
    #                       Resize resolution as a model hyperparameter
    @staticmethod
    @abstractmethod
    def get_transforms():
        '''
        Returns the transforms used for the model.
        Note: Internal model code should use self._transforms instead of calling this method for performance reasons.
        '''
        pass

    @classmethod
    @abstractmethod
    def from_hyperparameters(cls, hyperparameters, **kwargs):
        '''
        Creates a new instance of the model with the given model hyperparameters.
        `hyperparameters` has to be a dictionary with the keys matching those returned by `get_hyperparameters_optuna`.
        The dictionary may contain additional keys that are not part of the model hyperparameters,
        this ensures that you can simply pass the entire result hyperparameters of an hpo run.  
        Note that implementations can have additional arguments.
        '''
        pass

    @classmethod
    def get_model_class_by_name(cls, name):
        '''
        Returns the model class with the given name.
        '''
        import Models.DINOv2Classifier
        import Models.MaskRCNNObjectDetector
        import Models.EfficientNetV2Classifier
        if importlib.util.find_spec("mmpretrain") is not None and importlib.util.find_spec("mmcv") is not None:
            import Models.SwinTransformerV2Classifier # mmpretrain and mmcv have to be installed for this

        classes_to_check = set(cls.__subclasses__())
        while len(classes_to_check) > 0:
            current_class = classes_to_check.pop()
            if current_class.name == name:
                return current_class
            classes_to_check.update(current_class.__subclasses__())
        
        raise ValueError(f"No model with name {name} found.")