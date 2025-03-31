from abc import ABC, abstractmethod
import torch.nn as nn

from Models.ImageModel import ImageModel


class ObjectDetector(ImageModel):

    @classmethod
    def get_all_available_object_detector_names(cls):
        '''
        Returns a list of all available object detector classes.
        '''
        import Models.MaskRCNNObjectDetector

        classes_to_check = set(cls.__subclasses__())
        all_classes = []
        while len(classes_to_check) > 0:
            current_class = classes_to_check.pop()
            all_classes.append(current_class)
            classes_to_check.update(current_class.__subclasses__())

        return list(filter(lambda name_attr: isinstance(name_attr, str), map(lambda cls: cls.name, all_classes)))