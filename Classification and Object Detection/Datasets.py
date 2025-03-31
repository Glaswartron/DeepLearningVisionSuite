'''
This module contains the paths to the datasets used for binary/multiclass image classification and object detection tasks.
The keys in the datasets dictionary are the names of the datasets to be used in config files and arguments.
The directories are expected to have the following structure:
- "images" directory with the images
- "train.csv", "val.csv" and "test.csv" files with the image filenames and labels for classification and
  the COCO-style JSON for object detection
'''

import logging

from ImageClassificationDataset import ImageClassificationSingleViewDataset, ImageClassificationMultiViewDataset
from ObjectDetectionDataset import ObjectDetectionDataset
from Models.Classifier import Classifier
from Models.ObjectDetector import ObjectDetector


# Possible improvement: Move this to a json file or similar
datasets = {
    "dataset1": "Data/.../Dataset1",
    "dataset2": "Data/.../Dataset2",
    "dataset3": "Data/.../Dataset3"
}
 

# Possible improvement: Maybe remove dependence on model class?
def make_datasets(model_class, dataset, classification_multiview, combine_train_val,
                  data_augmentation_transform=None,
                  train=True, val=True, test=True):
    """
    Instantiates PyTorch datasets for training, validation, and testing based on the specified model class
    and dataset and using the given augmentations.
    Args:
        model_class (type): The class of the model (e.g., Classifier or ObjectDetector).
        dataset (str): The name of the dataset to be used.
        classification_multiview (bool): Whether to use multi-view classification dataset.
        combine_train_val (bool): Whether to combine the training and validation datasets into a combined training set.
        data_augmentation_transform (torchvision.transforms.Compose, optional): Data augmentation transforms to be applied to the training dataset.
        train (bool, optional): Whether to include the training dataset. Default is True.
        val (bool, optional): Whether to include the validation dataset. Default is True.
        test (bool, optional): Whether to include the test dataset. Default is True.
    Returns:
        dict: A dictionary containing the datasets for "train", "val", and "test". Also returns the
        final transforms used for training, validation, and testing ("train_transform", "val_transform", "test_transform").
        If combine_train_val is True, the "train" key will contain the combined training and validation datasets
        and the 'val' key will not be present.
    """

    if combine_train_val and (not train or not val):
        raise ValueError("Cannot combine training and validation datasets if either training or validation dataset is not requested.")

    dataset_path = datasets[dataset]

    result_datasets = {}

    if issubclass(model_class, Classifier):
        if not classification_multiview:
            logging.info("Using single-view classification dataset")
            dataset_class = ImageClassificationSingleViewDataset
        else:
            logging.info("Using multi-view classification dataset")
            dataset_class = ImageClassificationMultiViewDataset
    elif issubclass(model_class, ObjectDetector):
        logging.info("Using object detection dataset")
        dataset_class = ObjectDetectionDataset

    if not combine_train_val:
        if train:
            result_datasets["train"] = dataset_class(data_path=dataset_path, set_type="train",
                                                     augmentation_transform=data_augmentation_transform)
        if val:
            result_datasets["val"] = dataset_class(data_path=dataset_path, set_type="val",
                                                   augmentation_transform=None)
    else:
        # set_type = "train_val" means that the dataset will load both the training and validation samples
        # (see logic in dataset classes)
        train_dataset = dataset_class(data_path=dataset_path, set_type="train_val",
                                      augmentation_transform=data_augmentation_transform)
        result_datasets["train"] = train_dataset

    if test:
        result_datasets["test"] = dataset_class(data_path=dataset_path, set_type="test",
                                                augmentation_transform=None)

    return result_datasets
