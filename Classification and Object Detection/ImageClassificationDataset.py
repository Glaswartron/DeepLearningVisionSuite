'''
This module contains the dataset class for single-view/multi-view binary/multiclass classification.
'''

import pandas as pd

import os
import PIL

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


class ImageClassificationSingleViewDataset(Dataset):
    """
    A custom dataset class for loading single-view data from a classification dataset.

    Args:
        data_path (str): The path to the data directory for the selected dataset (train, val, test).
        set_type (str): The type of dataset ("train", "val", "test", "train_val"). "train_val" indicates
            that the dataset should load both the training and validation samples.
        augmentation_transform (callable, optional): A transform to apply to the image on top of the model-specific transforms.
        
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the image and label at the given index.
    """

    def __init__(self, data_path, set_type, augmentation_transform=None):
        self.data_path = data_path
        self.augmentation_transform = augmentation_transform
        self.set_type = set_type

        # Read file/label csv
        if set_type != "train_val":
            set_path = os.path.join(data_path, f"{set_type}.csv")
            if not os.path.exists(set_path):
                raise ValueError(f"Set path {set_path} does not exist")
            set_df = pd.read_csv(set_path)
        else:
            train_path = os.path.join(data_path, "train.csv")
            val_path = os.path.join(data_path, "val.csv")
            if not os.path.exists(train_path):
                raise ValueError(f"Train path {train_path} does not exist")
            if not os.path.exists(val_path):
                raise ValueError(f"Val path {val_path} does not exist")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            set_df = pd.concat([train_df, val_df], ignore_index=True)

        # Get labels
        self.labels = set_df["label"].values

        # Get image filenames
        self.file_names = set_df["filename"].values

        self.multiclass = len(set(self.labels)) > 2
        self.num_classes = len(set(self.labels)) if self.multiclass else -1

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Args:
            idx (int): The index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the image and label.

        """
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load image
        image_path = os.path.join(self.data_path, self.file_names[idx])
        image = PIL.Image.open(image_path) #.convert("RGB") should not be used here, might have caused problems in the past but hard to say for sure. Maybe it didnt since images are already RGB.

        if self.augmentation_transform:
            image = self.augmentation_transform(image)

        # ToTensor is deprecated, this is the official replacement
        tensor_transforms = transforms.Compose([transforms.ToImage(),
                                                transforms.ToDtype(torch.float32, scale=True)])

        image = tensor_transforms(image)

        return image, label
    
    
class ImageClassificationMultiViewDataset(Dataset):
    """
    A custom dataset class for loading multi-view data from a image classification dataset.

    Args:
        data_path (str): The path to the data directory for the selected dataset (train, val, test).
        set_type (str): The type of dataset ("train", "val", "test").
        transform (callable, optional): A transform to apply to the image.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the image and label at the given index.
    """

    def __init__(self, data_path, set_type, augmentation_transform=None):
        self.data_path = data_path
        self.augmentation_transform = augmentation_transform
        self.set_type = set_type

        self.data = self._prepare_multi_view_dataset(data_path, set_type)

        self.multiclass = len(set(self.labels)) > 2
        self.num_classes = len(set(self.labels)) if self.multiclass else -1

    # Adapted from another student
    def __getitem__(self, idx):
        # Load and preprocess your data here
        # For example, loading an image using PIL and applying transformations

        tensor_transforms = transforms.Compose([transforms.ToImage(),
                                                transforms.ToDtype(torch.float32, scale=True)])

        view_set = self.data[idx]
        transformed_views = []
        for item in view_set:
            img_path = os.path.join(self.data_path, item[0]) # File subpath already contains "images/"
            img = PIL.Image.open(img_path) # .convert('RGB')
            # Transform images individually
            if self.augmentation_transform:
                img = self.augmentation_transform(img)
            img = tensor_transforms(img)
            transformed_views.append(img)
        # Multi-view case (return list of transformed views and label)
        # Pad transformed_views to length 6 if necessary
        padding_length = 6 - len(transformed_views)
        if padding_length > 0:
            # Choose a suitable padding strategy based on your data and task
            # Create a tensor of zeros with the same shape as the existing elements
            if len(transformed_views) > 0:
                # Assuming all elements in transformed_views have the same shape
                padding_shape = transformed_views[0].shape
                padding_value = torch.zeros(padding_shape)
            else:
                # Handle the case where there are no elements in transformed_views (potentially raise an error)
                raise ValueError("No elements found in transformed_views. Padding requires at least one element for reference.")

            final_views = transformed_views + [padding_value] * padding_length
        else:
            final_views = transformed_views

        label = max([item[1] for item in view_set])

        return final_views, torch.tensor(label)
    
    # Adapted from another student
    def _prepare_multi_view_dataset(self, data_path, set_type):
        if set_type in ["train", "val", "test"]:
            # Read file/label csv
            set_path = os.path.join(data_path, f"{set_type}.csv")
            if not os.path.exists(set_path):
                raise ValueError(f"Set path {set_path} does not exist")
            set_df = pd.read_csv(set_path)
        elif set_type == "train_val":
            train_path = os.path.join(data_path, "train.csv")
            val_path = os.path.join(data_path, "val.csv")
            if not os.path.exists(train_path):
                raise ValueError(f"Train path {train_path} does not exist")
            if not os.path.exists(val_path):
                raise ValueError(f"Val path {val_path} does not exist")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            set_df = pd.concat([train_df, val_df], ignore_index=True)

        # Get labels
        self.single_view_labels = set_df["label"].values

        # Get image filenames
        self.file_names = set_df["filename"].values

        multi_view_data = set() # Set eliminates duplicates
        for image_name in self.file_names:
            if 'num' in image_name:
                common_identifier = image_name.split('num')[0]
                multi_view_data.add(common_identifier)

        # Possible improvement: Maybe make better and more efficient
        final_list = [] # List of lists with all the different crops of all the different views
        for item in multi_view_data:
            sub_list_5 = []
            sub_list_20 = []
            sub_list_40 = []
            for file_name, label in zip(self.file_names, self.single_view_labels):
                if item in file_name and "tol5" in file_name:
                    sub_list_5.append((file_name, label))
                if item in file_name and "tol20" in file_name:
                    sub_list_20.append((file_name, label))
                if item in file_name and "tol40" in file_name:
                    sub_list_40.append((file_name, label))
            if(len(sub_list_5) > 0):
                final_list.append(sub_list_5)
            if(len(sub_list_20) > 0):
                final_list.append(sub_list_20)
            if(len(sub_list_40) > 0):
                final_list.append(sub_list_40)

        # Get multi-view labels
        self.labels = [max([item[1] for item in view_set]) for view_set in final_list]
        
        return final_list

    def __len__(self):
        return len(self.data)