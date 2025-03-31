'''
This module contains the data transforms required for the pretrained models used in image classification.
These are exported as constant dictionaries with keys "train", "val" and "test".
The transforms can be used in the Datasets for the classification task.
The transforms for the individual models are taken from the model documentation and/or examples on the internet.
'''

import PIL

import torchvision.transforms.v2 as transforms

# Update January: This module is mostly obsolete with the inclusion of model transforms
# into the model modules themselves.

''' 
Possible improvement: Consider adding padding similar to https://github.com/legel/dinov2/blob/main/notebooks/classification.ipynb.
                      See https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md for info about what image sizes are supported.
'''
DINOV2_TRANSFORMS = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)), # originally 256
        # transforms.Resize((448, 448)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)), # originally 256
        # transforms.Resize((448, 448)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)), # originally 256
        # transforms.Resize((448, 448)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

MASKRCNN_TRANSFORMS = {
    'train': transforms.Compose([
        # Inverted dimensions: first height, second width
        transforms.Resize((800, 1067), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((800, 1067), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((800, 1067), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ])
}

EFFICIENTNETV2_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize((384, 384), interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((384, 384), interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((384, 384), interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

SWINTRANSFORMERV2_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}