'''
This module contains the dataset class for object detection datasets.
'''

import os
import json
import PIL

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
# from torchvision.transforms import transforms
import torchvision.transforms.v2 as transforms
# import torchvision.datapoints as datapoints

class ObjectDetectionDataset(Dataset):   
    def __init__(self, data_path, set_type, augmentation_transform):
        self.data_path = data_path
        self.set_type = set_type # train, val, test
        self.augmentation_transform = augmentation_transform
        
        images_path = os.path.join(data_path, "images")
        coco_path = os.path.join(data_path, f"{self.set_type}.json")

        # Note that train_val for object detection is just its own 
        # dataset json file and needs no custom processing here

        with open(coco_path, "r") as f:
            complete_coco_dict = json.load(f) # Relevant keys are "images", "annotations", "categories"
            image_ids = [img_dict["id"] for img_dict in complete_coco_dict["images"]]
            self.image_paths = [os.path.join(images_path, os.path.basename(img_dict["file_name"])) for img_dict in complete_coco_dict["images"]]
            self.coco_categories_dict = complete_coco_dict["categories"]
            # First get all annotations + image ids, then ensure that all annotations are in the same order as the images in terms of ids
            all_annotations = complete_coco_dict["annotations"]
            image_ids_to_annotations = {}
            for id in image_ids:
                image_ids_to_annotations[id] = [annotation for annotation in all_annotations if annotation["image_id"] == id]
            # List of lists of dicts, each list contains all annotations dictionaries for one image
            self.coco_annotations_per_image = [image_ids_to_annotations[id] for id in image_ids]

        # Link specific categories to supercategories, supercategories will be used as labels
        supercategories = [category["supercategory"] for category in self.coco_categories_dict]
        supercategories_ordered_no_dups = []
        [supercategories_ordered_no_dups.append(supercategory) for supercategory in supercategories if supercategory not in supercategories_ordered_no_dups]
        self.supercategories_to_ids = {supercategories_ordered_no_dups: i for i, supercategories_ordered_no_dups in enumerate(supercategories_ordered_no_dups)}
        self.category_id_to_supercategory_id = {}
        for category in self.coco_categories_dict:
            self.category_id_to_supercategory_id[category["id"]] = self.supercategories_to_ids[category["supercategory"]]

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.image_paths[idx]
        annotations = self.coco_annotations_per_image[idx]

        image = PIL.Image.open(img_path)

        # Taken from another students code 

        # transform image to tensor
        image = transforms.PILToTensor()(image).float() / 255
        # targets
        boxes = [] # List of bounding boxes for each annotation/object
        labels = [] # List of labels (category_id) for each annotation/object
        masks = [] # List of grayscale masks for each annotation/object
        # iterate through all annotations of one image
        for annotation in annotations:
            # get annotation data for bbox
            bbox = torch.FloatTensor(annotation['bbox'])
            # change format (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
            bbox[2:] = bbox[:2] + bbox[2:]
            # append bbox to target of this image
            boxes.append(bbox)
            # get category of annotation. Benedikt: Use supercategory
            # This is actually really important. Were using the supercategories as classification targets,
            # not the normal categories, because they are too specific.
            # labels.append(annotation['category_id'])
            labels.append(self.category_id_to_supercategory_id[annotation['category_id']])
            # generate segmentation mask
            masked = PIL.Image.new('L', (image.shape[2], image.shape[1]))
            for mask in annotation['segmentation']:
                PIL.ImageDraw.Draw(masked).polygon(mask, outline=1, fill=1)
            # transform mask to the right format
            masked = transforms.PILToTensor()(masked)
            masked = torch.squeeze(masked)
            # append mask
            masks.append(masked)

        # combine alls bboxes, labels and masks of the image and add them as label
        boxes = torch.stack(boxes)
        labels = torch.LongTensor(labels)
        masks = torch.stack(masks)

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        masks = tv_tensors.Mask(masks)

        if self.augmentation_transform:
            image, boxes, masks = self.augmentation_transform(image, boxes, masks)
        
        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["labels"] = labels
        target["image_id"] = idx # Possible improvement: Not optimal I guess, doesnt necessarily match the original image ids?
        target["area"] = areas
        target["iscrowd"] = torch.zeros((len(annotations),), dtype=torch.int64)

        return image, target

    @property
    def num_classes(self):
        return len(self.supercategories_to_ids)

    def __len__(self):
        return len(self.image_paths)