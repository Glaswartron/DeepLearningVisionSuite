'''
This module contains the LightningObjectDetector class, which is a PyTorch Lightning module
for training an object detection model on an image dataset.
The class provides the necessary methods for training, validation, and testing of
the object detector, configuring the optimizer and learning rate scheduler and logging
metrics.
An LightningObjectDetector wraps a PyTorch model which is passed on initialization
along with all hyperparameters for training.
'''
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

import lightning.pytorch as pl

import torchmetrics.detection.mean_ap

import logging
import inspect


class LightningObjectDetector(pl.LightningModule):
    """
    A PyTorch Lightning module for training an object detection model.
    """

    def __init__(self, 
                 model,
                 learning_rate,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 learning_rate_scheduler=None,
                 learning_rate_scheduler_params={},
                 learning_rate_scheduler_usage_params={
                     "interval": "epoch",
                     "frequency": 1,
                     "monitor": "val_loss"
                 }):
        
        super().__init__()

        self.model = model

        # Hyperparameters

        self.learning_rate = learning_rate

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

        if optimizer_params is not None and "lr" in optimizer_params:
            logging.info("You don't need to specify the learning rate in the optimizer_params. Use the learning_rate parameter instead.")
            optimizer_params.pop("lr")

        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_object_params = learning_rate_scheduler_params
        self.learning_rate_scheduler_usage_params = learning_rate_scheduler_usage_params

        self.validation_mAP = torchmetrics.detection.mean_ap.MeanAveragePrecision()

        self.test_predictions = []
        self.test_targets = []

    # Implementation and type annotation required for torchscripting the module
    def forward(self, x: List[torch.Tensor]):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets) # In train mode, model outputs loss dict
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # PyTorch Lightning puts the model into eval mode for validation
        predictions = self.model(images)
        self.validation_mAP.update(predictions, targets)

        self.model.train()
        loss_dict = self.model(images, targets) # In train mode, model outputs loss dict
        loss = sum(loss for loss in loss_dict.values())
        # Calling log during validation will automatically accumulate and log at the end of the epoch
        self.log('val_loss', loss, prog_bar=True)
        self.model.eval()

        return loss
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images) # In eval mode, model outputs predictions

        self.model.train()
        loss_dict = self.model(images, targets) # In train mode, model outputs loss dict
        loss = sum(loss for loss in loss_dict.values())
        # Calling log during testing will automatically accumulate and log at the end of the epoch
        self.log('test_loss', loss, prog_bar=True)
        self.model.eval()

        # Put predictions on CPU, detach them from comp graph, add them to list
        predictions = [{key: value.cpu().detach() if isinstance(value, torch.Tensor) else value for key, value in prediction.items()} for prediction in predictions]
        self.test_predictions.extend(predictions)

        # Put targets on CPU, detach them from comp graph, add them to list
        targets = [{key: value.cpu().detach() if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in targets]
        self.test_targets.extend(targets)

        return None
    
    def configure_optimizers(self):
        # If optimizer is a class, instantiate it
        if inspect.isclass(self.optimizer):
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate, **self.optimizer_params)

        # If learning rate scheduler is a class, instantiate it
        if self.learning_rate_scheduler is not None:
            if inspect.isclass(self.learning_rate_scheduler):
                # Assure that self.learning_rate_scheduler is a usable object
                self.learning_rate_scheduler = self.learning_rate_scheduler(self.optimizer, **self.learning_rate_scheduler_object_params)

        return_dict = {
            "optimizer": self.optimizer,
        }
        if self.learning_rate_scheduler is not None:
            return_dict["lr_scheduler"] = {
                "scheduler": self.learning_rate_scheduler,
                **self.learning_rate_scheduler_usage_params
            }

        return return_dict
    
    def on_validation_epoch_end(self):
        # Evaluate and log mAP
        mAP_metrics_epoch = self.validation_mAP.compute()
        mAP = mAP_metrics_epoch["map"]
        self.log("val_mAP", mAP, prog_bar=True)
        logging.info(f"Validation mAP: {np.format_float_positional(mAP.cpu().numpy(), precision=2)}")
    
    def on_train_epoch_start(self):
        # Log learning rate for every train epoch
        if self.learning_rate_scheduler is not None:
            self.log("learning_rate", self.learning_rate_scheduler.get_last_lr()[0], prog_bar=True)