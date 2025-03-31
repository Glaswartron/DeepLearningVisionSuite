'''
This module contains the LightningImageClassifier class, which is a PyTorch Lightning module
for training a classifier for both binary or multiclass image classification.
The class provides the necessary methods for training, validation, and testing of
the classifier, configuring the optimizer and learning rate scheduler and logging
metrics.
An LightningImageClassifier wraps a PyTorch model which is passed on initialization
along with all hyperparameters for training.
'''
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import lightning.pytorch as pl
import torchmetrics

from sklearn.metrics import accuracy_score, f1_score, classification_report

import logging
import inspect


class LightningImageClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for training a classifier for image classification.
    """

    def __init__(self, 
                 model,
                 learning_rate,
                 multiclass=False,
                 num_classes=-1,
                 loss_criterion=nn.BCEWithLogitsLoss(),
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

        self.loss_criterion = loss_criterion

        self.learning_rate = learning_rate

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

        if optimizer_params is not None and "lr" in optimizer_params:
            logging.info("You don't need to specify the learning rate in the optimizer_params. Use the learning_rate parameter instead.")
            optimizer_params.pop("lr")

        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_object_params = learning_rate_scheduler_params
        self.learning_rate_scheduler_usage_params = learning_rate_scheduler_usage_params

        task = "binary" if not multiclass else "multiclass"
        average = "micro" if not multiclass else "macro"
        num_classes = num_classes if multiclass else None
        self.validation_f1 = torchmetrics.F1Score(task=task, average=average, num_classes=num_classes)

        self.multiclass = multiclass
        self.num_classes = num_classes

        self.test_predictions = []
        self.test_predictions_raw = []
        self.test_labels = []

    def forward(self, x):
        # Implementation required for torchscripting the module
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        if not self.multiclass:
            outputs = outputs.flatten()
            labels = labels.flatten().float()
        else:
            labels = labels.flatten()
        loss = self.loss_criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        if not self.multiclass:
            outputs = outputs.flatten()
            labels = labels.flatten().float()
        else:
            labels = labels.flatten()
        loss = self.loss_criterion(outputs, labels)
        # Calling log from validation_step automatically accumulates and logs at the end of the epoch
        self.log('val_loss', loss)

        if self.multiclass:
            labels = labels.flatten().long()

        # Calculate predictions
        preds = torch.sigmoid(outputs).round() #.cpu().numpy()

        if self.multiclass:
            _, preds = torch.max(preds, 1)

        # Update torchmetrics metric with results from this step
        self.validation_f1.update(preds, labels)

        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        if not self.multiclass:
            outputs = outputs.flatten() 
            labels = labels.flatten().float()
        else:
            labels = labels.flatten()
        
        loss = self.loss_criterion(outputs, labels)
        # Calling log from test_step automatically accumulates and logs at the end of the epoch
        self.log('test_loss', loss, batch_size=1)

        # Calculate predictions
        y_pred_raw = torch.sigmoid(outputs).cpu().numpy()
        if not self.multiclass:
            y_pred = y_pred_raw.round().astype(int).tolist()
        else:
            y_pred = y_pred_raw.argmax(axis=1).tolist()

        # Save predictions and labels to calculate metrics outside of this class
        labels = labels.cpu().numpy()
        self.test_predictions.extend(y_pred)

        self.test_labels.extend(labels)
        if not self.multiclass:
            self.test_predictions_raw.extend(y_pred_raw.tolist())

        return loss
    
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
        # Calculate and log F1 score
        f1 = self.validation_f1.compute()

        self.log("val_f1", f1, prog_bar=True)
        logging.info(f"Validation F1: {np.format_float_positional(f1.cpu().numpy(), precision=2)}")

        # Reset metrics for next epoch
        self.validation_f1.reset()

        # val_preds = np.array(self.val_predictions_epoch)
        # val_labels = np.array(self.val_labels_epoch)

        # # Calculate metrics
        # accuracy = accuracy_score(val_labels, val_preds)
        # f1 = f1_score(val_labels, val_preds)

        # # Log metrics
        # self.log("val_accuracy", accuracy, prog_bar=True)
        # self.log("val_f1", f1, prog_bar=True)
        # logging.debug(f"Accuracy: {accuracy}, F1: {f1}")
        # logging.info(classification_report(val_labels, val_preds))

        # # Reset predictions and labels for next epoch
        # self.val_predictions_epoch = []
        # self.val_labels_epoch = []
        
    
    # Note: dataloaders were moved outside of the class

    # def train_dataloader(self):
    #     return self.train_loader
    
    # def val_dataloader(self):
    #     return self.val_loader
    
    # def test_dataloader(self):
    #     return self.test_loader
    
    def on_train_epoch_start(self):
        if self.learning_rate_scheduler is not None:
            self.log("learning_rate", self.learning_rate_scheduler.get_last_lr()[0], prog_bar=True)