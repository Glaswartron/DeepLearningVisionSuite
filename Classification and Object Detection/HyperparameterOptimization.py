'''
This module contains the hyperparameter optimization functionality for all image classification and object detection models.
It uses Optuna to optimize hyperparameters for a given model class (e.g. DINOv2, Mask R-CNN) and dataset.
The main function is optimize_hyperparameters(), which runs the hyperparameter optimization and returns the Optuna study object.

Logs to an Optuna SQLite database at the provided results_path and TensorBoard in the "results_path/run_name/TensorBoard" subdirectory.
'''

import pandas as pd

import lightning.pytorch as pl

from torch.utils.data import DataLoader

import os
import logging

from TrainAndTestModel import train_model
from Models.Classifier import Classifier
from Models.ObjectDetector import ObjectDetector
import Utility

import optuna
import optuna.integration
import optuna.distributions


def _objective(trial, model_class, classification_multiview,
               train_set, validation_set, results_path, run_name, dataset_name,
               training_epochs, use_early_stopping, early_stopping_patience, use_pruning,
               dataloaders_num_workers, dataloaders_persistent_workers,
               gpu, use_grad_acc, grad_acc_partial_batch_size, batch_size, strict_ckpt_loading, optimize_early_stopping_patience):
    """
    Objective function for Optuna hyperparameter optimization.
    Trains and evaluates a model on the given datasets.
    """

    logging.info(f"Trial {trial.number}, Run name: {run_name}, Dataset: {dataset_name}, Model: {model_class.name}, GPU: {gpu}")

    ''' Dataloaders '''

    if batch_size is None:
        # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", ["8", "16", "32", "64"])
        # String values are necessary so it gets shown correctly in Optuna dashboard
        # Update: Doesnt work
        # See relevant PR: https://github.com/optuna/optuna-dashboard/pull/993
        # Fixed and merged, but not included yet in the latest release...
        batch_size = int(batch_size)

    if issubclass(model_class, Classifier):
        replacement = trial.suggest_categorical("sampler_replacement", [True, False])

    ''' Hyperparameters for model itself '''

    '''
    Requires model_class to implement the get_hyperparameters() static method,
    samples hyperparameters using the Optuna trial object and returns them as a dictionary.
    '''
    model_hyperparameters = model_class.get_hyperparameters_optuna(trial)

    ''' Hyperparameters for training the model. Note that these are the same for all models. '''
    # --> See Optuna examples repo with PyTorch examples

    optimizer_str = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp", "AdamW"])

    # No ReduceLROnPlateau so that LR scheduling doesnt depend on validation loss
    learning_rate_scheduler_str = trial.suggest_categorical("learning_rate_scheduler",
                                                            ["StepLR", "ExponentialLR", "CosineAnnealingLR"])

    # Here, learning rate is treated indepentently of optimizer_params
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 0.1, log=True)

    # Naming convention for optimizer and lr scheduler params:
    # optimizer_[param], lr_scheduler_[param]. Name here and in Optuna should match.
    # Only add "optimizer_" and "lr_scheduler_" prefix, nothing else.
    # Will get parsed correctly later.

    # Default for optimizer weight decay is 0
    optimizer_use_weight_decay = trial.suggest_categorical("optimizer_use_weight_decay", [True, False])
    optimizer_weight_decay = trial.suggest_float("optimizer_weight_decay", 1e-6, 1e-2, log=True) if optimizer_use_weight_decay else 0
    optimizer_params = {
        "optimizer_weight_decay": optimizer_weight_decay
    }

    if learning_rate_scheduler_str == "StepLR":
        learning_rate_scheduler_params = {
            "lr_scheduler_step_size": trial.suggest_int("lr_scheduler_step_size", 1, 12),
            "lr_scheduler_gamma": trial.suggest_float("lr_scheduler_gamma", 0.05, 0.95)
        }
    elif learning_rate_scheduler_str == "ExponentialLR":
        learning_rate_scheduler_params = {
            "lr_scheduler_gamma": trial.suggest_float("lr_scheduler_gamma", 0.05, 0.95)
        }
    elif learning_rate_scheduler_str == "CosineAnnealingLR":
        learning_rate_scheduler_params = {
            # LR scheduler steps every epoch, so the max number of steps is the number of epochs
            "lr_scheduler_T_max": training_epochs
        }

    # Callbacks for PyTorch Lightning Trainer (Early Stopping and Optuna Pruning)

    callbacks = []

    ''' 
    Early stopping and Optuna pruning
    Early stopping looks at the training target = val_loss. Pruning looks at the HPO target = val_f1 or val_mAP.
    '''
    if use_early_stopping and optimize_early_stopping_patience:
        # Not usually included in HPO anymore, but option still exists for backwards compatibility
        early_stopping_patience = trial.suggest_int("early_stopping_patience", 3, 14)
    if use_pruning:
        monitor = "val_f1" if issubclass(model_class, Classifier) else "val_mAP"
        pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor=monitor)
        callbacks.append(pruning_callback)

    # Checkpoint callback. Saves the best model checkpoint based on the validation F1 score.
    if issubclass(model_class, Classifier):
        monitor = "val_f1"
        # Filename e.g. "temp_Trial_0_epoch=10_val_f1=0.85.ckpt"
        # Possible improvement: This is maybe not optimal because filename contains two dots
        filename = f"temp_Trial_{trial.number}_{{epoch}}_{{val_f1}}"
    elif issubclass(model_class, ObjectDetector):
        monitor = "val_mAP"
        filename = f"temp_Trial_{trial.number}_{{epoch}}_{{val_mAP}}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode="max",
        dirpath=os.path.join(results_path, run_name),
        filename=filename,
        save_top_k=1
    )
    callbacks.append(checkpoint_callback)

    # Define a TensorBoard logger with a custom run directory and name
    tensorboard_run_dir = os.path.join(results_path, run_name, f"TensorBoard")
    tensorboard_logger = pl.loggers.TensorBoardLogger(tensorboard_run_dir, name=f"Trial_{trial.number}")

    # Make dict with all hyperparameters. Has to match model_class.from_hyperparameters().
    hyperparameters = {
        **model_hyperparameters,
        "optimizer": optimizer_str,
        "learning_rate_scheduler": learning_rate_scheduler_str,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_use_weight_decay": optimizer_use_weight_decay,
        **optimizer_params,
        **learning_rate_scheduler_params
    }
    if issubclass(model_class, Classifier):
        hyperparameters["dataloaders_sampler_replacement"] = replacement

    # Log hyperparameters to TensorBoard
    tensorboard_logger.log_hyperparams(hyperparameters)

    # Train the model
    logging.info(f"Trial {trial.number}: Training model with hyperparameters: {hyperparameters}")
    model_lightning_module, trainer = train_model(
        model_class=model_class,
        classification_multiview=classification_multiview,
        hyperparameters=hyperparameters,
        epochs=training_epochs,
        gpu=gpu,
        train_dataset=train_set,
        validation_dataset=validation_set,
        dataloaders_num_workers=dataloaders_num_workers,
        dataloaders_persistent_workers=dataloaders_persistent_workers,
        use_grad_acc=use_grad_acc,
        grad_acc_partial_batch_size=grad_acc_partial_batch_size,
        lightning_callbacks=callbacks,
        early_stopping_patience=early_stopping_patience,
        tensorboard_logger=tensorboard_logger
    )

    # Run validation again for the best checkpoint
    collate_function = Utility.classification_dataloader_collate_function if issubclass(model_class, Classifier) else Utility.object_detection_dataloader_collate_function
    val_dataloader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=dataloaders_num_workers,
        persistent_workers=dataloaders_persistent_workers,
        collate_fn=collate_function
    )
    metrics = trainer.validate(model_lightning_module, dataloaders=val_dataloader, ckpt_path="best")
    if issubclass(model_class, Classifier):
        val_score = metrics[0]["val_f1"]
    elif issubclass(model_class, ObjectDetector):
        val_score = metrics[0]["val_mAP"]

    # Log additional metrics to Optuna
    trial.set_user_attr("val_loss", trainer.callback_metrics["val_loss"].item())
    
    return val_score


class HPOResultsSelectionCallback:
    '''
    Optuna study callback to always save the best checkpoint and hyperparameter csv based on the validation score.
    During each trial PyTorch Lightning saves a temporary best checkpoint for this trial.
    Then this callback compares the temporary checkpoint with the best checkpoint from previous trials
    and always keeps the best one while deleting the other.
    '''

    def __init__(self, results_path):
        self.results_path = results_path

    def __call__(self, study, trial):
        file_names = os.listdir(self.results_path)

        # Wont work if multiple processes work on the HPO in parallel
        
        # Get the current temp checkpoint path
        temp_checkpoint_file_name = list(filter(lambda fn: fn.startswith("temp") and fn.endswith(".ckpt"), file_names))[0]
        temp_checkpoint_path = os.path.join(self.results_path, temp_checkpoint_file_name)

        # Get the best checkpoint path
        best_checkpoint_file_names_list = list(filter(lambda fn: fn.startswith("best") and fn.endswith(".ckpt"), file_names))
        
        # If there is no best checkpoint yet, rename the temp checkpoint to the first best checkpoint
        if len(best_checkpoint_file_names_list) == 0:
            # No best checkpoint found, so just rename the temp checkpoint as the first best checkpoint
            best_checkpoint_path = temp_checkpoint_path.replace("temp", "best")
            os.rename(temp_checkpoint_path, best_checkpoint_path)
            logging.info(f"Trial {trial.number}: No best checkpoint found. Current trial checkpoint is the first best checkpoint.")
            # Save the current trials hyperparameters as the best hyperparameters CSV
            best_hyperparameters_path = os.path.join(self.results_path, f"best_hyperparameters_Trial_{trial.number}.csv")
            pd.DataFrame([study.best_params]).to_csv(best_hyperparameters_path, index=False)
            logging.info(f"Trial {trial.number}: Saved best hyperparameters CSV with the hyperparameters from this trial.")
            return
        
        best_checkpoint_file_name = best_checkpoint_file_names_list[0]
        best_checkpoint_path = os.path.join(self.results_path, best_checkpoint_file_name)

        # Compare the temp and best checkpoints based on the validation F1 score
        metric_substring = "val_f1=" if "val_f1=" in temp_checkpoint_file_name else "val_mAP="
        temp_checkpoint_score = float(temp_checkpoint_file_name.split(metric_substring)[1].removesuffix(".ckpt"))
        best_checkpoint_score = float(best_checkpoint_file_name.split(metric_substring)[1].removesuffix(".ckpt"))

        # If the temp checkpoint has a higher score than the best checkpoint, replace the best checkpoint with the temp checkpoint
        if temp_checkpoint_score > best_checkpoint_score:
            # Only rename, dont delete the old best results to keep a history and be able to fall back in case of bugs etc
            old_best_checkpoint_new_path = best_checkpoint_path.replace("best", "prev_best")
            os.rename(best_checkpoint_path, old_best_checkpoint_new_path)
            new_best_checkpoint_path = temp_checkpoint_path.replace("temp", "best")
            os.rename(temp_checkpoint_path, new_best_checkpoint_path)
            logging.info(f"Trial {trial.number}: Replaced best checkpoint with the checkpoint from this trial. Trial score: {temp_checkpoint_score}, Best score: {best_checkpoint_score}")

            # Replace the best hyperparameters CSV with the current trials hyperparameters
            best_hyperparameters_file_name = list(filter(lambda fn: fn.startswith("best_hyperparameters") and fn.endswith(".csv"), file_names))[0]
            best_hyperparameters_path = os.path.join(self.results_path, best_hyperparameters_file_name)
            old_best_hyperparameters_new_path = best_hyperparameters_path.replace("best", "prev_best")
            os.rename(best_hyperparameters_path, old_best_hyperparameters_new_path)
            best_hyperparameters_path = os.path.join(self.results_path, f"best_hyperparameters_Trial_{trial.number}.csv")
            pd.DataFrame([study.best_params]).to_csv(best_hyperparameters_path, index=False)
            logging.info(f"Trial {trial.number}: Saved best hyperparameters CSV with the hyperparameters from this trial.")
        # Otherwise, delete the temp checkpoint
        else:
            os.remove(temp_checkpoint_path)


class HPOLogParameterRangesCallback:
    '''
    Optuna study callback to log the parameter ranges of the hyperparameters to a JSON file.
    
    Note that this is run after the trial has finished, so ranges are only logged after the first trial has completed.
    '''

    def __init__(self, results_path):
        self.results_path = results_path

    def __call__(self, study, trial):
        # On the first trial log the ranges once for the entire study (or if the first trial didnt finish / the file was deleted)
        hparams_ranges_path = os.path.join(self.results_path, "hyperparameter_ranges.txt")
        if trial.number == 0 or not os.path.exists(hparams_ranges_path):
            distributions = [f"{key}: {str(value)}" for key, value in trial.distributions.items()]
            with open(hparams_ranges_path, "w") as f:
                f.write("\n".join(distributions))
            logging.info(f"Trial {trial.number}: Saved hyperparameter ranges to {hparams_ranges_path}")
        else:
            '''
            On the later trials check if the ranges have changed because of optional
            model hyperparameters that are only used in some trials (e.g. later layer sizes).
            '''
            with open(hparams_ranges_path, "r") as f:
                old_ranges = set(f.read().splitlines()) # No newline at the end of the strings
            new_ranges = set([f"{key}: {str(value)}" for key, value in trial.distributions.items()])
            new_ranges = new_ranges.union(old_ranges) # Combine old and new ranges, remove duplicates
            if old_ranges != new_ranges:
                # If the ranges have changed, update the ranges in the file
                with open(hparams_ranges_path, "w") as f:
                    f.write("\n".join(new_ranges))
                logging.info(f"Trial {trial.number}: Updated hyperparameter ranges in {hparams_ranges_path}")


def optimize_hyperparameters(model_class, n_trials, train_set, validation_set, results_path, run_name, dataset_name,
                             training_epochs,
                             use_early_stopping,
                             use_pruning,
                             dataloaders_num_workers, dataloaders_persistent_workers,
                             gpu, use_grad_acc, grad_acc_partial_batch_size, 
                             classification_multiview=False,
                             allow_continue_run=False,
                             optuna_sqlite_path=None,
                             early_stopping_patience=None,
                             batch_size=None, strict_ckpt_loading=True,
                             pruning_n_startup_trials=None, pruning_n_warmup_steps=None,
                             optimize_early_stopping_patience=False):
    """
    Optimize hyperparameters for a given model using Optuna.
    Uses F1 or mAP score on the validation set as the objective.

    Args:
    model_class: The model class to optimize hyperparameters for, e.g. DinoV2Classifier, MaskRCNNObjectDetector
    n_trials: The number of Optuna trials to run
    train_set: The training dataset
    validation_set: The validation dataset. Each trial will be evaluated on this dataset.
    results_path: The path to the results directory
    run_name: The ID/name of the hyperparameter optimization run
    dataset_name: The name of the dataset as a string for logging purposes
    training_epochs: The number of epochs to train the model in each trial for
    use_early_stopping: Whether to use early stopping during training
    use_pruning: Whether to use Optuna pruning during training
    dataloaders_num_workers: The number of workers to use for the PyTorch DataLoaders
    dataloaders_persistent_workers: Whether to use persistent workers for the PyTorch DataLoaders
    gpu: The GPU index to use for training
    use_grad_acc: Whether to use gradient accumulation
    grad_acc_partial_batch_size: The number of images in a partial batch for gradient accumulation, has to be a divisor of 8, 16, 32 and 64
    classification_multiview: Whether to use multi-view training for classification models. Default is False.
    allow_continue_run: Whether to continue an existing Optuna study with the same run_name if it exists. Default is False.
    optuna_sqlite_path: The path to the Optuna SQLite database. Default is None, which uses the results_path.
    early_stopping_patience: Optional, required when use_early_stopping=True. The number of epochs to wait for improvement before stopping training
    batch_size: The optional fixed batch size to use for training. If None, the batch size will be optimized as a hyperparameter. Default is None.
    strict_ckpt_loading: Whether to use strict checkpoint loading. Default is True.
    pruning_n_startup_trials: Optional, required when use_pruning=True. n_startup_trials for the MedianPruner
    pruning_n_warmup_steps: Optional, required when use_pruning=True. n_warmup_steps for the MedianPruner
    optimize_early_stopping_patience: Whether to optimize the early stopping patience as a hyperparameter. Default is False.

    Returns:
    The Optuna study object containing the results of the hyperparameter optimization.
    """

    logging.info("Starting hyperparameter optimization")
    logging.info(f"Optuna version: {optuna.__version__}")

    if use_early_stopping and early_stopping_patience is None:
        raise ValueError("early_stopping_patience is required when use_early_stopping is True")
    
    if use_pruning and (pruning_n_startup_trials is None or pruning_n_warmup_steps is None):
        raise ValueError("pruning_n_startup_trials and pruning_n_warmup_steps are required when use_pruning is True")

    # Create run directory for this HPO run
    run_dir = os.path.join(results_path, run_name)
    if not os.path.exists(run_dir):
        logging.info(f"Run directory for this HPO run does not exist. Creating it at {run_dir}.")
        os.makedirs(run_dir)

    # Create TensorBoard directory for this HPO run
    tensorboard_dir = os.path.join(run_dir, "TensorBoard")
    already_exists = os.path.exists(tensorboard_dir)
    if not already_exists:
        logging.info(f"TensorBoard directory for this HPO run does not exist. Creating it at {tensorboard_dir}")
        os.makedirs(tensorboard_dir)
    elif not allow_continue_run:
        # Don't overwrite existing runs if ALLOW_CONTINUE_RUN is False
        logging.error("TensorBoard directory for this HPO run already exists. Exiting.")
        exit()

    # Possibly resume study, if a study with that HPO_RUN already exists
    if allow_continue_run and already_exists:
        logging.info("TensorBoard run directory for this HPO run already exists and allow_continue_run is set to True. Resuming Optuna study.")
        # Remove temp checkpoints from previous executions
        temp_checkpoint_files = [f for f in os.listdir(run_dir) if f.startswith("temp_Trial")]
        if len(temp_checkpoint_files) > 0:
            logging.warning("Removing temporary checkpoint file(s) from previous executions.")
            for temp_checkpoint_file in temp_checkpoint_files:
                os.remove(os.path.join(run_dir, temp_checkpoint_file))

    # Pruner for Optuna
    if use_pruning:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=pruning_n_startup_trials, n_warmup_steps=pruning_n_warmup_steps)
    else:
        pruner = None

    # Callback to select and save the best checkpoint and hyperparameter csv
    hpo_results_selection_callback = HPOResultsSelectionCallback(run_dir)

    # Callback to log the hyperparameter ranges to a JSON file
    hpo_log_parameter_ranges_callback = HPOLogParameterRangesCallback(run_dir)

    # Format sqlite path and study name for Optuna study
    if optuna_sqlite_path is not None:
        sqlite_path = f"sqlite:///{optuna_sqlite_path}"
    else:
        sqlite_path = os.path.join(results_path, f"HPO_Optuna_SQLite_DB.db")
        sqlite_path = f"sqlite:///{sqlite_path}"
    study_name = f"{run_name} {model_class.name}"

    optuna.logging.enable_propagation()

    # Hyperparameter optimization
    study = optuna.create_study(direction="maximize",
                                study_name=study_name,
                                storage=sqlite_path, 
                                load_if_exists=allow_continue_run,
                                pruner=pruner)
    
    # If study is resumed, only run until the specified number of trials
    # Can be increased anytime in the config :)
    if allow_continue_run and already_exists:
        current_trial_count = len(study.trials)
        if current_trial_count >= n_trials:
            logging.warning(f"HPO is already finished. {current_trial_count} trials have been run, " +
                            f"which is equal to or more than n_trials={n_trials}. " +
                            "If you want to run more trials, increase n_trials in the config.")
            return study
        else:
            logging.info(f"Resuming HPO with {current_trial_count} trials already run. " +
                         f"Running {n_trials - current_trial_count} additional trials.")
            n_trials = n_trials - current_trial_count

    parametrized_objective = lambda trial: _objective(
        trial=trial,
        model_class=model_class,
        classification_multiview=classification_multiview,
        train_set=train_set,
        validation_set=validation_set,
        results_path=results_path,
        run_name=run_name,
        dataset_name=dataset_name,
        training_epochs=training_epochs,
        use_early_stopping=use_early_stopping,
        early_stopping_patience=early_stopping_patience,
        use_pruning=use_pruning,
        dataloaders_num_workers=dataloaders_num_workers,
        dataloaders_persistent_workers=dataloaders_persistent_workers,
        gpu=gpu,
        use_grad_acc=use_grad_acc,
        grad_acc_partial_batch_size=grad_acc_partial_batch_size,
        batch_size=batch_size,
        strict_ckpt_loading=strict_ckpt_loading,
        optimize_early_stopping_patience=optimize_early_stopping_patience
    )
    callbacks = [hpo_results_selection_callback, hpo_log_parameter_ranges_callback]
    study.optimize(parametrized_objective, n_trials=n_trials, callbacks=callbacks)

    # Log best hyperparameters
    logging.info(f"Best hyperparameters found (Trial {study.best_trial.number}):")
    logging.info(study.best_params)
    if issubclass(model_class, Classifier):
        logging.info(f"Best F1 score: {study.best_value}")
    elif issubclass(model_class, ObjectDetector):
        logging.info(f"Best mAP score: {study.best_value}")

    logging.info("Hyperparameter optimization finished")

    return study