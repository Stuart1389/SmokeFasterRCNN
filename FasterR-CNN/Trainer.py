import sys
import torch
import os
import time
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from TrainingSteps import train_step, validate_step
from Plot_Graphs import plot_all_loss
from Logger import Logger
from Get_Values import checkColab, setTrainValues, setGlobalValues
from SmokeModel import SmokeModel
from datetime import timedelta

current_dir = os.getcwd()
# add libr as source
relative_path = os.path.join(current_dir, '..', 'Libr')
sys.path.append(relative_path)


class Trainer:
    # Constructor
    def __init__(self, model, train_dataloader, validate_dataloader, device, plot_train_loss=True, checkpoint = None):
        # initialising variables
        self.model = model
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.device = device
        self.epochs = setTrainValues("EPOCHS") # number of epochs to train on
        self.patience = setTrainValues("PATIENCE") # num of epochs to do with no imrpovement
        self.plot_train_loss = plot_train_loss
        self.checkpoint = checkpoint # used to train a model from a checkpoint
        self.best_val_loss = float('inf')  # first value make mega
        # this isnt storing the best value, but the train loss value when validation loss was best
        self.best_train_loss = float('inf')  # first value make mega
        self.epochs_no_improve = 0 # number of epochs with no improvement
        self.epochs_trained = 0 # number of epochs trained (tells us epochs trained for early stopping)

        # initialise optimizer and scheduler
        self.params = [p for p in model.parameters() if p.requires_grad] # get model parameters
        self.optimizer = torch.optim.SGD( # Set to static gradient descent
            self.params,
            lr=0.001, # Learning rate
            momentum=0.9, # speeds up optimization, decrease time to convergence
            weight_decay=0.0005 # tries to prevent larger weights, to prevent overfitting
        )

        # Scheduler starts at higher learning rate and slows down during training
        # Trying to get to convergence quicker without overshooting
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, # SGD
            step_size=3, # Reduces learning rate every 3 epochs
            gamma=0.1 # learning rate becomes 10% of previous 0.5 -> 0.05
        )


    # function saves model parameters
    def save_model(self):
        # Save model parameters
        # this is reduntant cause we do it to put log file in same folder but leaving in case i want to do funny business
        # Creating directory which im gunna save model state_dicts in
        model_path = Path("savedModels")
        model_path.mkdir(parents=True, # make parent dir if doesnt exist
                         exist_ok=True) # dont cry if already exists
        # Setting name
        model_name = setTrainValues("model_name") + ".pth"
        model_save_path = model_path / model_name
        print(f"saving the model to: {model_save_path}")
        # save model parameters to file
        torch.save(obj=self.model.state_dict(), f=model_save_path)


    # function defines training loop used to train model
    def train_loop(self):
        # start timer
        start_time = time.time()
        model_path = Path("savedModels")
        model_path.mkdir(parents=True, exist_ok=True)
        # Creating log file to save console in a .txt file
        log_filename = model_path / (setTrainValues("model_name") + ".txt")
        print(time.strftime("%Y-%m-%d_%H-%M-%S"))
        sys.stdout = Logger(log_filename)

        # lists to hold loss values from train and validate steps
        train_loss_vals = []
        validate_loss_vals = []
        train_loss_it_vals = []
        validate_loss_it_vals = []

        # !!TRAINING LOOP START!!
        for epoch in range(self.epochs):
            # TRAINING STEP
            _, train_loss_it_dict, train_loss_dict = train_step(
                self.model, self.optimizer, self.train_dataloader, self.device, epoch, print_freq=10
            )
            self.lr_scheduler.step()

            # VALIDATION STEP
            _, validate_loss_it_dict, validate_loss_dict = validate_step(
                self.model, self.validate_dataloader, device=self.device
            )

            # holds number of epochs model trained on
            self.epochs_trained += 1

            # Add loss dicts to list
            # Average loss per epoch
            train_loss_vals.append(train_loss_dict)
            validate_loss_vals.append(validate_loss_dict)
            # loss per iteration
            train_loss_it_vals.append(train_loss_it_dict)
            validate_loss_it_vals.append(validate_loss_it_dict)

            # get average training loss and display
            current_train_loss = train_loss_dict["avg_total_loss"]
            print(f"Current average training loss: {current_train_loss:.4f}")

            # Early stopping get average validation loss
            current_val_loss = validate_loss_dict["avg_total_loss"]
            print(f"Current average validation loss: {current_val_loss:.4f}")

            # Early stopping implementation, at each epoch check for improvement in validation loss
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_train_loss = current_train_loss
                self.epochs_no_improve = 0
                print(f"Validation loss improved to {self.best_val_loss:.4f}, saving model...")
                self.save_model() # checkpointing epochs with positive loss values (and by that i mean lower)
            else:
                self.epochs_no_improve += 1
                print(f"No improvement in validation loss for {self.epochs_no_improve} epochs. Current best loss is {self.best_val_loss:.4f}")
            # stop if loss doesnt improve
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping due to no improvement. Best val loss: {self.best_val_loss:.4f} Best train loss: {self.best_train_loss:.4f}")
                break

        # Plotting the mega graphs using lists of loss dicts from training
        plot_all_loss(train_loss_vals, validate_loss_vals, train_loss_it_vals, validate_loss_it_vals)
        # Display highest vram usage in gpu during training
        print("Highest VRAM used: {:.2f} MB".format(torch.cuda.max_memory_allocated() / 1024 / 1024))
        torch.cuda.reset_peak_memory_stats() # reset
        # stop writing to log and go back to only outputting to console
        sys.stdout.file.close()
        sys.stdout = sys.__stdout__
        # calculate total time taken to train
        end_time = time.time()
        total_training_time = end_time - start_time
        avg_time_per_epoch = total_training_time / self.epochs_trained
        print(f"Total training time: {str(timedelta(seconds=total_training_time)).split('.')[0]}")
        print(f"Average time per epoch: {str(timedelta(seconds=avg_time_per_epoch)).split('.')[0]}")
        print(f"Model trained on {self.epochs_trained} epochs.")
        print("Finished")
        print(f"Best val loss: {self.best_val_loss:.4f} Best train loss: {self.best_train_loss:.4f}")

def main():
    # reproducibility
    SEED = 1390
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Setting variables for instance
    #device = setGlobalValues("device")
    device = "cuda" if torch.cuda.is_available() else "cpu" # device agnostic
    # Create instance of SmokeModel class
    smoke_model = SmokeModel()
    # get dataloaders from model class
    train_dataloader, validate_dataloader, _ = smoke_model.get_dataloader() # don't need test dataloader
    # get model from model class
    model, in_features, model.roi_heads.box_predictor = smoke_model.get_model()

    """
    train_dataloader, validate_dataloader, _ = Model.getDataloader() # don't need test dataloader
    # get model from model class
    model, in_features, model.roi_heads.box_predictor = Model.getModel()
    """

    model.to(device, non_blocking=True) # put model on gpu (or cpu :( )

    # create instance of class
    trainer = Trainer(model, train_dataloader, validate_dataloader, device)
    # Start training loop
    trainer.train_loop()

if __name__ == '__main__':
    current_dir = os.getcwd()
    # add libr as source
    relative_path = os.path.join(current_dir, '..', 'Libr')
    sys.path.append(relative_path)
    main()


    # IoU - Measures overlap between 2 bounding boxes
    # Precission - Measures how many positive predictions were correct | high = few false positives
    # Recall - Measures how many objects the model detected | high = detected most objects
    # Loss - how wrong model is, lower is better
    # Learning rate - how much model changes weights during each iteration
    # Loss classifier - how well the model correctly classifies an object, lower is better
    # Loss box reg - measures how accuratley predicted bbox is when compared to og bbox, lower is better
    # Loss objectness - model confidence at seperating background from object, lower is better
    # Loss rpn box reg - RPN(regional proposal network) regression loss, measures region accuracy, lower is better