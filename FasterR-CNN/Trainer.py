import copy
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
from TrainingStepsKnowDistil import train_step_know_distil
from Logger import Logger
from GetValues import checkColab, setTrainValues, setGlobalValues
from SmokeModel import SmokeModel
from PlotGraphs import PlotGraphs
from datetime import timedelta
from tabulate import tabulate
import wandb
from torch.profiler import profile, schedule, tensorboard_trace_handler
from AdaptStudentFeatures import AdaptFeatures
from torch.amp import GradScaler, autocast
from SmokeUtils import get_layers_to_fuse


current_dir = os.getcwd()
# add libr as source
relative_path = os.path.join(current_dir, '..', 'Libr')
sys.path.append(relative_path)


class Trainer:
    # Constructor
    def __init__(self, model, train_dataloader, validate_dataloader, device, plot_train_loss=True, checkpoint = None, teacher=None, adaptive_model=None):
        # initialising variables
        self.model = model
        self.teacher_model = teacher
        self.adaptive_model = adaptive_model
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
        self.model_name = setTrainValues("model_name")
        self.log_model_name = setTrainValues("model_id") + "_" + self.model_name
        print(self.log_model_name)
        self.batch_size = setTrainValues("BATCH_SIZE")
        self.cur_train_iteration = 0
        self.cur_val_iteration = 0
        self.know_distil = setTrainValues("know_distil")

        # profiler
        self.start_profiler = setTrainValues("start_profiler")
        self.record_trace = setTrainValues("record_trace")
        self.logWB = setTrainValues("logWB")


        # getting hyperparameters
        self.learning_rate = setTrainValues("learning_rate")
        self.momentum = setTrainValues("momentum")
        self.weight_decay = setTrainValues("weight_decay")
        self.step_size = setTrainValues("step_size")
        self.gamma = setTrainValues("gamma")

        # Creating save path
        self.save_path = Path("savedModels/" + self.model_name)
        self.save_path.mkdir(parents=True, # make parent dir if doesnt exist
                         exist_ok=True) # dont cry if already exists
        self.scalar = None
        # mixed precission
        if(setTrainValues("amp_mixed_precission")):
            self.scalar = GradScaler('cuda')

        # initialise optimizer and scheduler
        self.params = [p for p in model.parameters() if p.requires_grad] # get model parameters
        self.optimizer = torch.optim.SGD( # Set to static gradient descent
            self.params,
            lr=self.learning_rate, # Learning rate
            momentum=self.momentum, # speeds up optimization, decrease time to convergence
            weight_decay=self.weight_decay # tries to prevent larger weights, to prevent overfitting
        )

        # Scheduler starts at higher learning rate and slows down during training
        # Trying to get to convergence quicker without overshooting
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, # SGD
            step_size=self.step_size, # Reduces learning rate every 3 epochs
            gamma=self.gamma # learning rate becomes 10% of previous 0.5 -> 0.05
        )

        # Init wandb
        if(not self.logWB):
            # disable wandb
            os.environ["WANDB_MODE"] = "disabled"
            print("wandbdisabled")


        wandb.init(project="smoke-detection", name=self.log_model_name, config={
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience
        })



    # function saves model parameters
    def save_model(self, final_save = None):
        state_dict = self.model.state_dict()
        if(setTrainValues("quant_aware_training")):
            # need to convert before saving but
            # cant convert mid train
            # so deep copy model and get its state_dict
            temp_model = copy.deepcopy(self.model)
            temp_model.to('cpu')
            temp_model.eval()
            temp_model.backbone.body = torch.ao.quantization.convert(temp_model.backbone.body, inplace=True)
            state_dict = temp_model.state_dict()
            #self.model.to('cpu')
            #self.model.backbone.body = torch.ao.quantization.convert(self.model.backbone.body, inplace=True)
        # Save model parameters
        # Setting name
        model_save_name = self.model_name + ".pth"
        model_save_path = self.save_path / model_save_name
        print(f"saving the model to: {model_save_path}")
        # save model parameters to file
        torch.save(obj=state_dict, f=model_save_path)


    # function defines training loop used to train model
    def train_loop(self):
        # start timer
        start_time = time.time()
        # Creating log file to save console in a .txt file
        log_filename = self.save_path / (setTrainValues("model_name") + ".txt")
        print(time.strftime("%Y-%m-%d_%H-%M-%S"))
        sys.stdout = Logger(log_filename)

        # lists to hold loss values from train and validate steps
        train_loss_vals = []
        validate_loss_vals = []
        train_loss_it_vals = []
        validate_loss_it_vals = []

        # Profiler
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.save_path / 'profiler_train_output'),
            record_shapes=False,
            profile_memory=False,
            with_stack=self.record_trace,
            with_flops=False,
            with_modules=False
        )

        if(setTrainValues("quant_aware_training")):
            # QUANT AWARE TRAINING
            # quant fusions must be done in eval mode
            self.model.eval()
            #Quant aware training
            self.model.backbone.body.qconfig = torch.ao.quantization.get_default_qconfig('x86')
            #print(self.model.backbone.body)

            module_names = list(self.model.backbone.body.named_modules())
            # get layers to fure, see smoke utils
            layers_to_fuse = get_layers_to_fuse(module_names)

            self.model.backbone.body = torch.ao.quantization.fuse_modules(self.model.backbone.body,layers_to_fuse)
            # prepare requires train mode
            self.model.train()
            self.model.backbone.body = torch.ao.quantization.prepare_qat(self.model.backbone.body, inplace=False)

        # !!TRAINING LOOP START!!
        for epoch in range(self.epochs):
            if(self.start_profiler):
                profiler.start()
            # TRAINING STEP
            with torch.profiler.record_function("TRAINING"):
                if(self.know_distil):
                    _, train_loss_it_dict, train_loss_dict, self.cur_train_iteration = train_step_know_distil(
                        self.model, self.optimizer, self.train_dataloader, self.device, epoch, self.epochs,
                        self.cur_train_iteration, print_freq=10, teacher=self.teacher_model, adaptive_model=self.adaptive_model
                    )
                else:
                    _, train_loss_it_dict, train_loss_dict, self.cur_train_iteration = train_step(
                        self.model, self.optimizer, self.train_dataloader, self.device, epoch, self.cur_train_iteration, print_freq=10, scaler=self.scalar
                    )
                self.lr_scheduler.step()
            with torch.profiler.record_function("VALIDATING"):
            # VALIDATION STEP
                _, validate_loss_it_dict, validate_loss_dict, self.cur_val_iteration = validate_step(
                    self.model, self.validate_dataloader, self.device, epoch, self.cur_val_iteration
                )

            # holds number of epochs model trained on
            self.epochs_trained += 1
            #if(epoch > 8 and setTrainValues("quant_aware_training")):
                #self.model.backbone.body.apply(torch.ao.quantization.disable_observer)
            #if (epoch > 8 and setTrainValues("quant_aware_training")):
                #self.model.backbone.body.apply(torch.nn.intrinsic.qat.freeze_bn_stats)



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

            #wandb logging
            wandb.log({
                "avg_train_loss": current_train_loss,
                "avg_val_loss": current_val_loss,
            })

            # Early stopping implementation, at each epoch check for improvement in validation loss
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_train_loss = current_train_loss
                self.epochs_no_improve = 0
                print(f"Validation loss improved to {self.best_val_loss:.4f}, saving model...")
                self.save_model() # checkpointing epochs with lower loss values
            else:
                self.epochs_no_improve += 1
                print(f"No improvement in validation loss for {self.epochs_no_improve} epochs. Current best loss is {self.best_val_loss:.4f}")
            # stop if loss doesnt improve
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping due to no improvement. Best val loss: {self.best_val_loss:.4f} Best train loss: {self.best_train_loss:.4f}")
                break
            if(self.start_profiler):
                profiler.stop()


        # GETTING METRICS
        # Creating instance of class to plot graphs from loss values
        plot_graphs = PlotGraphs(train_loss_vals, validate_loss_vals,
                                 train_loss_it_vals, validate_loss_it_vals)
        # Get highest vram reserved value since program start
        cur_highest_vram = torch.cuda.max_memory_reserved() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats() # reset
        # calculate total time taken to train
        end_time = time.time()
        total_training_time = end_time - start_time
        avg_time_per_epoch = total_training_time / self.epochs_trained

        # display profiler metrics
        if profiler is not None and self.start_profiler:
            key_averages = profiler.key_averages()
            print(key_averages.table())

        # Display results
        tokens_spent = self.display_results(cur_highest_vram, total_training_time,
                              avg_time_per_epoch, self.epochs_trained,
                              self.best_val_loss, self.best_train_loss)
        # stop writing to log and go back to only outputting to console
        sys.stdout.file.close()
        sys.stdout = sys.__stdout__
        # Logging values
        wandb.log({
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "total_train_time(seconds)": total_training_time,
            "avg_time_per_epoch(seconds)": avg_time_per_epoch,
            "tokens_spent": tokens_spent,
            "epochs_trained": self.epochs_trained,
            "reserved_vram": cur_highest_vram
        })

        wandb.finish()
        if(setTrainValues("save_at_end")):
            self.save_model()
        # wandb logging
        self.save_model(True)

    # function displays metrics collected from training loop
    def display_results(self, cur_highest_vram, total_training_time,
                        avg_time_per_epoch, epochs_trained,
                        best_val_loss, best_train_loss):
        """
        print("Highest VRAM used: {:.2f} MB".format(cur_highest_vram))
        print(f"Total training time: {str(timedelta(seconds=total_training_time)).split('.')[0]}")
        print(f"Average time per epoch: {str(timedelta(seconds=avg_time_per_epoch)).split('.')[0]}")
        print(f"Model trained on {self.epochs_trained} epochs.")
        print(f"Best val loss: {self.best_val_loss:.4f} Best train loss: {self.best_train_loss:.4f}")
        print("Finished")
        """

        # Getting values used during training
        gpu, tokens = self.get_GPU_name(self.model_name)
        total_training_time_fmt = total_training_time / 3600 # get time in hours
        tokens_spent = total_training_time_fmt * tokens

        # set patience to same as epoch when we want to compare times
        if(self.patience == self.epochs):
            patience_val = "N/A"
        else:
            patience_val = self.patience

        # Data for table
        data = [
            ["Model Name", f"{self.model_name}"],
            ["GPU\nTokens per hours\nTokens used",
             f"{gpu}\n{tokens}\n{tokens_spent:.2f}"], # lazy
            ["Batch Size", f"{self.batch_size}"],
            ["Epochs Set", f"{self.epochs}"],
            ["Patience", f"{patience_val}"],
            ["Epochs Trained", f"{epochs_trained}"],
            ["----- Training metrics -----"],
            ["Best Validation Loss", f"{best_val_loss:.4f}"],
            ["Best Training Loss", f"{best_train_loss:.4f}"],
            ["Total Training Time", str(timedelta(seconds=total_training_time)).split(".")[0]],
            ["Average Time per Epoch", str(timedelta(seconds=avg_time_per_epoch)).split(".")[0]],
            ["Highest VRAM Used", f"{cur_highest_vram:.2f} MB"],
        ]

        # Displaying table
        print(tabulate(data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

        # Ugly output for copy and paste into excel
        print("\n Ugly output:\n")
        # Get each metric from the table above and print, check if row is empty because of row break
        for row in data:
            if len(row) > 1:  # Check if the row has value
                value = row[1]
                if value == f"{gpu}\n{tokens}\n{tokens_spent:.2f}":
                    print(f'"{value}"', end="\t")  # Add quotes to paste into same cell
                else:
                    print(value, end="\t")

        print(f'"lr={self.learning_rate}, momentum={self.momentum}, '
              f'weight_decay={self.weight_decay}, step_size={self.step_size}, '
              f'gamma={self.gamma}"')

        return tokens_spent
    # Function gets the gpu name from model name so it can be copied into a testing sheet
    def get_GPU_name(self, model_name):
        # list of gpus
        GPU_dict = {"2080" : ("RTX 2080", 0),
                           "100" : ("A100", 8.47),
                           "l4" : ("L4", 2.4),
                           "t4" : ("T4", 1.44)}

        model_name_lower = model_name.lower()
        for key in GPU_dict:
            if key in model_name_lower:
                return GPU_dict[key]
        return "N/A", 0  # no gpu in name

def main():
    # reproducibility
    SEED = 1390
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    non_blocking = setTrainValues("non_blocking")

    # Setting variables for instance
    #device = setGlobalValues("device")
    device = "cuda" if torch.cuda.is_available() else "cpu" # device agnostic
    print(device)

    # Create instance of SmokeModel class
    smoke_model = SmokeModel()
    # get dataloaders from model class
    train_dataloader, validate_dataloader, _ = smoke_model.get_dataloader() # don't need test dataloader
    # get model from model class
    get_student = setTrainValues("know_distil")
    if (setTrainValues("know_distil")):
        # getting teacher model for knowlege distil
        teacher_model = smoke_model.get_model(get_teacher=True) # Getting teacher
        teacher_model.to(device, non_blocking=non_blocking)
        adaptive_model = AdaptFeatures()
        adaptive_model.to(device, non_blocking=non_blocking)
        print(adaptive_model)
    # getting model to train
    model, in_features, model.roi_heads.box_predictor = smoke_model.get_model(know_distil=get_student)

    """
    train_dataloader, validate_dataloader, _ = Model.getDataloader() # don't need test dataloader
    # get model from model class
    model, in_features, model.roi_heads.box_predictor = Model.getModel()
    """

    model.to(device, non_blocking=non_blocking) # put model on gpu (or cpu :( )

    # create instance of class
    if(get_student):
        trainer = Trainer(model, train_dataloader, validate_dataloader, device, teacher=teacher_model, adaptive_model=adaptive_model)
    else:
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


