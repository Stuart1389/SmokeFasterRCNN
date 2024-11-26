def saveModel(model):
    # Creating directory which im gunna save model state_dicts in
    MODEL_PATH = Path("savedModels")
    MODEL_PATH.mkdir(parents=True,  # make parent dir if doesnt exist
                     exist_ok=True)  # dont error if already exists

    # Setting name
    MODEL_NAME = "valTestB.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model state dict, smaller than saving entire model
    print(f"saving the model to :{MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)


def savePlot(plot):
    # Creating directory to save plot
    PLOT_PATH = Path("savedPlots")
    PLOT_PATH.mkdir(parents=True, exist_ok=True)  # Make parent dir if it doesn't exist

    # Setting name using current date and time
    CURRENTDATEANDTIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format as YYYY-MM-DD_HH-MM-SS
    PLOT_SAVE_PATH = PLOT_PATH / f"loss_plot_{CURRENTDATEANDTIME}.png"

    # Saving plot to path
    print(f"Saving the plot to: {PLOT_SAVE_PATH}")
    plot.savefig(PLOT_SAVE_PATH)
    plt.close(plot)  # Close plot


def train_loop(EPOCHS, model, optimizer, train_dataloader, test_dataloader, device, lr_scheduler, plot_train_loss):
    train_loss_vals = []
    for epoch in range(EPOCHS):
        # train for one epoch, printing every 10 iterations
        m, train_loss = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate using scheduler
        lr_scheduler.step()
        # evaluate on test dataset
        evaluate(model, test_dataloader, device=device)
        #TEST TEMP
        #print_loss(model, train_dataloader, device)
        # get loss vals for graph
        train_loss_vals.append(train_loss)
    if(plot_train_loss):
        plot_loss(train_loss_vals)


def plot_loss(all_train_losses):
    # Flatten list of losses and track which epoch each loss is from
    all_losses = []
    epoch_labels = []

    for epoch, epoch_losses in enumerate(all_train_losses):
        all_losses.extend(epoch_losses)  # Add loss for the current epoch
        epoch_labels.extend([epoch + 1] * len(epoch_losses))  # Label each loss with its epoch number

    # Plot the loss over all iterations
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_losses, label="Loss", color='blue')

    # Annotate the plot with epoch numbers at the first iteration for each epoch
    for epoch, epoch_losses in enumerate(all_train_losses):
        first_loss_index = sum(
            len(all_train_losses[i]) for i in range(epoch))  # Index of the first loss in the current epoch
        # Add dot at the first loss of the epoch
        ax.scatter(first_loss_index, epoch_losses[0], color='red', zorder=5)
        # Add epoch number as annotation
        ax.annotate(f'{epoch + 1}',
                    (first_loss_index, epoch_losses[0]),
                    textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8, color='black')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Over Time')
    ax.legend()

    # Save plot using savePlot function
    savePlot(fig)

    plt.show()

def main():
    import sys
    import torch
    import torchvision
    from pathlib import Path
    import Model
    import numpy as np
    import random
    SEED = 1390
    torch.manual_seed(SEED) # manual seed for consistant results
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get model and dataloaders from Model.py
    train_dataloader, test_dataloader = Model.getDataloader()
    model, in_features, model.roi_heads.box_predictor = Model.getModel(fine_tune=True)
    print(f"Model: {model}")
    print(device)
    print(torch.cuda.is_available())
    plot_train_loss = True
    model.to(device) # Put model on gpu
    EPOCHS = 5

    params = [p for p in model.parameters() if p.requires_grad] # get model parameters
    optimizer = torch.optim.SGD( # Set to static gradient descent
        params,
        lr=0.005, # Learning rate
        momentum=0.9, # speeds up optimization, to decrease time to convergence
        weight_decay=0.0005 # tries to prevent larger weights, leads to less overfitting
    )

    # Scheduler starts at higher learning rate and slows down during training
    # Trying to get to convergence quicker without overshooting
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, # SGD
        step_size=3, # Reduces learning rate every 3 epochs
        gamma=0.1 # learning rate becomes 10% of previous 0.5 -> 0.05
    )

    # Function trains model
    train_loop(EPOCHS, model, optimizer, train_dataloader, test_dataloader, device, lr_scheduler, plot_train_loss)

    # Checks if we want to save the model state_dict
    print("Finished")
    print("Save model parameters?")
    #save_model_parameters = input()
    save_model_parameters = "yes"
    if save_model_parameters == "yes":
        saveModel(model)

    # IoU - Measures overlap between 2 bounding boxes
    # Precission - Measures how many positive predictions were correct | high = few false positives
    # Recall - Measures how many objects the model detected | high = detected most objects
    # Loss - how wrong model is, lower is better
    # Learning rate - how much model changes weights during each iteration
    # Loss classifier - how well the model correctly classifies an object, lower is better
    # Loss box reg - measures how accuratley predicted bbox is when compared to og bbox, lower is better
    # Loss objectness - model confidence at seperating background from object, lower is better
    # Loss rpn box reg - RPN(regional proposal network) regression loss, measures region accuracy, lower is better

if __name__ == '__main__':
    from pathlib import Path
    from datetime import datetime
    import torch
    import matplotlib.pyplot as plt
    import os
    import sys
    # Mark dir for google colab
    current_dir = os.getcwd()
    relative_path = os.path.join(current_dir, '..', 'Libr')
    sys.path.append(relative_path)
    from engine import train_one_epoch, evaluate, print_loss
    main()



