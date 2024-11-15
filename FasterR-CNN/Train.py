def saveModel(model):
    from pathlib import Path
    import torch

    # Creating directory which im gunna save model state_dicts in
    MODEL_PATH = Path("savedModels")
    MODEL_PATH.mkdir(parents=True,  # make parent dir if doesnt exist
                     exist_ok=True)  # dont error if already exists

    # Setting name
    MODEL_NAME = "smokeDetBaseLarge15.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model state dict, smaller than saving entire model
    print(f"saving the model to :{MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)


def trainNoCheckpoint(EPOCHS, model, optimizer, train_dataloader, test_dataloader, device, lr_scheduler):
    epoch_loss_vals = []
    for epoch in range(EPOCHS):
        # train for one epoch, printing every 10 iterations
        m, epoch_loss = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate using scheduler
        lr_scheduler.step()
        # evaluate on test dataset
        evaluate(model, test_dataloader, device=device)
        epoch_loss_vals.append(epoch_loss)
    plot_loss(epoch_loss_vals)


def plot_loss(all_epoch_losses):
    # Flatten the list of losses and track which epoch the loss is from
    all_losses = []
    epoch_labels = []

    for epoch, epoch_losses in enumerate(all_epoch_losses):
        all_losses.extend(epoch_losses)  # Add all losses for the current epoch
        epoch_labels.extend([epoch + 1] * len(epoch_losses))  # Label each loss with the corresponding epoch

    # Plot the loss over all iterations
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label="Loss", color='blue')

    # Annotate the plot with epoch numbers at the first iteration of each epoch
    for epoch, epoch_losses in enumerate(all_epoch_losses):
        first_loss_index = sum(len(all_epoch_losses[i]) for i in range(epoch))  # Index of the first loss in the current epoch
        # Add dot
        plt.scatter(first_loss_index, all_epoch_losses[epoch][0], color='red', zorder=5)
        # Add text
        plt.annotate(f'{epoch + 1}',
                     (first_loss_index, all_epoch_losses[epoch][0]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center', fontsize=8, color='black')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Over Time (Training Progress)')
    plt.legend()
    plt.show()

"""
def trainCheckpoint(EPOCHS, model, optimizer, train_dataloader, test_dataloader, device, lr_scheduler):
    for epoch in range(EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate using scheduler
        lr_scheduler.step()
        # evaluate on test dataset
        val_loss = evaluate(model, val_dataloader, device=device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # Optionally, add early stopping based on validation loss
        if epoch - best_epoch > 5:
            print("Stopping early due to no improvement in validation loss.")
            break
"""

def main():
    import sys
    import torch
    import torchvision
    from pathlib import Path
    import Model
    import numpy as np
    import random
    # Mark dir for google colab
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

    model.to(device) # Put model on gpu
    EPOCHS = 2
    checkpoint = False

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

    trainNoCheckpoint(EPOCHS, model, optimizer, train_dataloader, test_dataloader, device, lr_scheduler)

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
    import matplotlib.pyplot as plt
    import os
    import sys
    current_dir = os.getcwd()
    relative_path = os.path.join(current_dir, '..', 'Libr')
    sys.path.append(relative_path)
    from engine import train_one_epoch, evaluate
    main()



