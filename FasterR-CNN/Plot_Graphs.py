# This plots the mega graph from loss values got during training
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from itertools import chain
from Get_Values import checkColab, setTrainValues
def plot_all_loss(train_dict, validate_dict, train_loss_it_vals, validate_loss_it_vals):
    # get loss types from averafe and iterations loss dictionaries
    loss_types = train_dict[0].keys()
    loss_it_types = train_loss_it_vals[0][0].keys()

    # get number of columns/which is equal to number of loss types
    num_cols = len(loss_types)

    # Create subplots with 3 rows (epoch/validation average losses, train loss iterations, validate loss iterations)
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 15), sharey=False)
    fig.suptitle("Train vs Validate Loss", fontsize=16)

    # epoch/validation average losses (1st row)
    for i, loss_type in enumerate(loss_types):
        train_losses = [epoch[loss_type] for epoch in train_dict]
        validate_losses = [epoch[loss_type] for epoch in validate_dict]

        # Plot on the respective axis
        axes[0, i].plot(range(1, len(train_dict) + 1), train_losses, label='Train Loss', marker='o')
        axes[0, i].plot(range(1, len(validate_dict) + 1), validate_losses, label='Validate Loss', marker='o', color="orange")

        axes[0, i].set_title(loss_type.replace('_', ' ').capitalize())
        axes[0, i].set_xlabel('Epochs')
        axes[0, i].legend()
        if i == 0:
            axes[0, i].set_ylabel('Loss Value')

    # Plot train loss iterations (2nd row)
    for i, loss_it_types in enumerate(loss_it_types):
        #print(train_loss_it_vals)
        train_it_losses = [d[loss_it_types] for sublist in train_loss_it_vals for d in sublist]
        axes[1, i].plot(range(1, len(train_it_losses) + 1), train_it_losses, label='Train Loss')
        axes[1, i].set_title(loss_it_types.replace('_', ' ').capitalize())
        axes[1, i].set_xlabel('Iterations')
        axes[1, i].legend()
        if i == 0:
            axes[1, i].set_ylabel('Loss Value')

    # validate loss iterations (3rd row)
    loss_it_types = validate_loss_it_vals[0][0].keys()
    for i, loss_it_types in enumerate(loss_it_types):
        #print(validate_loss_it_vals)
        validate_it_losses = [d[loss_it_types] for sublist in validate_loss_it_vals for d in sublist]
        axes[2, i].plot(range(1, len(validate_it_losses) + 1), validate_it_losses, label='Validate Loss', color="orange")
        axes[2, i].set_title(loss_it_types.replace('_', ' ').capitalize())
        axes[2, i].set_xlabel('Iterations')
        axes[2, i].legend()
        if i == 0:
            axes[2, i].set_ylabel('Loss Value')

    # left, bottom, right, top
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Save the plot and data
    savePlot(fig)
    saveJSON(train_dict, validate_dict, train_loss_it_vals, validate_loss_it_vals)

# funtion saves plot so it can be used later
def savePlot(plot):
    # Creating directory to save plot
    model_name = setTrainValues("model_name")
    PLOT_PATH = Path("savedModels/" + model_name)
    PLOT_PATH.mkdir(parents=True, exist_ok=True)  # Make parent dir if it doesn't exist
    # Setting name using current date and time
    CURRENTDATEANDTIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format as YYYY-MM-DD_HH-MM-SS
    filename = setTrainValues("plotJSON_fname")
    PLOT_SAVE_PATH = PLOT_PATH / (filename + f"_{CURRENTDATEANDTIME}.png") # just name it same as json who cares

    # Saving plot to path
    print(f"Saving the plot to: {PLOT_SAVE_PATH}")
    plot.savefig(PLOT_SAVE_PATH)
    plt.close(plot)

def saveJSON(train_dict, validate_dict, train_loss_it_vals, validate_loss_it_vals):
    # combine train and validate into dict
    data = {
        "train": train_dict,
        "validate": validate_dict,
        "train_it": train_loss_it_vals,
        "validate_it": validate_loss_it_vals,
    }

    # Creating dict to save json
    model_name = setTrainValues("model_name")
    JSON_PATH = Path("savedModels/" + model_name)
    JSON_PATH.mkdir(parents=True, exist_ok=True)  # Make parent dir if it doesn't exist

    filename = setTrainValues("plotJSON_fname")
    file_dir = JSON_PATH / (filename + ".json")
    # save to json
    with open(file_dir, 'w') as f:
        json.dump(data, f, indent=4)


