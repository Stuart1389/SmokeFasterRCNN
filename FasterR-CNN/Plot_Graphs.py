def plot_all_loss(train_dict, validate_dict):
    import matplotlib.pyplot as plt

    # Extracting the loss types dynamically from the first dictionary
    loss_types = train_dict[0].keys()

    # Creating subplots for each loss type
    fig, axes = plt.subplots(1, len(loss_types), figsize=(20, 5), sharey=False)
    fig.suptitle("Train vs Validate Loss")

    # Plot each loss type
    for i, loss_type in enumerate(loss_types):
        train_losses = [epoch[loss_type] for epoch in train_dict]
        validate_losses = [epoch[loss_type] for epoch in validate_dict]

        # Plot on respective axis
        axes[i].plot(range(1, len(train_dict) + 1), train_losses, label='Train', marker='o')
        axes[i].plot(range(1, len(validate_dict) + 1), validate_losses, label='Validate', marker='o', color="orange")

        axes[i].set_title(loss_type.replace('_', ' ').capitalize())
        axes[i].set_xlabel('Epochs')
        axes[i].legend()
        if i == 0:
            axes[i].set_ylabel('Loss Value')

    plt.tight_layout()
    plt.show()
    savePlot(fig)
    saveJSON(train_dict, validate_dict)

def savePlot(plot):
    from pathlib import Path
    from datetime import datetime
    import matplotlib.pyplot as plt
    # Creating directory to save plot
    PLOT_PATH = Path("savedPlots")
    PLOT_PATH.mkdir(parents=True, exist_ok=True)  # Make parent dir if it doesn't exist
    # Setting name using current date and time
    CURRENTDATEANDTIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format as YYYY-MM-DD_HH-MM-SS
    PLOT_SAVE_PATH = PLOT_PATH / f"lossB_plot_{CURRENTDATEANDTIME}.png"

    # Saving plot to path
    print(f"Saving the plot to: {PLOT_SAVE_PATH}")
    plot.savefig(PLOT_SAVE_PATH)
    plt.close(plot)  # Close plot

def saveJSON(train_dict, validate_dict):
    import json
    from pathlib import Path
    from Get_Values import checkColab, setTrainValues
    # combine train and validate
    data = {
        "train": train_dict,
        "validate": validate_dict
    }

    # Creating directory to save json
    JSON_PATH = Path("savedPlots")
    JSON_PATH.mkdir(parents=True, exist_ok=True)  # Make parent dir if it doesn't exist

    filename = setTrainValues("plotJSON_fname")
    file_dir = JSON_PATH / (filename + ".json")
    # save to json
    with open(file_dir, 'w') as f:
        json.dump(data, f, indent=4)


