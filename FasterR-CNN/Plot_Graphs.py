def plot_all_loss(train_dict, validate_dict, train_loss_it_vals, validate_loss_it_vals):
    import matplotlib.pyplot as plt
    from itertools import chain

    # Extracting the loss types dynamically from the first dictionary
    loss_types = train_dict[0].keys()
    loss_it_types = train_loss_it_vals[0][0].keys()

    # Calculate the number of columns needed
    num_cols = len(loss_types)

    # Create subplots with 3 rows (1st row for epoch-level losses, 2nd row for train iterations, 3rd for validation iterations)
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 15), sharey=False)
    fig.suptitle("Train vs Validate Loss", fontsize=16)

    # Plot epoch-level losses (1st row)
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

    # Plot iter-level losses (2nd row)
    for i, loss_it_types in enumerate(loss_it_types):
        #print(train_loss_it_vals)
        train_it_losses = [d[loss_it_types] for sublist in train_loss_it_vals for d in sublist]

        # Plot on the respective axis
        axes[1, i].plot(range(1, len(train_it_losses) + 1), train_it_losses, label='Train Loss')
        axes[1, i].set_title(loss_it_types.replace('_', ' ').capitalize())
        axes[1, i].set_xlabel('Iterations')
        axes[1, i].legend()
        if i == 0:
            axes[1, i].set_ylabel('Loss Value')

    # Plot iter-level losses (2nd row)
    loss_it_types = validate_loss_it_vals[0][0].keys()
    for i, loss_it_types in enumerate(loss_it_types):
        #print(validate_loss_it_vals)
        validate_it_losses = [d[loss_it_types] for sublist in validate_loss_it_vals for d in sublist]

        # Plot on the respective axis
        axes[2, i].plot(range(1, len(validate_it_losses) + 1), validate_it_losses, label='Validate Loss', color="orange")
        axes[2, i].set_title(loss_it_types.replace('_', ' ').capitalize())
        axes[2, i].set_xlabel('Iterations')
        axes[2, i].legend()
        if i == 0:
            axes[2, i].set_ylabel('Loss Value')





    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to fit the suptitle
    plt.show()

    # Save the plot and data
    savePlot(fig)
    saveJSON(train_dict, validate_dict, train_loss_it_vals, validate_loss_it_vals)

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

def saveJSON(train_dict, validate_dict, train_loss_it_vals, validate_loss_it_vals):
    import json
    from pathlib import Path
    from Get_Values import checkColab, setTrainValues
    # combine train and validate
    data = {
        "train": train_dict,
        "validate": validate_dict,
        "train_it": train_loss_it_vals,
        "validate_it": validate_loss_it_vals,
    }

    # Creating directory to save json
    JSON_PATH = Path("savedPlots")
    JSON_PATH.mkdir(parents=True, exist_ok=True)  # Make parent dir if it doesn't exist

    filename = setTrainValues("plotJSON_fname")
    file_dir = JSON_PATH / (filename + ".json")
    # save to json
    with open(file_dir, 'w') as f:
        json.dump(data, f, indent=4)


