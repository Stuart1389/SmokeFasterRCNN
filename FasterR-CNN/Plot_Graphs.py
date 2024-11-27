def plot_all_loss(train_dict, validate_dict):
    import matplotlib.pyplot as plt

    # Extracting the loss types dynamically from the first dictionary
    loss_types = train_dict[0].keys()

    # Creating subplots for each loss type
    fig, axes = plt.subplots(1, len(loss_types), figsize=(20, 5), sharey=True)
    fig.suptitle("Train vs Validate Loss")

    # Plot each loss type
    for i, loss_type in enumerate(loss_types):
        train_losses = [epoch[loss_type] for epoch in train_dict]
        validate_losses = [epoch[loss_type] for epoch in validate_dict]

        # Plot on respective axis
        axes[i].plot(range(1, len(train_dict) + 1), train_losses, label='Train', marker='o')
        axes[i].plot(range(1, len(validate_dict) + 1), validate_losses, label='Validate', marker='x')

        axes[i].set_title(loss_type.replace('_', ' ').capitalize())
        axes[i].set_xlabel('Epochs')
        axes[i].legend()
        if i == 0:
            axes[i].set_ylabel('Loss Value')

    plt.tight_layout()
    plt.show()