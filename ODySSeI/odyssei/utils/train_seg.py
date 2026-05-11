import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
import os
from copy import deepcopy
from tqdm import tqdm
from utils.eval_seg import test_fn, metrics, metrics_train


def kaiming_init(m):
    """
    Kaiming He initialization for convolutional and linear layers with the ReLU non-linearity

    Arguments:
        m (torch.nn.Module): Module whose weights and biases need to be initialized
    """

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def per_epoch_train_func(
    model, scheduler, train_dataloader, criterion, optimizer, current_epoch_num, device
):
    """
    Train segmentation model for a single epoch

    Arguments:
        model (torch.nn.Module): Model to be trained
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        train_dataloader (CustomDataloader): Training Set data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        current_epoch_num (int): The current training epoch
        device (str): Device to be used

    Returns:
        average loss (float): Loss for the current training epoch averaged over all batches
        average accuracy (float): Accuracy for the current training epoch averaged over all batches
        average lr (float): Learning rate for the current training epoch averaged over all batches
        average f1 score (float): F1-Score for the current training epoch averaged over all batches
    """

    model.train()

    # Initialize history variables
    loss_net = []
    accuracy_net = []
    lr_net = []
    f1_score_net = []

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Loss and gradient step
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update learning rate
        if scheduler:
            scheduler.step()

        # Calculate metrics
        preds = (outputs.sigmoid() > 0.5).to(torch.int8)
        acc, f1_score = metrics_train(preds, targets)
        loss_val = loss.item()

        # Save metrics
        loss_net.append(loss_val)
        accuracy_net.append(acc)
        f1_score_net.append(f1_score)

        if scheduler:
            last_lr = scheduler.get_last_lr()[0]
        else:
            # to make sure this doesnt interfere with training,
            # in case we use an optimizer where this doesnt exist
            try:
                last_lr = optimizer.param_groups[-1]["lr"]
            except:
                last_lr = 0

        lr_net.append(last_lr)

        # Print metrics
        if batch_idx % 10 == 0:
            print(
                f"Train Epoch #: {current_epoch_num}-{batch_idx:03d} "
                f"Batch Loss: {loss_val:0.2e} "
                f"Batch Accuracy: {acc:0.3f} "
                f"Batch F1 Score: {f1_score:0.3f} "
                f"Learning Rate: {last_lr:0.3e} "
            )

    return (
        np.mean(loss_net),
        np.mean(accuracy_net),
        np.mean(lr_net),
        np.mean(f1_score_net),
    )


def train(
    model,
    main_save_dir,
    num_epochs,
    criterion,
    optimizer,
    train_dataloader,
    val_dataloader,
    scheduler=None,
    device="cpu",
    per_epoch_train_func=per_epoch_train_func,
    val_func=test_fn,
    wandb_project="default",
    wandb_run="test",
):
    """
    Main training function

    Arguments:
        model (torch.nn.Module): Model to be trained
        main_save_dir (str): Main directory for saving the best trained model weights
        num_epochs (int): Number of training epochs
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        train_dataloader (torch.utils.data.DataLoader): Training Set data loader
        val_dataloader (torch.utils.data.DataLoader): Validation Set data loader
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        device (str): Device to be used
        per_epoch_train_func (function): Training function for a single epoch
        val_func (function): Validation function
        wandb_project (str): Name of the W&B project
        wandb_run (str): Name of the W&B run for logging results
    """

    model = model.to(device=device)
    model.train()

    # Initialize variables
    best_model_weights = None
    best_val_acc = 0.0
    best_val_f1_score = 0.0

    # add wandb logging
    print("Starting wandb logging...")

    run = wandb.init(project=wandb_project, name=wandb_run)
    # define the custom x-axis metrics
    wandb.define_metric("train/epoch")
    wandb.define_metric("val/epoch")
    # define which metrics will be plotted against it
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")

    # Train model
    for epoch in tqdm(range(1, num_epochs + 1)):
        # Perform one epoch of training
        train_loss, train_acc, lr, train_f1_score = per_epoch_train_func(
            model, scheduler, train_dataloader, criterion, optimizer, epoch, device
        )

        # Perform one epoch of validation
        val_loss, val_acc, val_f1_score, _, _, _, _, _, _ = val_func(
            model, criterion, val_dataloader, device
        )

        # Save best model based on the highest F1 Score on the validation set
        if val_f1_score > best_val_f1_score:
            best_val_acc = val_acc
            best_val_f1_score = val_f1_score
            best_model_weights = deepcopy(model.state_dict())

        # Plot training curves
        run.log(
            {
                "train/epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/f1-score": train_f1_score,
                "train/lr": lr,
                "val/epoch": epoch,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1-score": val_f1_score,
            }
        )

    print("wandb logging finished!")
    wandb.finish()

    print("Saving best model...")
    torch.save(
        best_model_weights,
        os.path.join(main_save_dir, "checkpoints", f"{wandb_run}.pt"),
    )
