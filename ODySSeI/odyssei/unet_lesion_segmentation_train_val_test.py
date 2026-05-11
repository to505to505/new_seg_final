import wandb
import argparse
import os
import torch
import json
import gc
import numpy as np
import matplotlib.pyplot as plt

from utils.data_seg import seed_everything, build_dataloaders
from models.UNet_5_encoder_layers_with_dropout import UNet
from utils.loss_seg import (
    iou_loss,
    dice_loss,
    dice_with_bce_loss,
    soft_cldice,
    soft_dice_cldice,
)
from utils.train_seg import kaiming_init, train
from utils.eval_seg import test_fn, show_results_tensor

import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # execution settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_dataset", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run", type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--loss_func", type=str)

    # extract arguments
    args = parser.parse_args()
    CUSTOM_DATASET = args.custom_dataset
    WANDB_PROJECT = args.wandb_project
    WANDB_RUN = args.wandb_run
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LOSS_FUNC = args.loss_func

    ROTATE = False
    PERSPECTIVE = False
    ERASE = True
    SCALE = True
    HFLIP = True
    TRANSLATE = True
    HSV = True

    print("Seeding everything...")
    seed_everything(SEED)

    DATA_ID = 2

    main_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )

    print("Building the dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        os.path.join(main_dir, "data", CUSTOM_DATASET),
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        transform=True,
        data_id=DATA_ID,
        rotate=ROTATE,
        perspective=PERSPECTIVE,
        erase=ERASE,
        scale=SCALE,
        hflip=HFLIP,
        translate=TRANSLATE,
        hsv=HSV,
    )

    print("Loading the model...")
    model = UNet(main_C_in=3, num_classes=1, add_batch_norm=False)
    model.apply(kaiming_init)

    # Loss Function Determination
    if LOSS_FUNC.startswith("bce"):
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                int(LOSS_FUNC[-1]),
            )
        )
    elif LOSS_FUNC == "dice":
        criterion = dice_loss
    elif LOSS_FUNC == "iou":
        criterion = iou_loss
    elif LOSS_FUNC.startswith("dicebce"):
        criterion = dice_with_bce_loss(bce_pos_w=int(LOSS_FUNC[-1]))
    elif LOSS_FUNC == "cldice":
        criterion = soft_cldice()
    elif LOSS_FUNC == "dicecldice":
        criterion = soft_dice_cldice()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * NUM_EPOCHS) // train_loader.batch_size,
    )

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(DEVICE)

    print("Training and validating the model...")
    train(
        model=model,
        main_save_dir=main_dir,
        num_epochs=NUM_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        scheduler=scheduler,
        device=DEVICE,
        wandb_project=WANDB_PROJECT,
        wandb_run=WANDB_RUN,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading the best model...")
    model = UNet(main_C_in=3, num_classes=1, add_batch_norm=False)
    model.load_state_dict(
        torch.load(
            os.path.join(main_dir, "checkpoints", f"{WANDB_RUN}.pt"),
            weights_only=True,
            map_location=DEVICE,
        )
    )
    model.to(DEVICE)

    os.mkdir(os.path.join(main_dir, "checkpoints", WANDB_RUN))

    print("Performing the final evaluation of the model...")
    print("TEST SET")
    test_metrics = dict()
    (
        test_metrics["loss"],
        test_metrics["acc"],
        test_metrics["f1_score"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["iou"],
        test_metrics["cldice"],
        test_metrics["hd"],
        test_metrics["mhd"],
    ) = test_fn(model, criterion, test_loader, DEVICE, compute_metric_std_dev=True)
    with open(
        os.path.join(main_dir, "data") + "/" + WANDB_RUN + "/test_metrics.json", "w"
    ) as t:
        json.dump(test_metrics, t)

    print("VALIDATION SET")
    val_metrics = dict()
    (
        val_metrics["loss"],
        val_metrics["acc"],
        val_metrics["f1_score"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["iou"],
        val_metrics["cldice"],
        val_metrics["hd"],
        val_metrics["mhd"],
    ) = test_fn(model, criterion, val_loader, DEVICE, compute_metric_std_dev=True)
    with open(
        os.path.join(main_dir, "data") + "/" + WANDB_RUN + "/val_metrics.json", "w"
    ) as v:
        json.dump(val_metrics, v)

    print("TRAINING SET")
    training_metrics = dict()
    (
        training_metrics["loss"],
        training_metrics["acc"],
        training_metrics["f1_score"],
        training_metrics["precision"],
        training_metrics["recall"],
        training_metrics["iou"],
        training_metrics["cldice"],
        training_metrics["hd"],
        training_metrics["mhd"],
    ) = test_fn(model, criterion, train_loader, DEVICE, compute_metric_std_dev=True)
    with open(
        os.path.join(main_dir, "data") + "/" + WANDB_RUN + "/training_metrics.json", "w"
    ) as tr:
        json.dump(training_metrics, tr)

    print("Visualizing predictions...")
    data_to_show = iter(test_loader)
    model.eval()
    model.to(DEVICE)
    batch_x, batch_y = next(data_to_show)
    batch_x = batch_x.to(DEVICE)
    output = model(batch_x)
    show_results_tensor(
        batch_x,
        batch_y,
        output,
        main_plot_name=os.path.join(main_dir, "data") + "/" + WANDB_RUN,
        rescale=True,
        data_id=DATA_ID,
    )
