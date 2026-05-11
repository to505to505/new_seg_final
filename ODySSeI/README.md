# <div align="center">ODySSeI</div>

<div align="center">
An <b>O</b>pen-Source End-to-End Framework for Automated <b>D</b>etection, <b>S</b>egmentation, and <b>S</b>everity Estimation of Lesions in <b>I</b>nvasive Coronary Angiography Images
</div>

<div align="center">
  <img width="128" alt="logo" src="https://github.com/user-attachments/assets/9d079da1-b285-401c-8d35-8580d93da842" />
</div>

## Web Interface

ODySSeI is live at [swisscardia.epfl.ch](https://swisscardia.epfl.ch).

Please find below a demo of ODySSeI's web interface. Demo images are available in the ``demo_images`` folder within the ``data`` folder.

https://github.com/user-attachments/assets/ae494d67-9bd1-4f03-9e33-3bbcf05dcff3


## Setting Up the Repository

To use ODySSeI, please run the following code snippet in your terminal:

```bash
git clone https://github.com/LTS4/ODySSeI
cd odyssei
python -m pip install -e .
```

Next, please place your custom ICA dataset folder, ``custom_dataset``, within the ``data`` folder.

## Training, Validation, and Testing of our Lesion Detection Model (YOLO11m)

Please run the following code snippet in your terminal:

```bash
python odyssei/lesion_detection_train_val_test.py --pretrained_model_file=PRETRAINED_MODEL_FILE --custom_dataset=CUSTOM_DATASET --wandb_project=WANDB_PROJECT --wand_run=WANDB_RUN --num_epochs=NUM_EPOCHS
```
Here,
- PRETRAINED_MODEL_FILE (str) = Pretrained Model Weights (yolo11m.pt)
- CUSTOM_DATASET (str) = Name of the ``custom_dataset`` folder in the ``data`` folder; Please follow the [YOLO dataset format](https://docs.ultralytics.com/datasets/detect/)
- WANDB_PROJECT (str) = Name of the W&B Project
- WANDB_RUN (int) = Name of the W&B Run for Logging Results
- NUM_EPOCHS (int) = Number of Training Epochs

## Training, Validation, and Testing of Lesion Segmentation Models (DeepLabv3+ and U-Net)

Please run the following code snippet in your terminal:

For DeepLabv3+:
```bash
python odyssei/deeplab_lesion_segmentation_train_val_test.py --pretrained_model_file=PRETRAINED_MODEL_FILE --custom_dataset=CUSTOM_DATASET --wandb_project=WANDB_PROJECT --wand_run=WANDB_RUN --num_epochs=NUM_EPOCHS
```

For U-Net:
```bash
python odyssei/unet_lesion_segmentation_train_val_test.py --custom_dataset=CUSTOM_DATASET --wandb_project=WANDB_PROJECT --wand_run=WANDB_RUN --seed=SEED --batch_size=BATCH_SIZE --num_epochs=NUM_EPOCHS --loss_func=LOSS_FUNC
```

Here, 
- CUSTOM_DATASET (str) = Name of the ``custom_dataset`` folder in the ``data`` folder; Please follow the following structure:
    - ```
      CUSTOM_DATASET (Name of the Custom Dataset)
      |- images (This is the subfolder where you need to save your ICA images)
      |- masks (This is the subfolder where you need to save your corresponding ground truth segmentation masks)
      ```
- WANDB_PROJECT (str) = Name of the W&B Project
- WANDB_RUN (int) = Name of the W&B Run for Logging Results
- SEED (int) = Seed for Reproducibility
- BATCH_SIZE (int) = Batch Size (Recommended: 16)
- NUM_EPOCHS (int) = Number of Training Epochs
- LOSS_FUNC (int) = Loss Function, e.g., bce1, bce2, bce3, iou, dice, cldice, dicecldice, dicebce1, dicebce2 (Recommended: dicebce2)

