import ultralytics
import wandb
import argparse
import os
import json
from ultralytics import YOLO

# ----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # execution settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_file", type=str)
    parser.add_argument("--custom_dataset", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run", type=str)
    parser.add_argument("--num_epochs", type=int)

    # extract arguments
    args = parser.parse_args()
    PRETRAINED_MODEL_FILE = args.pretrained_model_file
    CUSTOM_DATASET = args.custom_dataset
    WANDB_PROJECT = args.wandb_project
    WANDB_RUN = args.wandb_run
    NUM_EPOCHS = args.num_epochs
    
    SINGLE_CLS_FLAG = True

    print("Running ultralytics checks")
    ultralytics.checks()

    print("Loading the pretrained YOLO11 model")
    model = YOLO(PRETRAINED_MODEL_FILE)

    wandb.init(allow_val_change=True)

    print("Training the model")

    main_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )

    results = model.train(
        data=os.path.join(main_dir, "data", CUSTOM_DATASET, "data.yaml"),
        epochs=NUM_EPOCHS,
        imgsz=512,
        batch=64,
        single_cls=SINGLE_CLS_FLAG,
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        augment=False,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        erasing=0.4,
    )

    # Finish and then disable wandb logging
    wandb.finish()
    wandb.init(mode="disabled")

    print("Loading trained YOLO model for inference")
    model = YOLO(
        os.path.join(
            main_dir, "src", "models", WANDB_PROJECT, WANDB_RUN, "weights", "best.pt"
        )
    )

    print("Evaluating val set performance")
    val_det_metrics = model.val(
        batch=64, imgsz=512, project=WANDB_PROJECT, name=WANDB_RUN + "_Val_Metrics"
    )
    val_metrics = val_det_metrics.results_dict

    with open(
        os.path.join(main_dir, "src", "models", WANDB_PROJECT, WANDB_RUN)
        + "/_Val_Metrics/val_metrics.json",
        "w",
    ) as v:
        json.dump(val_metrics, v)

    print("Evaluating test set performance")
    test_det_metrics = model.val(
        split="test",
        imgsz=512,
        batch=64,
        project=WANDB_PROJECT,
        name=WANDB_RUN + "_Test_Metrics",
    )
    test_metrics = test_det_metrics.results_dict

    with open(
        os.path.join(main_dir, "src", "models", WANDB_PROJECT, WANDB_RUN)
        + "/_Test_Metrics/test_metrics.json",
        "w",
    ) as t:
        json.dump(test_metrics, t)

    print("Saving val set predictions")
    model(
        os.path.join(main_dir, "data", CUSTOM_DATASET, "images", "val"),
        imgsz=512,
        save=True,
        save_txt=True,
        save_conf=True,
        project=WANDB_PROJECT,
        name=WANDB_RUN + "_Val_Predictions",
    )

    print("Saving test set predictions")
    model(
        os.path.join(main_dir, "data", CUSTOM_DATASET, "images", "test"),
        imgsz=512,
        save=True,
        save_txt=True,
        save_conf=True,
        project=WANDB_PROJECT,
        name=WANDB_RUN + "_Test_Predictions",
    )

    print("Done!")
