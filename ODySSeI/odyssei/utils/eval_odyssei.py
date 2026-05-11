import numpy as np
import pandas as pd
import torch
import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
from tqdm import tqdm
from torchvision.utils import save_image

import warnings

warnings.filterwarnings("ignore")


def compute_per_lesion_severity(img):
    """
    Algorithm to compute lesion severity

    Arguments:
        img (numpy.array): Predicted segmentation mask
    """

    dist_transform_mat = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_transform_mat = dist_transform_mat.transpose()

    skele = skeletonize(img)

    pts = np.column_stack(np.where(skele.transpose() == 1))

    sorted_pts = sorted(pts.tolist(), key=lambda x: x[0])
    dist_net = []

    for central_point in sorted_pts:
        dist_net.append(
            2 * dist_transform_mat[int(central_point[0]), int(central_point[1])]
        )

    dist_net = np.array(dist_net)

    peaks, _ = find_peaks(dist_net)

    if len(peaks) == 0:
        MLD = np.nan
        DS = np.nan
    elif len(peaks) == 1:
        MLD = min(dist_net[peaks[0] : -1])
        DS = 1 - (MLD / max(dist_net))
    else:
        MLD = min(dist_net[peaks[0] : peaks[-1]])
        DS = 1 - (MLD / max(dist_net))

    severity = [MLD, DS]

    return severity


@torch.no_grad()
def predict_and_save(model, dataloader, device, orig_lc_sizes_df, main_save_dir):
    """
    Predict lesion segmentation masks and estimate lesion severity

    Args:
        model (torch.nn.Module): Model
        dataloader (torch.utils.data.DataLoader): Data loader
        device (str): Device to be use
        orig_lc_sizes_df (pandas.DataFrame): Dataframe containing the original heights and widths of the cropped lesion images before resizing them
        main_save_dir (str): Main directory for saving predicted segmentation masks and lesion severity metrics

    """
    model.eval()

    # Initialize model variables
    pred_net = []

    for inputs in dataloader:
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)
        preds = (outputs.sigmoid() > 0.5).to(torch.int8)

        # Append all outputs
        pred_net.append(preds)

    pred_net = torch.cat(pred_net, dim=0)

    num_batches = pred_net.shape[0]
    lc_img_name_net = orig_lc_sizes_df["image_name"].values
    height_net = orig_lc_sizes_df["height"].values
    width_net = orig_lc_sizes_df["width"].values

    pred_severity_net = []

    os.mkdir(os.path.join(main_save_dir, "LSEG_RES"))

    for b in tqdm(range(num_batches)):
        pred = pred_net[b, :, :, :]

        pred_float = pred.float()

        save_image(
            pred_float,
            os.path.join(main_save_dir, "LSEG_RES", "LSEG_" + lc_img_name_net[b]),
        )

        pred_net_arr = pred.permute(1, 2, 0).squeeze(2).cpu().numpy()

        pred_net_arr = A.Resize(height=height_net[b], width=width_net[b])(
            image=pred_net_arr.astype(np.uint8)
        )["image"]

        pred_severity = compute_per_lesion_severity(pred_net_arr.astype(np.uint8))

        pred_severity_net.append(
            {
                "image_name": lc_img_name_net[b],
                "MLD": pred_severity[0],
                "DS": pred_severity[1],
            }
        )

    pred_severity_df = pd.DataFrame(pred_severity_net)

    pred_severity_df.to_csv(
        os.path.join(main_save_dir, "predicted_lesion_severity.csv"),
        encoding="utf-8-sig",
        index=False,
    )
