import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from skimage.morphology import skeletonize
from skimage.metrics import hausdorff_distance

from utils.data_seg import reverse_img_normalization

import warnings

warnings.filterwarnings("ignore")


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Ref: [Official Repo] https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v * s) / np.sum(s)


@torch.no_grad()
def metrics_train(batched_pred_net, batched_target_net):
    """
    Compute metrics on the training set

    Arguments:
        batched_pred_net (torch.tensor): Batch of all the model predictions
        batched_target_net (torch.tensor): Batch of all the corresponding ground truths

    Returns:
        mean_accuracy (float): Average accuracy
        mean_f1_score (float): Average F1-Score
    """

    accuracy_net = []
    f1_score_net = []

    num_batches = batched_pred_net.shape[0]

    for b in range(num_batches):
        pred_net = batched_pred_net[b, :, :, :]
        target_net = batched_target_net[b, :, :, :]

        TP = ((pred_net == 1) & (target_net == 1)).sum().item()
        TN = ((pred_net == 0) & (target_net == 0)).sum().item()
        FP = ((pred_net == 1) & (target_net == 0)).sum().item()
        FN = ((pred_net == 0) & (target_net == 1)).sum().item()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy_net.append(accuracy)

        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        if precision + recall > 0:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = 0

        f1_score_net.append(f1_score)

    mean_accuracy, mean_f1_score = np.mean(accuracy_net), np.mean(f1_score_net)

    return mean_accuracy, mean_f1_score


@torch.no_grad()
def metrics(batched_pred_net, batched_target_net, compute_std_dev=False):
    """
    Compute metrics on the validation/test set

    Arguments:
        batched_pred_net (torch.tensor): Batch of all the model predictions
        batched_target_net (torch.tensor): Batch of all the corresponding ground truths
        compute_std_net (bool): Flag to decide whether/not to compute the standard deviation of the metrics

    Returns:
        final_accuracy (float/list): Average accuracy/Average and std. dev. of accuracy
        final_precision (float/list): Average precision/Average and std. dev. of precision
        final_recall (float/list): Average recall/Average and std. dev. of recall
        final_f1_score (float/loss): Average F1-Score/Average and std. dev. of F1-Score
        final_iou (float/loss): Average IoU/Average and std. dev. of IoU
        final_cldice (float/loss): Average clDice/Average and std. dev. of clDice
        final_hd (float/loss): Average HD/Average and std. dev. of HD
        final_mhd (float/loss): Average MHD/Average and std. dev. of MHD
    """

    accuracy_net = []
    precision_net = []
    recall_net = []
    f1_score_net = []
    iou_net = []
    cldice_net = []
    hd_net = []
    mhd_net = []

    num_batches = batched_pred_net.shape[0]

    for b in range(num_batches):
        pred_net = batched_pred_net[b, :, :, :]
        target_net = batched_target_net[b, :, :, :]

        TP = ((pred_net == 1) & (target_net == 1)).sum().item()
        TN = ((pred_net == 0) & (target_net == 0)).sum().item()
        FP = ((pred_net == 1) & (target_net == 0)).sum().item()
        FN = ((pred_net == 0) & (target_net == 1)).sum().item()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy_net.append(accuracy)

        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        precision_net.append(precision)

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        recall_net.append(recall)

        if precision + recall > 0:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = 0

        f1_score_net.append(f1_score)

        if TP + FP + FN > 0:
            iou = TP / (TP + FP + FN)
        else:
            iou = 0

        iou_net.append(iou)

        pred_net_arr = pred_net.permute(1, 2, 0).squeeze(2).cpu().numpy()
        target_net_arr = target_net.permute(1, 2, 0).squeeze(2).cpu().numpy()

        tprec = cl_score(pred_net_arr, skeletonize(target_net_arr))
        tsens = cl_score(target_net_arr, skeletonize(pred_net_arr))

        if tprec + tsens > 0:
            cldice = 2 * tprec * tsens / (tprec + tsens)
        else:
            cldice = 0

        cldice_net.append(cldice)

        if np.max(pred_net_arr) < 1.0 or np.max(target_net_arr) < 1.0:
            hd = np.nan
            mhd = np.nan
        else:
            hd = hausdorff_distance(pred_net_arr, target_net_arr, method="standard")
            mhd = hausdorff_distance(pred_net_arr, target_net_arr, method="modified")

        hd_net.append(hd)
        mhd_net.append(mhd)

    (
        mean_accuracy,
        mean_precision,
        mean_recall,
        mean_f1_score,
        mean_iou,
        mean_cldice,
        mean_hd,
        mean_mhd,
    ) = (
        np.mean(accuracy_net),
        np.mean(precision_net),
        np.mean(recall_net),
        np.mean(f1_score_net),
        np.mean(iou_net),
        np.mean(cldice_net),
        np.nanmean(hd_net),
        np.nanmean(mhd_net),
    )

    (
        final_accuracy,
        final_precision,
        final_recall,
        final_f1_score,
        final_iou,
        final_cldice,
        final_hd,
        final_mhd,
    ) = (
        mean_accuracy,
        mean_precision,
        mean_recall,
        mean_f1_score,
        mean_iou,
        mean_cldice,
        mean_hd,
        mean_mhd,
    )

    if compute_std_dev:
        (
            std_dev_accuracy,
            std_dev_precision,
            std_dev_recall,
            std_dev_f1_score,
            std_dev_iou,
            std_dev_cldice,
            std_dev_hd,
            std_dev_mhd,
        ) = (
            np.std(accuracy_net, ddof=1),
            np.std(precision_net, ddof=1),
            np.std(recall_net, ddof=1),
            np.std(f1_score_net, ddof=1),
            np.std(iou_net, ddof=1),
            np.std(cldice_net, ddof=1),
            np.nanstd(hd_net, ddof=1),
            np.nanstd(mhd_net, ddof=1),
        )

        (
            final_accuracy,
            final_precision,
            final_recall,
            final_f1_score,
            final_iou,
            final_cldice,
            final_hd,
            final_mhd,
        ) = (
            [mean_accuracy, std_dev_accuracy],
            [mean_precision, std_dev_precision],
            [mean_recall, std_dev_recall],
            [mean_f1_score, std_dev_f1_score],
            [mean_iou, std_dev_iou],
            [mean_cldice, std_dev_cldice],
            [mean_hd, std_dev_hd],
            [mean_mhd, std_dev_mhd],
        )

    return (
        final_accuracy,
        final_precision,
        final_recall,
        final_f1_score,
        final_iou,
        final_cldice,
        final_hd,
        final_mhd,
    )


@torch.no_grad()
def test_fn(model, criterion, dataloader, device, compute_metric_std_dev=False):
    """
    Evaluate the model's performance on the validation/test set
    Args:
        model (torch.nn.Module): Model to be evaluated
        criterion (torch.nn.Module): Loss function
        dataloader (torch.utils.data.DataLoader): Testing/Validation Set data loader
        device (str): Device to use.
        compute_metric_std_net (bool): Flag to decide whether/not to compute the standard deviation of the metrics

    Returns:
        loss (float): Average loss
        accuracy (float/list): Average accuracy/Average and std. dev. of accuracy
        f1_score (float/loss): Average F1-Score/Average and std. dev. of F1-Score
        precision (float/list): Average precision/Average and std. dev. of precision
        recall (float/list): Average recall/Average and std. dev. of recall
        iou (float/loss): Average IoU/Average and std. dev. of IoU
        cldice (float/loss): Average clDice/Average and std. dev. of clDice
        hd (float/loss): Average HD/Average and std. dev. of HD
        mhd (float/loss): Average MHD/Average and std. dev. of MHD
    """
    model.eval()

    # Initialize model variables
    pred_net = []
    output_net = []
    target_net = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        preds = (outputs.sigmoid() > 0.5).to(torch.int8)

        # Append all outputs
        output_net.append(outputs)
        pred_net.append(preds)
        target_net.append(targets)

    output_net = torch.cat(output_net, dim=0)
    pred_net = torch.cat(pred_net, dim=0)
    target_net = torch.cat(target_net, dim=0)

    # Calculate loss and metrics
    loss = criterion(output_net, target_net).item()
    acc, precision, recall, f1_score, iou, cldice, hd, mhd = metrics(
        pred_net, target_net, compute_metric_std_dev
    )

    # Print metrics

    if not compute_metric_std_dev:
        print(
            "Averaged Eval Metrics: loss: {:.3f}, F1-Score: {:.3f} (Precision: {:.3f}, Recall: {:.3f}), Accuracy: {:.3f}, IOU: {:.3f}, Cl-Dice: {:.3f}, HD: {:.3f}, MHD: {:.3f}".format(
                loss, f1_score, precision, recall, acc, iou, cldice, hd, mhd
            )
        )

    return loss, acc, f1_score, precision, recall, iou, cldice, hd, mhd


def show_overlay(img, mask, prediction, main_plot_name, rescale=False, data_id=2):
    """
    Plot the results of a single image, its ground truth mask, and corresponding prediction.

    Args:
        img (torch.Tensor): Image
        mask (torch.Tensor): Ground truth mask
        prediction (torch.Tensor): Prediction
        rescale (bool): Flag to decide whether/not to rescale the images
        data_id (int): ID of the dataset for retrieving stats for reversal of image normalization
    """

    _, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # Invert normalization to get better plots
    if rescale:
        img = reverse_img_normalization(img, data_id=data_id)

    # Make sure dimensions are consistent
    img = torch.permute(img, (1, 2, 0)).cpu().numpy()
    mask = torch.permute(mask, (1, 2, 0)).cpu().numpy()
    prediction = torch.permute(prediction.detach(), (1, 2, 0)).cpu().numpy()

    # Display ground truth on the left subplot
    axes[0].imshow(img)
    axes[0].imshow(mask, alpha=0.5)
    axes[0].axis("off")
    axes[0].set_title("Ground Truth")

    # Display prediction heatmap on the right subplot
    axes[1].imshow(img)
    heatmap = axes[1].imshow(prediction, cmap="viridis", alpha=0.5)
    axes[1].axis("off")
    axes[1].set_title("Prediction Heatmap")

    # Add colorbar for the heatmap
    cbar = plt.colorbar(heatmap, ax=axes[1], orientation="vertical")
    cbar.set_label("Prediction Confidence")

    plt.savefig(f"{main_plot_name}.pdf")


def show_results_tensor(
    batch_image, batch_mask, batch_prediction, main_plot_name, rescale=False, data_id=2
):
    """
    Plot the results of a batch of images, masks and predictions

    Args:
        batch_image (torch.Tensor): Batch of images
        batch_mask (torch.Tensor): Batch of ground truth masks
        batch_prediction (torch.Tensor): Batch of predictions
        main_plot_name (str): Name of the plot to be saved
        rescale (bool): Flag to decide whether/not to rescale the images
        data_id (int): ID of the dataset for retrieving stats for reversal of image normalization
    """
    batch_size = batch_image.shape[0]

    print("Visualizing {} examples".format(batch_size))

    # Computing the probabilities of the predictions
    batch_prediction = batch_prediction.sigmoid()

    for index in range(batch_size):
        image = batch_image[[index], :, :, :].squeeze(0)
        mask = batch_mask[[index], :, :, :].squeeze(0)
        prediction = batch_prediction[[index], :, :, :].squeeze(0)

        show_overlay(
            image,
            mask,
            prediction,
            main_plot_name + "/plot_" + str(index),
            rescale=rescale,
            data_id=data_id,
        )
