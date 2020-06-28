import numpy as np

EPSILON = 1e-12


def _dice_ratio(pred, gt):
    """
    calculate the dice ratio for segmentation
    just for one category
    ==============================================
    input:
    pred: prediction
    gt: ground truth

    output:
    dice: dice score
    """
    if pred.shape != gt.shape:
        raise ValueError("The shape of prediction and ground truth must be equal.")
    dice = (2 * 1.0 * np.sum((pred * gt).astype(np.uint8)) + EPSILON) / \
           (np.sum(pred.astype(np.uint8)) + np.sum(gt.astype(np.uint8)) + EPSILON)

    return dice


def dice_ratio(pred, gt, report=False):
    """
    calculate the dice ratio for segmentation
    for multi categories
    In this function the default value of background is 0
    =====================================================
    input:
    pred: prediction
    gt: ground truth
    report: option for return the dice score of each class

    output:
    dice: dice score
    report_dict: dice score of each class
    """
    if pred.shape != gt.shape:
        raise ValueError("The shape of prediction and ground truth must be equal.")

    category_list = []
    report_dict = dict()
    id_list = np.unique(gt)

    if len(id_list) == 1 and id_list[0] == 0:
        raise ValueError("The ground truth only contain background 0")

    for c in id_list:
        if c == 0:
            pass
        else:
            pred_c = pred == c
            gt_c = gt == c
            dice_score = _dice_ratio(pred_c, gt_c)
            category_list.append(dice_score)
            report_dict[c] = dice_score
    if report:
        return np.mean(category_list), report_dict
    else:
        return np.mean(category_list)

