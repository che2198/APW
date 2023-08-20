import numpy as np

def equalized_odds_difference(predictions, true_labels, sensitive_features, target_label):

    binary_predictions = np.where(predictions > 0.5, 1, 0)
    positive_mask = true_labels == target_label

    tpr_0 = np.mean(binary_predictions[(sensitive_features == 0) & positive_mask] == target_label)
    tpr_1 = np.mean(binary_predictions[(sensitive_features == 1) & positive_mask] == target_label)

    return np.abs(tpr_0 - tpr_1)

def demographic_parity_difference(predictions, sensitive_features):

    binary_predictions = np.where(predictions > 0.5, 1, 0)

    rate_0 = np.mean(binary_predictions[sensitive_features == 0])
    rate_1 = np.mean(binary_predictions[sensitive_features == 1])

    return np.abs(rate_0 - rate_1)