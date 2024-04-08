import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_log_error
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import r2_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_edge_prediction_metrics(predicts: np.ndarray, labels: np.ndarray):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    # predicts = predicts.cpu().detach().numpy()
    # labels = labels.cpu().numpy()

    index = labels != 0
    labels = labels[index]
    predicts = predicts[index]

    if len(labels[labels < 0]) > 0:
        lmse_val = 0
    else:
        predicts[predicts < 0] = 0
        lmse_val = mean_squared_log_error(labels, predicts)

    mae = mean_absolute_error(labels, predicts)
    rmse = np.sqrt(mean_squared_error(labels, predicts))
    r2 = r2_score(labels, predicts)
    corr, p_value = pearsonr(labels, predicts)



    mape = 100 * np.abs((predicts - labels) / (predicts + 1e-10)).sum() / len(predicts)
    # mape =
    # pcc =

    # average_precision = average_precision_score(y_true=labels, y_score=predicts)
    # roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'lmse': lmse_val, 'r2': r2, 'pcc': corr, 'p_value': p_value}
    # return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
