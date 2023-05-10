import torch
from sklearn import metrics


def confusion_matrix(true_labels, pred_labels, pos_label=1):
    # returns
    #   (tp, fp, fn, tn)

    # tp : true positives (# correctly labeled as blue)
    tp = torch.sum(pred_labels[true_labels == pos_label] == pos_label)

    # tn : true negatives (# correctly labeled as red)
    tn = torch.sum(pred_labels[true_labels != pos_label] != pos_label)

    # fp : false positives (# incorrectly labeled as blue)
    fp = torch.sum(pred_labels[true_labels != pos_label] == pos_label)

    # fn : false negative (# incorrectly labeled as red)
    fn = torch.sum(pred_labels[true_labels == pos_label] != pos_label)

    return tp, fp, fn, tn


def compute_statistics(true_labels, pred_labels, pos_label=1):

    # confusion matrix
    tp, fp, fn, tn = confusion_matrix(true_labels.reshape(-1), pred_labels.reshape(-1), pos_label=pos_label)

    # total number of positives
    p = tp + fn

    # total number of negatives
    n = fp + tn

    # accuracy
    acc = (tp + tn) / (p + n)

    # true positive rate (sensitivity, recall, hit rate, 1 - fnr)
    tpr = torch.tensor(torch.nan) if p == 0 else tp / p

    # true negative rate (specificity, selectivity, 1 - fpr)
    tnr = torch.tensor(torch.nan) if n == 0 else tn / n

    # false positive rate (fall-out, 1 - tnr)
    fpr = torch.tensor(torch.nan) if n == 0 else fp / n

    # false negative rate (miss rate, 1 - tpr)
    fnr = torch.tensor(torch.nan) if p == 0 else fn / p

    # precision (positive predictive value, 1 - fdr)
    ppv = tp / (tp + fp)

    # negative predictive value (1 - FOR)
    npv = tn / (tn + fn)

    # false discovery rate (1 - ppv)
    fdr = fp / (fp + tp)

    # false omission rate (1 - npv)
    FOR = fn / (fn + tn)

    # f1 score
    f1 = 2 * (ppv * tpr) / (ppv + tpr)

    info = {'TP': tp.item() if isinstance(tp, torch.Tensor) else tp,
            'FP': fp.item(),
            'FN': fn.item(),
            'TN': tn.item(),
            'P': p.item(), 'N': n.item(), 'accuracy': acc.item(), 'F1': f1.item(),
            'TPR': tpr.item(), 'FPR': fpr.item(), 'TNR': tnr.item(), 'FNR': fnr.item(),
            'PPV': ppv.item(), 'NPV': npv.item(), 'FDR': fdr.item(), 'FOR': FOR.item()}

    return info


def independence(y_true, y_pred, s):
    ind_s0_y0 = (y_pred[s == 0] == 0).sum() / (s == 0).sum()
    ind_s1_y0 = (y_pred[s == 1] == 0).sum() / (s == 1).sum()

    ind_s0_y1 = (y_pred[s == 0] == 1).sum() / (s == 0).sum()
    ind_s1_y1 = (y_pred[s == 1] == 1).sum() / (s == 1).sum()

    out = {'y = 0': {'s = 0': ind_s0_y0.item(), 's = 1': ind_s1_y0.item()},
           'y = 1': {'s = 0': ind_s0_y1.item(), 's = 1': ind_s1_y1.item()}
           }

    return out


def separation(y_true, y_pred, s):
    sep_s0_y0 = y_pred[(y_true == 0) * (s == 0)].sum() / ((y_true == 0) * (s == 0)).sum()
    sep_s1_y0 = y_pred[(y_true == 0) * (s == 1)].sum() / ((y_true == 0) * (s == 1)).sum()

    sep_s0_y1 = y_pred[(y_true == 1) * (s == 0)].sum() / ((y_true == 1) * (s == 0)).sum()
    sep_s1_y1 = y_pred[(y_true == 1) * (s == 1)].sum() / ((y_true == 1) * (s == 1)).sum()

    out = {'y = 0': {'s = 0': sep_s0_y0.item(), 's = 1': sep_s1_y0.item()},
           'y = 1': {'s = 0': sep_s0_y1.item(), 's = 1': sep_s1_y1.item()}
           }
    return out


def sufficiency(y_true, y_pred, s):
    suf_s0_r0 = y_true[(y_pred == 0) * (s == 0)].sum() / ((y_pred == 0) * (s == 0)).sum()
    suf_s1_r0 = y_true[(y_pred == 0) * (s == 1)].sum() / ((y_pred == 0) * (s == 1)).sum()

    suf_s0_r1 = y_true[(y_pred == 1) * (s == 0)].sum() / ((y_pred == 1) * (s == 0)).sum()
    suf_s1_r1 = y_true[(y_pred == 1) * (s == 1)].sum() / ((y_pred == 1) * (s == 1)).sum()

    out = {'r = 0': {'s = 0': suf_s0_r0.item(), 's = 1': suf_s1_r0.item()},
           'r = 1': {'s = 0': suf_s0_r1.item(), 's = 1': suf_s1_r1.item()}
           }
    return out


def fairness_metrics(y_true, y_pred, s):
    out_ind = independence(y_true, y_pred, s)
    out_sep = separation(y_true, y_pred, s)
    out_suf = sufficiency(y_true, y_pred, s)

    out = {'independence': out_ind,
           'separation': out_sep,
           'sufficiency': out_suf
           }
    return out


def store_statistics(z, y_pred, x, y):
    out = compute_statistics(y, y_pred)
    cm = metrics.confusion_matrix(y, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y, z.detach(), pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {'stats': out, 'cm': cm, 'fpr': fpr, 'tpr': tpr, 'auc': auc}








