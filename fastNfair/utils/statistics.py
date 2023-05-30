import torch
from sklearn import metrics


def confusion_matrix(true_labels, pred_labels, pos_label=1):
    # returns
    #   (tp, fp, fn, tn)
    # TODO: implement!

    # tp : true positives (# correctly labeled as blue)
    tp = torch.ones(1)

    # tn : true negatives (# correctly labeled as red)
    tn = torch.ones(1)

    # fp : false positives (# incorrectly labeled as blue)
    fp = torch.ones(1)

    # fn : false negative (# incorrectly labeled as red)
    fn = torch.ones(1)

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
    # TODO: implement!

    out = {'y = 0': {'s = 0': 0.0, 's = 1': 0.0},
           'y = 1': {'s = 0': 0.0, 's = 1': 0.0}
           }

    return out


def separation(y_true, y_pred, s):
    # TODO: improve this implementation
    print('y_true: ', y_true)
    print('y_pred: ', y_pred)
    total_s0 = y_true[s == 0].numel()
    total_s1 = y_true[s == 1].numel()
    print('total_s0: ', total_s0)

    # calculating false positives
    y_pred_negs = y_pred[(y_true == 0)] # y_pred only where y_true = 0
    s_negs = s[(y_true == 0)] # s only where y_true = 0
    num_fp0 = y_pred_negs[(s_negs == 0)].sum()
    num_fp1 = y_pred_negs[(s_negs == 1)].sum()

    # calculating true positives
    y_pred_pos = y_pred[(y_true == 1)]  # y_pred only where y_true = 1
    s_pos = s[(y_true == 1)]  # s only where y_true = 1
    num_tp0 = y_pred_pos[(s_pos == 0)].sum()
    num_tp1 = y_pred_pos[(s_pos == 1)].sum()

    fp0 = num_fp0 / total_s0
    fp1 = num_fp1 / total_s1
    tp0 = num_tp0 / total_s0
    tp1 = num_tp1 / total_s1

    out = {'y = 0': {'s = 0': fp0, 's = 1': fp1},
           'y = 1': {'s = 0': tp0, 's = 1': tp1}
           }
    print(out)
    return out


def sufficiency(y_true, y_pred, s):
    # TODO: implement!

    out = {'r = 0': {'s = 0': 0.0, 's = 1': 0.0},
           'r = 1': {'s = 0': 0.0, 's = 1': 0.0}
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


if __name__ == "__main__":
    # create synthetic data

    # for reproducibility
    torch.manual_seed(42)

    # choose number of samples
    n_samples = 10

    # true label
    y_true = torch.zeros(n_samples, dtype=torch.int8)
    y_true[torch.randperm(n_samples)[:n_samples // 2]] = 1

    # true attribute
    s = torch.zeros(n_samples, dtype=torch.int8)
    s[torch.randperm(n_samples)[:n_samples // 2]] = 1

    # predicted label
    y_pred = torch.zeros_like(y_true)
    y_pred[torch.randperm(n_samples)[:6]] = 1

    # test data
    print('y_true = ', y_true)
    print('y_pred = ', y_pred)
    print('     s = ', s)

    (tp, fp, fn, tn) = confusion_matrix(y_true, y_pred, pos_label=1)
    out_ind = independence(y_true, y_pred, s)
    out_sep = separation(y_true, y_pred, s)
    out_suf = sufficiency(y_true, y_pred, s)







