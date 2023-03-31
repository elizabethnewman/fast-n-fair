import torch


def confusion_matrix(true_labels, pred_labels, pos_label=1):
    # returns
    #   (tp, fp, fn, tn)

    # tp : true positives (# correctly labeled as blue)
    tp = sum(pred_labels[true_labels == pos_label] == pos_label)

    # tn : true negatives (# correctly labeled as red)
    tn = sum(pred_labels[true_labels != pos_label] != pos_label)

    # fp : false positives (# incorrectly labeled as blue)
    fp = sum(pred_labels[true_labels != pos_label] == pos_label)

    # fn : false negative (# incorrectly labeled as red)
    fn = sum(pred_labels[true_labels == pos_label] != pos_label)

    return tp, fp, fn, tn


def compute_statistics(true_labels, pred_labels, pos_label=1):

    # confusion matrix
    tp, fp, fn, tn = confusion_matrix(true_labels, pred_labels, pos_label=pos_label)

    # total number of positives
    p = tp + fn

    # total number of negatives
    n = fp + tn

    # accuracy
    acc = (tp + tn) / (p + n)

    # true positive rate (sensitivity, recall, hit rate, 1 - fnr)
    tpr = torch.nan if p == 0 else tp / p

    # true negative rate (specificity, selectivity, 1 - fpr)
    tnr = torch.nan if n == 0 else tn / n

    # false positive rate (fall-out, 1 - tnr)
    fpr = torch.nan if n == 0 else fp / n

    # false negative rate (miss rate, 1 - tpr)
    fnr = fn / p

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