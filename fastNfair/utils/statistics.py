import torch
from sklearn import metrics

# y_true -> true labels of dataset
# y_pred -> predicted labels of dataset generated by classifier
# s -> sensitive attributes of dataset

def confusion_matrix(y_true, y_pred, pos_label=1):
    # returns (tp, fp, fn, tn)
    neg_label = 1 - pos_label
    # indices where positives and negatives occur in true labels
    positive_idx = (y_true == pos_label)
    negative_idx = (y_true == neg_label)
    # tp : true positives (# correctly labeled as true)
    tp = (y_pred[positive_idx] == pos_label).sum()
    # tn : true negatives (# correctly labeled as false)
    tn = (y_pred[negative_idx] == neg_label).sum()
    # fp : false positives (# incorrectly labeled as true)
    fp = (y_pred[negative_idx] == pos_label).sum()
    # fn : false negative (# incorrectly labeled as false)
    fn = (y_pred[positive_idx] == neg_label).sum()
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
    # calculate P(y_pred = (0 or 1) | s = (0 or 1))

    # Numbers of occurrences of s = 0 and s = 1, respectively
    num_s0 = (s == 0).sum()
    num_s1 = (s == 1).sum()

    # num_sm_yn represents the number of occurrences of the joint condition s = m and y_pred = n
    num_s0_y0 = (y_pred[s == 0] == 0).sum()
    num_s0_y1 = (y_pred[s == 0] == 1).sum()
    num_s1_y0 = (y_pred[s == 1] == 0).sum()
    num_s1_y1 = (y_pred[s == 1] == 1).sum()

    # take ratios of numbers of occurrences to get P(y_hat = n|s = m)
    prob_y0_s0 = num_s0_y0 / num_s0
    prob_y1_s0 = num_s0_y1 / num_s0
    prob_y0_s1 = num_s1_y0 / num_s1
    prob_y1_s1 = num_s1_y1 / num_s1

    # populate a dictionary of dictionaries with the above probabilities
    out = {'y = 0': {'s = 0': prob_y0_s0.item(),
                     's = 1': prob_y0_s1.item()},
                     'Difference-': abs(prob_y0_s0.item() - prob_y0_s1.item()),
           'y = 1': {'s = 0': prob_y1_s0.item(),
                     's = 1': prob_y1_s1.item()},
                     'Difference~': abs(prob_y1_s0.item() - prob_y1_s1.item())
           }

    return out


def separation(y_true, y_pred, s):
    # returns out
    out = {'y = 0': {'s = 0': 0.0, 's = 1': 0.0},
           'y = 1': {'s = 0': 0.0, 's = 1': 0.0}
           }
    
    # calculate P = (Y_pred = 1 | Y_true = (0 or 1) , s = (0,1))

    # calculate the probability P(y_pred = 1 | y_true = 0, s = 0)
    # get the location where y_true = 0 and s = 0
    condition_met = (y_true == 0) & (s == 0)
    # probability that y_pred = 1 given y_true = 0 and s = 0
    prob_y1_y0_s0 = (y_pred[condition_met] == 1).sum() / (y_pred[condition_met]).numel()

    # calculate the probability P(y_pred = 1 | y_true = 0, s = 1)
    # get the location where y_true = 0 and s = 0
    condition_met = (y_true == 0) & (s == 1)
    # probability that y_pred = 1 given y_true = 0 and s = 1
    prob_y1_y0_s1 = (y_pred[condition_met] == 1).sum() / (y_pred[condition_met]).numel()

    # calculate the probability P(y_pred = 1 | y_true = 1, s = 0)
    # get the location where y_true = 1 and s = 0
    condition_met = (y_true == 1) & (s == 0)
    # probability that y_pred = 1 given y_true = 1 and s = 0
    prob_y1_y1_s0 = (y_pred[condition_met] == 1).sum() / (y_pred[condition_met]).numel()

    # calculate the probability P(y_pred = 1 | y_true = 1, s = 1)
    # get the location where y_true = 0 and s = 0
    condition_met = (y_true == 1) & (s == 1)
    # probability that y_pred = 1 given y_true = 1 and s = 1
    prob_y1_y1_s1 = (y_pred[condition_met] == 1).sum() / (y_pred[condition_met]).numel()

    out = {'y = 0': {'s = 0': prob_y1_y0_s0.item(),
                     's = 1': prob_y1_y0_s1.item()},
                     'Difference-': abs(prob_y1_y0_s0.item() - prob_y1_y0_s1.item()),
           'y = 1': {'s = 0': prob_y1_y1_s0.item(),
                     's = 1': prob_y1_y1_s1.item()},
                     'Difference~': abs(prob_y1_y1_s0.item() - prob_y1_y1_s1.item())
           }
    
    return out


def sufficiency(y_true, y_pred, s):
    # returns out
    out = {}
    # conditions for (y_pred=0, s=0), (y_pred=0, s=1), (y_pred=1, s=0), and (y_pred=1, s=1) since both are binary
    conditions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # calculate the probability P(y_true = 1 | y_pred = 0, s = 0)
    # get the location where y_pred = 0 and s = 0
    condition_met = (y_pred == 0) & (s == 0)
    # probability that y_true = 1 given y_pred = 0 and s = 0
    prob_y1_y0_s0 = (y_true[condition_met] == 1).sum() / (y_true[condition_met]).numel()

    # calculate the probability P(y_true = 1 | y_pred = 0, s = 1)
    # get the location where y_pred = 0 and s = 1
    condition_met = (y_pred == 0) & (s == 1)
    # probability that y_true = 1 given y_pred = 0 and s = 1
    prob_y1_y0_s1 = (y_true[condition_met] == 1).sum() / (y_true[condition_met]).numel()

    # calculate the probability P(y_true = 1 | y_pred = 1, s = 0)
    # get the location where y_pred = 1 and s = 0
    condition_met = (y_pred == 1) & (s == 0)
    # probability that y_true = 1 given y_pred = 0 and s = 0
    prob_y1_y1_s0 = (y_true[condition_met] == 1).sum() / (y_true[condition_met]).numel()

    # calculate the probability P(y_true = 1 | y_pred = 1, s = 1)
    # get the location where y_pred = 1 and s = 1
    condition_met = (y_pred == 1) & (s == 1)
    # probability that y_true = 1 given y_pred = 1 and s = 1
    prob_y1_y1_s1 = (y_true[condition_met] == 1).sum() / (y_true[condition_met]).numel()

    out = {'y_pred = 0': {'s = 0': prob_y1_y0_s0.item(),
                          's = 1': prob_y1_y0_s1.item()},
                          'Difference-': abs(prob_y1_y0_s0.item() - prob_y1_y0_s1.item()),
           'y_pred = 1': {'s = 0': prob_y1_y1_s0.item(),
                          's = 1': prob_y1_y1_s1.item()},
                          'Difference~': abs(prob_y1_y1_s0.item() - prob_y1_y1_s1.item())
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


def store_statistics(z, y_pred, x, y_true):
    out = compute_statistics(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_true, z.detach(), pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {'stats': out, 'cm': cm, 'fpr': fpr, 'tpr': tpr, 'auc': auc}


def fairness_metrics_test():
    # setup synthetic data
    y_true = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
    s      = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    print('y_true: ', y_true)
    print('y_pred: ', y_pred)
    print('     s: ', s)

    # independence
    ind_true = {'y = 0': {'s = 0': 2 / 5, 's = 1': 3 / 5},
                'y = 1': {'s = 0': 3 / 5, 's = 1': 2 / 5}}
    ind_computed = independence(y_true, y_pred, s)
    ind_pass = verify_fairness_metrics_almost_equal(ind_true, ind_computed)
    print('Independence passed: ', ind_pass)

    # separation
    sep_true = {'y = 0': {'s = 0': 1 / 2, 's = 1': 1 / 3},
                'y = 1': {'s = 0': 2 / 3, 's = 1': 1 / 2}}
    sep_computed = separation(y_true, y_pred, s)
    sep_pass = verify_fairness_metrics_almost_equal(sep_true, sep_computed)
    print('Separation passed: ', sep_pass)

    # sufficiency
    suf_true = {'y_pred = 0': {'s = 0': 1 / 2, 's = 1': 1 / 3},
                'y_pred = 1': {'s = 0': 2 / 3, 's = 1': 1 / 2}}
    suf_computed = sufficiency(y_true, y_pred, s)
    suf_pass = verify_fairness_metrics_almost_equal(suf_true, suf_computed)
    print('Sufficiency passed: ', suf_pass)

    passed = (ind_pass & sep_pass & suf_pass)
    return passed


def verify_fairness_metrics_almost_equal(out_true, out, tol=1e-6):
    flag = True
    for key1 in out_true.keys():
        for key2 in out_true[key1].keys():
            flag = flag & (abs(out_true[key1][key2] - out[key1][key2]) < tol)

    return flag


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

    (tp, fp, fn, tn) = confusion_matrix(y_true, y_pred, pos_label=1)
    out_ind = independence(y_true, y_pred, s)
    out_sep = separation(y_true, y_pred, s)
    out_suf = sufficiency(y_true, y_pred, s)

    fairness_metrics_test()



