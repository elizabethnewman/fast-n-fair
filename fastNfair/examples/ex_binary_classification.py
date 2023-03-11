import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#%%
# create data
torch.manual_seed(123)

# separating line
w = torch.tensor([-0.5, 1.0])
b = torch.tensor(0.0)

# data
mu = torch.tensor([0.0, 0.0]).reshape(1, -1)
sigma = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
n = 1000
x = mu + torch.randn(n, 2) @ sigma
labels = torch.sign(x @ w + b)

# plot
# plt.figure()
# a_low = x.min() - 0.2
# a_high = x.max() + 0.2
# X, Y = torch.meshgrid(torch.linspace(a_low, a_high, 20), torch.linspace(a_low, a_high, 20), indexing='ij')
# XY = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
# Z = XY @ w + b
#
# plt.contourf(X, Y, Z.reshape(X.shape), 0, colors=['r', 'b'], alpha=0.5)
#
# tol = 1e-10
# if w[1] < tol:
#     plt.plot(torch.zeros(20) + b / w[0], torch.linspace(a_low, a_high, 20), 'k', linewidth=5)
# else:
#     w = torch.sign(w[0]) * w
#     plt.plot(torch.linspace(a_low, a_high, 20), -(w[0] / w[1]) * torch.linspace(a_low, a_high, 20) + b / w[1], 'k', linewidth=5)
#
# plt.scatter(x[:, 0], x[:, 1], c=labels, cmap=mpl.colors.ListedColormap(['r', 'b']))
# plt.xlim(a_low, a_high)
# plt.ylim(a_low, a_high)
# plt.show()

#%% create fairness test problem

torch.manual_seed(42)


def generate_gaussian_normal_data(mu, sigma, label, attribute, n=100):
    x = 1.0 * torch.tensor(mu).reshape(-1, 2) + torch.randn(n, 2) @ (1.0 * torch.tensor(sigma))
    return x, np.array(n * [label]), np.array(n * [attribute])


def plot_prediction(x, w, b, tol=1e-10):
    x_low = x[:, 0].min() - 0.5
    x_high = x[:, 0].max() + 0.5
    y_low = x[:, 1].min() - 0.5
    y_high = x[:, 1].max() + 0.5

    X, Y = torch.meshgrid(torch.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 20),
                          torch.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 20), indexing='ij')
    XY = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    Z = XY @ w + b

    plt.contourf(X, Y, Z.reshape(X.shape), 0, colors=['r', 'b'], alpha=0.5, zorder=-10)

    # force first entry to be nonnegative
    if w[1] < tol:
        plt.plot(torch.zeros(20) - b / w[0], torch.linspace(y_low, y_high, 20), 'k', linewidth=5)
    else:
        plt.plot(torch.linspace(x_low, x_high, 20), -(w[0] / w[1]) * torch.linspace(x_low, x_high, 20) - b / w[1], 'k',
                 linewidth=5)

    plt.xlim([x_low, x_high])
    plt.ylim([y_low, y_high])


def get_predicted_labels(x, w, b):
    return torch.sign(x @ w + b)


blue_label, red_label = 1, -1
attr_A, attr_B = 'A', 'B'

# generate cluster for each class
mu_red_A, sigma_red_A = [3, -3], [[2, 0], [0, 1]]       # class 0, attr A
mu_red_B, sigma_red_B = [-3, -3], [[1, 0], [0, 2]]      # class 0, attr B
mu_blue_A, sigma_blue_A = [3, 3], [[1, 0], [0, 2]]      # class 1, attr A
mu_blue_B, sigma_blue_B = [-3, 3], [[2, 0], [0, 1]]     # class 1, attr B

# # let's assume A, B are socioeconomic regions (A = wealthy suburb, B = poor city neighborhood)
# # let's say blue is accepted to college (positive) and red is rejected from college (negative)
# mu_red_A, sigma_red_A = [80, 1200], [[10, 0], [0, 100]]        # class 0, attr A
# mu_red_B, sigma_red_B = [60, 800], [[20, 0], [0, 200]]       # class 0, attr B
# mu_blue_A, sigma_blue_A = [90, 1400], [[10, 0], [0, 100]]      # class 1, attr A
# mu_blue_B, sigma_blue_B = [80, 1400], [[10, 0], [0, 100]]      # class 1, attr B


# red class, sensitive attribute A
x_red_A, label_red_A, s_red_A = generate_gaussian_normal_data(mu_red_A, sigma_red_A, red_label, 'A')

# blue class, sensitive attribute A
x_blue_A, label_blue_A, s_blue_A = generate_gaussian_normal_data(mu_blue_A, sigma_blue_A, blue_label, 'A')

# red class, sensitive attribute B
x_red_B, label_red_B, s_red_B = generate_gaussian_normal_data(mu_red_B, sigma_red_B, red_label, 'B')

# blue class, sensitive attribute B
x_blue_B, label_blue_B, s_blue_B = generate_gaussian_normal_data(mu_blue_B, sigma_blue_B, blue_label, 'B')

# store as one dataset
x = torch.cat((x_red_A, x_blue_A, x_red_B, x_blue_B), dim=0)
y = np.concatenate((label_red_A, label_blue_A, label_red_B, label_blue_B))
s = np.concatenate((s_red_A, s_blue_A, s_red_B, s_blue_B))

# separating line
w = torch.tensor([1.0, 1.0])
b = torch.tensor(-1.0)

# create plot
plt.figure()
idx_blue_A = (y == blue_label) * (s == attr_A)
plt.plot(x[idx_blue_A, 0], x[idx_blue_A, 1], 'co', markersize=12, zorder=0)
plt.scatter(x[idx_blue_A, 0], x[idx_blue_A, 1], c='white', marker='${' + attr_A + '}$', zorder=1)

idx_red_A = (y == red_label) * (s == attr_A)
plt.plot(x[idx_red_A, 0], x[idx_red_A, 1], 'mo', markersize=12, zorder=0)
plt.scatter(x[idx_red_A, 0], x[idx_red_A, 1], c='white', marker='${' + attr_A + '}$')

idx_blue_B = (y == blue_label) * (s == attr_B)
plt.plot(x[idx_blue_B, 0], x[idx_blue_B, 1], 'cs', markersize=12, zorder=0)
plt.scatter(x[idx_blue_B, 0], x[idx_blue_B, 1], c='black', marker='${' + attr_B + '}$')

idx_red_B = (y == red_label) * (s == attr_B)
plt.plot(x[idx_red_B, 0], x[idx_red_B, 1], 'ms', markersize=12, zorder=0)
plt.scatter(x[idx_red_B, 0], x[idx_red_B, 1], c='black', marker='${' + attr_B + '}$')
plot_prediction(x, w, b)

plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#%% build confusion matrix
import sklearn.metrics

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


# overall classification rates
y_pred = get_predicted_labels(x, w, b)
info = compute_statistics(y, y_pred)
print('info', info)

# conditional classification rates

# independence (does prediction depend on attribute?)
idx_A = (s == attr_A)
info_A = compute_statistics(y_pred[idx_A], y[idx_A])
print('info_A', info_A)

idx_B = (s == attr_B)
info_B = compute_statistics(y_pred[idx_B], y[idx_B])
print('info_B', info_B)

# separation (given that we classified as positive, are we more likely to find A or B positive?)
idx_blue_A = (y == blue_label) * (s == attr_A)
info_blue_A = compute_statistics(y_pred[idx_blue_A], y[idx_blue_A])
print('info_blue_A', info_blue_A)

idx_blue_B = (y == blue_label) * (s == attr_B)
info_blue_A = compute_statistics(y_pred[idx_blue_B], y[idx_blue_B])
print('info_blue_A', info_blue_A)

# sufficiency (given we predicted positive, is this enough to say that the class really was positive?)
idx_pred_blue_A = (y_pred.numpy() == blue_label) * (s == attr_A)
tp_pred_blue_A, fp_pred_blue_A, fn_pred_blue_A, tn_pred_blue_A = confusion_matrix(y_pred[idx_pred_blue_A],
                                                                                  y[idx_pred_blue_A], 1, -1)

idx_pred_blue_B = (y_pred.numpy() == blue_label) * (s == attr_B)
tp_pred_blue_B, fp_pred_blue_B, fn_pred_blue_B, tn_pred_blue_B = confusion_matrix(y_pred[idx_pred_blue_B],
                                                                                  y[idx_pred_blue_B], 1, -1)
print(tp_pred_blue_A / (tp_pred_blue_A + fn_pred_blue_A), tp_pred_blue_B / (tp_pred_blue_B + fn_pred_blue_B))


#%%

# # measure accuracy (how many reds are in the red region? blues in blue region?)
# labels_red_A = get_predicted_labels(x_red_A, w, b)
# labels_red_B = get_predicted_labels(x_red_B, w, b)
#
# labels_blue_A = get_predicted_labels(x_blue_A, w, b)
# labels_blue_B = get_predicted_labels(x_blue_B, w, b)
#
#
# # overall accuracy
# correct_labels = (sum(labels_red_A == red_label)
#                   + sum(labels_red_B == red_label)
#                   + sum(labels_blue_A == blue_label)
#                   + sum(labels_blue_B == blue_label))
# accuracy = correct_labels / (len(labels_red_A) + len(labels_red_B) + len(labels_blue_A) + len(labels_blue_B))
#
# # accuracy per group
# accuracy_red_A = sum(labels_red_A == red_label) / len(labels_red_A)
# accuracy_red_B = sum(labels_red_B == red_label) / len(labels_red_B)
#
# accuracy_blue_A = sum(labels_blue_A == blue_label) / len(labels_blue_A)
# accuracy_blue_B = sum(labels_blue_B == blue_label) / len(labels_blue_B)
#
# # accuracy per attribute
# accuracy_A = (sum(labels_red_A == red_label) + sum(labels_blue_A == blue_label)) / (len(labels_red_A) + len(labels_blue_A))
# accuracy_B = (sum(labels_red_B == red_label) + sum(labels_blue_B == blue_label)) / (len(labels_red_B) + len(labels_blue_B))
#
# # confusion matrix
# # tp : true positives (# correctly labeled as blue)
# tp = sum(labels_blue_A == blue_label) + sum(labels_blue_B == blue_label)
#
# # tn : true negatives (# correctly labeled as red)
# tn = sum(labels_red_A == red_label) + sum(labels_red_B == red_label)
#
# # fp : false positives (# incorrectly labeled as blue)
# fp = sum(labels_red_A == blue_label) + sum(labels_red_B == blue_label)
#
# # fn : false negative (# incorrectly labeled as red)
# fn = sum(labels_blue_A == red_label) + sum(labels_blue_B == red_label)
#
# # fairness metrics
#
# # independence (demographic parity, statistical parity, group fairness, disparate impact)
# # is a predicted blue independent of underlying attribute?
# # i.e., is accuracy_blue_A == accuracy_blue_B?
# # can do something similar for red class
#
# # separation (error rate parity)
# # are true positive rates and false positive rates the same across attributes?
# tpr_A = sum(labels_blue_A == blue_label) / len(labels_blue_A)
# tpr_B = sum(labels_blue_B == blue_label) / len(labels_blue_B)
#
#

