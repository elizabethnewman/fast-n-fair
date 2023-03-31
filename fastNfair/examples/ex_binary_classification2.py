import torch
import hessQuik.networks as net
import hessQuik.activations as act
from fastNfair.data import generate_binary_data
from fastNfair.objective_function import ObjectiveFunctionLogisticRegression
from fastNfair.optimizers import GradientDescent
from fastNfair.training import basic_training, basic_training2

torch.manual_seed(42)

x_train, y_train, s_train, (blue_label, red_label), (attr_A, attr_B) = generate_binary_data()
y_train = torch.from_numpy(y_train)
y_train[y_train == -1] = 0
y_train = y_train.to(torch.long)

x_val, y_val, s_val, *_ = generate_binary_data()
y_val = torch.from_numpy(y_val)
y_val[y_val == -1] = 0
y_val = y_val.to(torch.long)

# # training without knowing attributes
my_net = net.fullyConnectedNN((x_train.shape[1], 1), act=act.identityActivation())
fctn = ObjectiveFunctionLogisticRegression(my_net)


# training
from time import time
import numpy as np

# set seed for reproducibility
torch.manual_seed(42)

# choose maximum number of epochs (outer loops) and batch size
max_epochs = 100

# setup optimizer and loss function
optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

# ---------------------------------------------------------------------------- #
# creating headers and printing formats
do_train = True
if do_train:
    headers = ('', '', '|', 'running', '', '|', 'train', '', '|', 'test', '')

    tmp = ('|', 'loss', 'accuracy')
    printouts = ('epoch', 'time') + 3 * tmp

    tmp = '{:<2s}{:<15.4e}{:<15.4f}'
    printouts_frmt = '{:<15d}{:<15.4f}' + 3 * tmp

    tmp = '{:<2s}{:<15s}{:<15s}'
    print(('{:<15s}{:<15s}' + 3 * tmp).format(*headers))
    print(('{:<15s}{:<15s}' + 3 * tmp).format(*printouts))
# ---------------------------------------------------------------------------- #

# initial evaluation
# loss_train = basic_training.test(my_net, criterion, x, y)
# loss_val = basic_training.test(my_net, criterion, x_val, y_val)


    loss_train = basic_training2.test(fctn, x_train, y_train)
    loss_val = basic_training2.test(fctn, x_val, y_val)

    # initial printouts
    info_iter = (-1, 0.0) + ('|',) + (2 * (0,)) + ('|',) + loss_train + ('|',) + loss_val
    print(printouts_frmt.format(*info_iter))

    # store history for plotting
    history = np.array([x for x in info_iter if not (x == '|')]).reshape(1, -1)

# ---------------------------------------------------------------------------- #
# main loop

    for epoch in range(max_epochs):

        # update weights using mini-batch approach
        t0 = time()
        # running_loss, running_acc = basic_training.train_one_epoch(my_net, optimizer, criterion, x, y)
        running_loss, running_acc = basic_training2.train_one_epoch(fctn, optimizer, x_train, y_train)
        t1 = time()

        # test network performance on all data at once
        # loss_train = basic_training.test(my_net, criterion, x, y)
        # loss_val = basic_training.test(my_net, criterion, x_val, y_val)
        loss_train = basic_training2.test(fctn, x_train, y_train)
        loss_val = basic_training2.test(fctn, x_val, y_val)

        # information from this iteration
        info_iter = (epoch, t1 - t0) + ('|',) + (running_loss, running_acc) + ('|',) + loss_train + ('|',) + loss_val
        print(printouts_frmt.format(*info_iter))

        # story information in history
        history = np.concatenate((history, np.array([x for x in info_iter if not (x == '|')]).reshape(1, -1)), axis=0)


#%%

# plot results
from fastNfair.utils import statistics as stats
import matplotlib.pyplot as plt
from pprint import pprint


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
    if torch.abs(w[1]) < tol:
        plt.plot(torch.zeros(20) - b / w[0], torch.linspace(y_low, y_high, 20), 'k', linewidth=5)
    else:
        plt.plot(torch.linspace(x_low, x_high, 20), -(w[0] / w[1]) * torch.linspace(x_low, x_high, 20) - b / w[1], 'k',
                 linewidth=5)

    plt.xlim([x_low, x_high])
    plt.ylim([y_low, y_high])


# xx, yy = torch.meshgrid(torch.linspace(-3, 3, 50), torch.linspace(-3, 3, 50), indexing='ij')
# xy = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
# zz = my_net(xy)[0]
# zz = zz.detach()
# pp = torch.sigmoid(zz)

# # learned
w = my_net[0].K.detach()
b = my_net[0].b.detach()

# high accuracy, unfair for FNR
# w = torch.tensor([-1.0, 1.0])
# b = torch.tensor([-1.0])

# fair for FNR, not for FPR
# w = torch.tensor([0.0, 1.0])
# b = torch.tensor([1.0])

# # perfect for A, not for B
# w = torch.tensor([-1.0, 0.0])
# b = torch.tensor([0.0])
#
# # great for B, not for A
# w = torch.tensor([1.0, 2.0])
# b = torch.tensor([6.0])
#
# # lower accuracy, but more fair
# w = torch.tensor([-2.0, 1.0])
# b = torch.tensor([6.0])


# p = my_net(x_train)[0]
# pred_train = 1 * (p > 0.5)
# acc = (sum(1 * (y_train.squeeze() > 0.5) == pred_train.squeeze()) / y_train.numel())
out_train = x_train @ w + b
pred_train = 1 * (out_train > 0.0)

out = stats.confusion_matrix(y_train, pred_train, pos_label=1)
stat_info = stats.compute_statistics(y_train, pred_train, pos_label=1)

stat_info_A = stats.compute_statistics(y_train[s_train == attr_A], pred_train[s_train == attr_A], pos_label=1)
stat_info_B = stats.compute_statistics(y_train[s_train == attr_B], pred_train[s_train == attr_B], pos_label=1)

print('{:>10}\t\t{:>5}\t\t{:>5}\t\t{:>5}'.format(' ','Total', 'A', 'B'))
for k in ['accuracy', 'TPR', 'FPR', 'FNR', 'TNR']:
    print('{:>10s}\t\t{:>0.5f}\t\t{:>0.5f}\t\t{:>0.5f}'.format(k, stat_info[k], stat_info_A[k], stat_info_B[k]))


plt.figure()
blue_label, red_label = 1, 0
attr_A, attr_B = 'A', 'B'

y_train_np = y_train.numpy()


# plt.contour(xx, yy, zz.reshape(xx.shape), levels=0)
idx_blue_A = (y_train_np == blue_label) * (s_train == attr_A)
plt.plot(x_train[idx_blue_A, 0], x_train[idx_blue_A, 1], 'co', markersize=12, zorder=0)
plt.scatter(x_train[idx_blue_A, 0], x_train[idx_blue_A, 1], c='white', marker='${' + attr_A + '}$', zorder=1)

idx_red_A = (y_train_np == red_label) * (s_train == attr_A)
plt.plot(x_train[idx_red_A, 0], x_train[idx_red_A, 1], 'mo', markersize=12, zorder=0)
plt.scatter(x_train[idx_red_A, 0], x_train[idx_red_A, 1], c='white', marker='${' + attr_A + '}$')

idx_blue_B = (y_train_np == blue_label) * (s_train == attr_B)
plt.plot(x_train[idx_blue_B, 0], x_train[idx_blue_B, 1], 'cs', markersize=12, zorder=0)
plt.scatter(x_train[idx_blue_B, 0], x_train[idx_blue_B, 1], c='black', marker='${' + attr_B + '}$')

idx_red_B = (y_train_np == red_label) * (s_train == attr_B)
plt.plot(x_train[idx_red_B, 0], x_train[idx_red_B, 1], 'ms', markersize=12, zorder=0)
plt.scatter(x_train[idx_red_B, 0], x_train[idx_red_B, 1], c='black', marker='${' + attr_B + '}$')

plot_prediction(x_train, w, b)
# plt.contourf(xx, yy, 1.0 * (pp.reshape(xx.shape) > 0), levels=2)
plt.show()


# plt.figure()
#
# # plt.contour(xx, yy, zz.reshape(xx.shape), levels=0)
# idx_blue = (y_train_np == blue_label)
# plt.plot(x_train[idx_blue, 0], x_train[idx_blue, 1], 'cx', markersize=12, zorder=0)
#
# idx_red = (y_train_np == red_label)
# plt.plot(x_train[idx_red, 0], x_train[idx_red, 1], 'mx', markersize=12, zorder=0)
#
# plot_prediction(x_train, w, b)
# # plt.contourf(xx, yy, 1.0 * (pp.reshape(xx.shape) > 0), levels=2)
# plt.show()
