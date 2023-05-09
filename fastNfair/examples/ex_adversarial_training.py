import matplotlib.pyplot as plt
import torch.optim

from fastNfair.data import generate_unfair_data
from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
from fastNfair.regularizers import RegularizerTikhonov, RegularizerInvariantRisk, RegularizerSeparation
from fastNfair.training.adversarial_training import train_one_epoch, test
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net

# for reproducibility
torch.manual_seed(42)

# generate data
p1, p2, push_unfair = 0.5, 0.5, 0.1
x_train, y_train, s_train = generate_unfair_data(200, p1=p1, p2=p2, push_unfair=push_unfair)

# create linear network
my_net = net.NN(lay.singleLayer(2, 1, act=act.identityActivation(), bias=True))

# create objective function
fctn = ObjectiveFunctionLogisticRegression(my_net)

# choose optimizer
opt = torch.optim.Adam(fctn.parameters(), lr=1e-2)

loss, acc = test(fctn, x_train, y_train)
print('%0.2d\t%0.4e\t%0.4f' % (-1, loss, acc))

# train!
max_epochs = 50
for i in range(max_epochs):

    out = train_one_epoch(fctn, opt, x_train, y_train, s_train, batch_size=20, robust=True, radius=5e-1,
                          regularizer=RegularizerSeparation(alpha=0e0))

    loss, acc = test(fctn, x_train, y_train)
    print('%0.2d\t%0.4e\t%0.4f' % (i, loss, acc))

#%% compute metrics

from fastNfair.utils import statistics as stats
from pprint import pprint

threshold = 0

y_pred = 1 * (fctn.net(x_train)[0] > threshold)
results = stats.compute_statistics(y_train, y_pred)

# results per group (independence)
y_pred_A = 1 * (fctn.net(x_train[s_train == 0])[0] > threshold)
results_A = stats.compute_statistics(y_train[s_train == 0], y_pred[s_train == 0])

y_pred_B = 1 * (fctn.net(x_train[s_train == 1])[0] > threshold)
results_B = stats.compute_statistics(y_train[s_train == 1], y_pred[s_train == 1])

#%%
from fastNfair.utils.statistics import fairness_metrics

threshold = 0

y_pred = 1 * (fctn.net(x_train)[0] > threshold).view(-1)
out = fairness_metrics(y_train, y_pred, s_train)

#%%
from sklearn import metrics

cm = metrics.confusion_matrix(y_train, y_pred)
metrics.ConfusionMatrixDisplay(cm).plot()
plt.show()

fpr, tpr, _ = metrics.roc_curve(y_train, fctn.net(x_train)[0].detach(), pos_label=1)
# metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
plt.plot(fpr, tpr, label='all')

auc = metrics.auc(fpr, tpr)
print('AUC, full:', auc)

fpr, tpr, _ = metrics.roc_curve(y_train[s_train == 0], fctn.net(x_train[s_train == 0])[0].detach(), pos_label=1)
# metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
plt.plot(fpr, tpr, label='s=0')

auc = metrics.auc(fpr, tpr)
print('AUC, s = 0:', auc)


fpr, tpr, _ = metrics.roc_curve(y_train[s_train == 1], fctn.net(x_train[s_train == 1])[0].detach(), pos_label=1)
# metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
plt.plot(fpr, tpr, label='s=1')

auc = metrics.auc(fpr, tpr)
print('AUC, s = 1:', auc)

plt.plot(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), '--', label='rand')

plt.legend()
plt.show()



#%%
import matplotlib as mpl
import matplotlib.pyplot as plt

x_train, y_train, s_train = generate_unfair_data(200, p1=p1, p2=p2, push_unfair=push_unfair)


# plot everything
x_grid, y_grid = torch.meshgrid(torch.linspace(0, 1, 300),
                                torch.linspace(0, 1, 300), indexing='ij')
xy_grid = torch.cat((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), dim=1)
c_grid = 1 * (fctn.net(xy_grid)[0] > 0)

# tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap = mpl.colors.ListedColormap(['r', 'b'])
plt.figure()
plt.contourf(x_grid, y_grid, c_grid.reshape(x_grid.shape), alpha=0.25, cmap=cmap)

t = torch.linspace(0, 1, 100)
tt = -t.reshape(-1, 1) + 1
with torch.no_grad():
    tt2 = -(my_net[0].K[0] / my_net[0].K[1]) * t - (my_net[0].b / my_net[0].K[1])
plt.plot(t, tt, 'k', linewidth=2)
plt.plot(t, tt2, 'r', linewidth=2)

cmap = mpl.colors.ListedColormap(['tab:red', 'tab:blue'])

idx1 = (s_train == 0)
plt.scatter(x_train[idx1, 0], x_train[idx1, 1], None, y_train[idx1], marker='$A$', cmap=cmap)

idx2 = (s_train == 1)
plt.scatter(x_train[idx2, 0], x_train[idx2, 1], None, y_train[idx2], marker='$B$', cmap=cmap)
plt.show()


plt.show()






