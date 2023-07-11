import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.optim
from fastNfair.data import gray_to_color, generate_color_mnist_binary, generate_mnist, visualize_color_mnist
from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
from fastNfair.regularizers import RegularizerInvariantRisk
from fastNfair.training import TrainerSGD, Evaluator
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
import time

evaluator = Evaluator()


# fro reproduciblity
torch.manual_seed(42)

# choose labels
labels = (0, 1)

# for reproducibility
torch.manual_seed(42)

# number of data points
# p_train, p_val, p_test = 0.2, 0.1, 0.9
p_train, p_val, p_test = 0.5, 0.5, 0.5
p_label = 0.25
n_train, n_val, n_test = 200, 50, 50

# generate data
(x, y), (x_t, y_t) = generate_mnist(n_train=n_train + n_val, n_test=n_test, labels=(0, 1))


# split data
x_train, digit_train = x[:n_train], y[:n_train]
x_val, digit_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, digit_test = x_t, y_t

# color data
x_train, y_train, s_train = generate_color_mnist_binary(x_train, digit_train, p_train, p_label=p_label)
x_val, y_val, s_val = generate_color_mnist_binary(x_val, digit_val, p_val, p_label=p_label)
x_test, y_test, s_test = generate_color_mnist_binary(x_test, digit_test, p_test, p_label=p_label)

# compute correlation between labels and digit
corr_digit_train = (1.0 * (y_train == digit_train)).sum() / y_train.numel()
corr_attr_train = (1.0 * (y_train == s_train)).sum() / y_train.numel()

corr_digit_val = (1.0 * (y_val == digit_val)).sum() / y_val.numel()
corr_attr_val = (1.0 * (y_val == s_val)).sum() / y_val.numel()

corr_digit_test = (1.0 * (y_test == digit_test)).sum() / y_test.numel()
corr_attr_test = (1.0 * (y_test == s_test)).sum() / y_test.numel()

print('Correlation of labels and attributes with true digit')
print('TRAIN: digit: %0.4f\tattr: %0.4f' % (corr_digit_train, corr_attr_train))
print('VAL:   digit: %0.4f\tattr: %0.4f' % (corr_digit_val, corr_attr_val))
print('TEST:  digit: %0.4f\tattr: %0.4f' % (corr_digit_test, corr_attr_test))

#%%
n = 64
visualize_color_mnist((x_train[:n], y_train[:n], s_train[:n]), n_rows=4)
plt.show()

#%%

# for reproducibility
torch.manual_seed(42)

# create linear network
# my_net = net.NN(net.fullyConnectedNN([x_train.shape[1] * x_train.shape[2] * x_train.shape[3], 5],
#                                      act=act.tanhActivation()),
#                 lay.singleLayer(5, 1, act=act.identityActivation(), bias=True)
#                 )

my_net = net.NN(lay.singleLayer(5, 1, act=act.identityActivation(), bias=True))

# create objective function
fctn = ObjectiveFunctionLogisticRegression(my_net)

# choose optimizer
opt = torch.optim.Adam(fctn.parameters(), lr=1e-3)

# construct trainer
trainer = TrainerSGD(opt, max_epochs=10, batch_size=5,
                     regularier=RegularizerInvariantRisk(alpha=0.0))

# train!
t0 = time.perf_counter()
results_train = trainer.train(fctn, (x_train.view(x_train.shape[0], -1), y_train, s_train),
                              (x_val.view(x_val.shape[0], -1), y_val, s_val),
                              (x_test.view(x_test.shape[0], -1), y_test, s_test),
                              verbose=True, robust=False, radius=1e1)
t1 = time.perf_counter()
results_train['total_time'] = t1 - t0


#%%
results_eval = evaluator.evaluate(fctn, (x_train.view(x_train.shape[0], -1), y_train, s_train),
                                  (x_val.view(x_val.shape[0], -1), y_val, s_val),
                                  (x_test.view(x_test.shape[0], -1), y_test, s_test))

#%%
from sklearn import metrics
import numpy as np
from operator import itemgetter

cm = itemgetter(*('TN', 'FN', 'FP', 'TP'))(results_eval['train']['full']['stats'])
metrics.ConfusionMatrixDisplay(np.array(cm).reshape(2, -1)).plot()
plt.show()

for j in ('full', 's = 0', 's = 1'):
    fpr, tpr, auc = itemgetter(*('fpr', 'tpr', 'auc'))(results_eval['train'][j])
    plt.plot(fpr, tpr, label=j + ': AUC = %0.4f' % auc)

plt.plot(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), '--', label='rand')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

#%%
# comparison of fairness metrics
from pprint import pprint
pprint(results_eval['train']['fairness'])
