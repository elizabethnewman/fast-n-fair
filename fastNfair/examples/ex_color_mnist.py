import torch
from torchvision import datasets
from fastNfair.data import gray_to_color, generate_color_mnist_binary, generate_mnist, visualize_color_mnist

from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
from fastNfair.training.adversarial_training import train_one_epoch, test
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net



# fro reproduciblity
torch.manual_seed(42)

# choose labels
labels = (0, 1)

# choose number of images (per class)
n_train = 1000
n_val = 100
n_test = 100

(x, y), (x_t, y_t) = generate_mnist(n_train=n_train + n_val, n_test=n_test)

x_train, y_train, s_train = generate_color_mnist_binary(x[:n_train], y[:n_train], 0.1)
x_val, y_val, s_val = generate_color_mnist_binary(x[n_train:], y[n_train:], 0.2)
x_test, y_test, s_test = generate_color_mnist_binary(x_t, y_t, 0.9)


#%%
import matplotlib.pyplot as plt
n = 64
visualize_color_mnist(x_train[:n], y_train[:n], s_train[:n], n_rows=4)
plt.show()

#%%
import torch.nn as nn

def my_view(x):
    return x.view(x.shape[0], -1)

# create linear network
# my_net = net.NN(lay.singleLayer(28 * 28 * 3, 1, act=act.identityActivation(), bias=True))
my_net = net.NN(net.fullyConnectedNN([x_train.shape[1] * x_train.shape[2] * x_train.shape[3], 20, 10],
                                     act=act.tanhActivation()),
                lay.singleLayer(10, 1, act=act.identityActivation(), bias=True)
                )


# create objective function
fctn = ObjectiveFunctionLogisticRegression(my_net)

# choose optimizer
opt = torch.optim.Adam(fctn.parameters(), lr=1e-3)

loss, acc = test(fctn, my_view(x_train), y_train)
print('%0.2d\t%0.4e\t%0.4f' % (-1, loss, acc))

# train!
for i in range(10):

    out = train_one_epoch(fctn, opt, my_view(x_train), y_train, s_train, batch_size=50, robust=True, radius=1e-1, regularizer=None)

    loss, acc = test(fctn, my_view(x_train), y_train)
    print('%0.2d\t%0.4e\t%0.4f' % (i, loss, acc))


# results on test set
loss, acc = test(fctn, my_view(x_test), y_test)
print('%s\t%0.4e\t%0.4f' % ('TEST: ', loss, acc))


#%%
from fastNfair.utils import statistics as stats
from pprint import pprint

y_pred = 1 * (fctn.net(my_view(x_train))[0] > 0.5)
results = stats.compute_statistics(y_train, y_pred)

# results per group

y_pred_A = 1 * (fctn.net(my_view(x_train[s_train == 0]))[0] > 0.5)
results_A = stats.compute_statistics(y_train[s_train == 0], y_pred[s_train == 0])

y_pred_B = 1 * (fctn.net(my_view(x_train[s_train == 1]))[0] > 0.5)
results_B = stats.compute_statistics(y_train[s_train == 1], y_pred[s_train == 1])

