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


import argparse
# argument parser
parser = argparse.ArgumentParser(prog='FastNFairColorMNIST',
                                 description='MNIST example with unfair coloring',
                                 epilog='fast-n-fair')

# reproducibility
parser.add_argument('--seed', default=42)

# data
parser.add_argument('--p_train', default=0.2, help='probability of color flipping')
parser.add_argument('--p_val', default=0.1, help='probability of color flipping')
parser.add_argument('--p_test', default=0.9, help='probability of color flipping')

parser.add_argument('--alpha', default=1e-1, help='regularization parameter')
parser.add_argument('--n-train', default=200, help='number of training points')
parser.add_argument('--n-val', default=50, help='number of training points')
parser.add_argument('--n-test', default=50, help='number of training points')

# training
parser.add_argument('--epochs', default=10)
parser.add_argument('--batch', default=5)
parser.add_argument('-lr', '--lr', default=1e-3)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-r', '--robust', action='store_true')
parser.add_argument('--radius', default=1e1)


# general
parser.add_argument('-p', '--plot', action='store_true')

# save
parser.add_argument('-s', '--save', action='store_true')
# parse
args = parser.parse_args()

# args.epochs = 10
# args.alpha = 0e0
# args.verbose = True
# args.robust = False
# args.plot = True

#%% generate data

# for reproducibility
torch.manual_seed(args.seed)

# https://github.com/facebookresearch/InvariantRiskMinimization/tree/main

# number of data points
n_train, n_val, n_test = args.n_train, args.n_val, args.n_test

(x, y), (x_t, y_t) = generate_mnist(n_train=n_train + n_val, n_test=n_test)

x_train, digit_train = x[:n_train], y[:n_train]
x_val, digit_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
x_test, digit_test = x_t, y_t

x_train, y_train, s_train = generate_color_mnist_binary(x_train, digit_train, args.p_train)
x_val, y_val, s_val = generate_color_mnist_binary(x_val, digit_val, args.p_val)
x_test, y_test, s_test = generate_color_mnist_binary(x_test, digit_test, args.p_test)


# compute correlation between labels and digit
corr_digit_train = (1.0 * (y_train == digit_train)).sum() / y_train.numel()
corr_attr_train = (1.0 * (y_train == s_train)).sum() / y_train.numel()

corr_digit_val = (1.0 * (y_val == digit_val)).sum() / y_val.numel()
corr_attr_val = (1.0 * (y_val == s_val)).sum() / y_val.numel()

corr_digit_test = (1.0 * (y_test == digit_test)).sum() / y_test.numel()
corr_attr_test = (1.0 * (y_test == s_test)).sum() / y_test.numel()

if args.verbose:
    print('Correlation of labels and attributes with true digit')
    print('TRAIN: digit: %0.4f\tattr: %0.4f' % (corr_digit_train, corr_attr_train))
    print('VAL:   digit: %0.4f\tattr: %0.4f' % (corr_digit_val, corr_attr_val))
    print('TEST:  digit: %0.4f\tattr: %0.4f' % (corr_digit_test, corr_attr_test))

if args.plot:
    n = 64
    visualize_color_mnist((x_train[:n], y_train[:n], s_train[:n]), n_rows=4)
    plt.show()

#%% training

# for reproducibility
torch.manual_seed(args.seed)


# create linear network
my_net = net.NN(net.fullyConnectedNN([x_train.shape[1] * x_train.shape[2] * x_train.shape[3], 20, 10],
                                     act=act.tanhActivation()),
                lay.singleLayer(10, 1, act=act.identityActivation(), bias=True)
                )

# create objective function
fctn = ObjectiveFunctionLogisticRegression(my_net)

# choose optimizer
opt = torch.optim.Adam(fctn.parameters(), lr=args.lr)

# construct trainer
trainer = TrainerSGD(opt, max_epochs=args.epochs, batch_size=args.batch,
                     regularier=RegularizerInvariantRisk(alpha=args.alpha))

# train!
t0 = time.perf_counter()
results_train = trainer.train(fctn, (x_train.view(x_train.shape[0], -1), y_train, s_train), (x_val.view(x_val.shape[0], -1), y_val, s_val), (x_test.view(x_test.shape[0], -1), y_test, s_test),
                              verbose=args.verbose, robust=args.robust, radius=args.radius)
t1 = time.perf_counter()
results_train['total_time'] = t1 - t0

#%% compute metrics
evaluator = Evaluator()
results_eval = evaluator.evaluate(fctn, (x_train.view(x_train.shape[0], -1), y_train, s_train), (x_val.view(x_val.shape[0], -1), y_val, s_val), (x_test.view(x_test.shape[0], -1), y_test, s_test))

#%% plots

if args.plot:
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

if args.save:
    import pickle
    import os
    dir_name = 'mnist_binary_results/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # make file
    filename = ''

    # record robust
    if args.robust:
        filename += 'robust'
    else:
        filename += 'nonrobust'

    # record alpha
    filename += '--alpha' + '-' + str(round(args.alpha, 2))

    # record correlations (p values)
    filename += '--p_(train-val-test)' + '-(' + str(round(args.p_train,2)) + '-' + str(round(args.p_val,2)) + '-' + str(round(args.p_test,2)) + ')'

    print('Saving as...')
    print(filename)

    with open(dir_name + filename + '.pkl', 'wb') as f:
        results = {'results_train': results_train, 'results_eval': results_eval, 'args': args}
        pickle.dump(results, f)






