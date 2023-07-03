import matplotlib.pyplot as plt
import torch.optim
from fastNfair.data import generate_unfair_data, visualize_unfair_data
from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
from fastNfair.training import TrainerSGD, Evaluator
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net

import argparse
# argument parser
parser = argparse.ArgumentParser(prog='FastNFairToyExample',
                                 description='Toy example with unfair data',
                                 epilog='fast-n-fair')

# reproducibility
parser.add_argument('--seed', default=42)

# data
parser.add_argument('--p1', default=0.5, help='percent of s = 0 in class y = 0')
parser.add_argument('--p2', default=0.5, help='percent of s = 0 in class y = 1')
parser.add_argument('--alpha', default=1e-1, help='unfair scale')
parser.add_argument('--u', default=(1.0, 1.0), help='unfair direction')
parser.add_argument('--n-train', default=200, help='number of training points')
parser.add_argument('--n-val', default=50, help='number of training points')
parser.add_argument('--n-test', default=50, help='number of training points')

# training
parser.add_argument('--epochs', default=10)
parser.add_argument('-lr', '--lr', default=1e-2)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-r', '--robust', action='store_true')
parser.add_argument('--radius', default=2e-1)

# general
parser.add_argument('-p', '--plot', action='store_true')

# parse
args = parser.parse_args()


args.epochs = 20
args.verbose = True
args.robust = False
args.radius = 1e-1
args.plot = True



#%% generate data

# for reproducibility
torch.manual_seed(args.seed)

# number of data points
n_train, n_val, n_test = args.n_train, args.n_val, args.n_test

x_train, y_train, s_train = generate_unfair_data(n_train + n_val, p1=args.p1, p2=args.p2, alpha=args.alpha, u=args.u)

x_val, y_val, s_val = x_train[n_train:], y_train[n_train:], s_train[n_train:]
x_train, y_train, s_train = x_train[:n_train], y_train[:n_train], s_train[:n_train]

# test data
x_test, y_test, s_test = generate_unfair_data(n_test, p1=args.p1, p2=args.p2, alpha=args.alpha, u=args.u)

if args.plot:
    visualize_unfair_data((x_train, y_train, s_train), domain=(-0.1, 1.1, -0.1, 1.1))
    plt.show()

#%% training

# for reproducibility
torch.manual_seed(args.seed)


# create linear network
my_net = net.NN(lay.singleLayer(2, 1, act=act.identityActivation(), bias=True))

# create objective function
fctn = ObjectiveFunctionLogisticRegression(my_net)

# choose optimizer
opt = torch.optim.Adam(fctn.parameters(), lr=args.lr)

# construct trainer
trainer = TrainerSGD(opt, max_epochs=args.epochs)

# train!
results_train = trainer.train(fctn, (x_train, y_train, s_train), (x_val, y_val, s_val), (x_test, y_test, s_test),
                              verbose=args.verbose, robust=args.robust, radius=args.radius)


#%% compute metrics
evaluator = Evaluator()
results_eval = evaluator.evaluate(fctn, (x_train, y_train, s_train), (x_val, y_val, s_val), (x_test, y_test, s_test))

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

    visualize_unfair_data((x_train, y_train, s_train), fctn.net, show_orig=True, domain=(-2, 2, -2, 2))
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


#%%

from pprint import pprint

pprint(results_eval['train']['fairness'])




