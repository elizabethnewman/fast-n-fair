import matplotlib.pyplot as plt
import torch.optim
from fastNfair.data import generate_adult
from fastNfair.data import generate_lsat
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
parser.add_argument('--n-train', default=1500, help='number of training points')
parser.add_argument('--n-val', default=500, help='number of training points')
parser.add_argument('--n-test', default=1000, help='number of training points')

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


args.epochs = 10
args.verbose = True
args.robust = False
args.radius = 1e1
args.plot = True

print(args)



#%% generate data

# for reproducibility
torch.manual_seed(args.seed)

# number of data points
n_train, n_val, n_test = args.n_train, args.n_val, args.n_test

# (x_train, y_train, s_train), (x_test, y_test, s_test) = generate_adult(load_dir='../data/raw/adult/', attr='sex')
(x, y, s), (x_test, y_test, s_test) = generate_lsat(p_train=0.8)

# split, but make sure val has some of all classes
import math
x_train, x_val = torch.empty(0, x.shape[1]), torch.empty(0, x.shape[1])
y_train, y_val = torch.empty(0, dtype=y.dtype), torch.empty(0, dtype=y.dtype)
s_train, s_val = torch.empty(0, dtype=s.dtype), torch.empty(0, dtype=s.dtype)

p_train = 0.8
for yi in [0, 1]:
    for sj in [0, 1]:
        idx_ij = ((y == yi) * (s == sj))
        x_ij = x[idx_ij]
        y_ij = y[idx_ij]
        s_ij = s[idx_ij]
        nn = y_ij.numel()
        n_train = min(math.floor(p_train * nn), 10)

        idx = torch.randperm(nn)
        x_train = torch.cat((x_train, x_ij[idx[:n_train]]), dim=0)
        y_train = torch.cat((y_train, y_ij[idx[:n_train]]))
        s_train = torch.cat((s_train, s_ij[idx[:n_train]]))

        x_val = torch.cat((x_val, x_ij[idx[:n_train]]), dim=0)
        y_val = torch.cat((y_val, y_ij[idx[:n_train]]))
        s_val = torch.cat((s_val, s_ij[idx[:n_train]]))

# x_val, y_val, s_val = x_train[n_train:], y_train[n_train:], s_train[n_train:]
# x_train, y_train, s_train = x_train[:n_train], y_train[:n_train], s_train[:n_train]
# x_test, y_test, s_test = x_test[:n_test], y_test[:n_test], s_test[:n_test]

if args.plot:
    print('make plot')

#%% training

# for reproducibility
torch.manual_seed(args.seed)


# create linear network
width = 10
depth = 4
# my_net = net.NN(lay.singleLayer(x_train.shape[1], width, act=act.tanhActivation()),
#                 net.resnetNN(width, depth, h=0.1, act=act.tanhActivation()),
#                 lay.singleLayer(width, 1, act=act.identityActivation(), bias=True)
#                 )
my_net = net.NN(lay.singleLayer(x_train.shape[1], 1, act=act.identityActivation(), bias=True))

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

#%%
from pprint import pprint
pprint(results_eval['train']['fairness'])


#%%
from fastNfair.data.lsat import visualize_lsat_data

visualize_lsat_data((x_train, y_train, s_train), my_net)
plt.show()