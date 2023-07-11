import matplotlib.pyplot as plt
import torch.optim
from fastNfair.data import generate_unfair_data, visualize_unfair_data
from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
from fastNfair.training import TrainerSGD, Evaluator
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
import sys
import argparse
# argument parser
parser = argparse.ArgumentParser(prog='FastNFairToyExample',
                                 description='Toy example with unfair data',
                                 epilog='fast-n-fair')

# reproducibility
parser.add_argument('--seed', default=42)

# data
parser.add_argument('--p1', default=0.5, type=float, help='percent of s = 0 in class y = 0')
parser.add_argument('--p2', default=0.5, type=float, help='percent of s = 0 in class y = 1')
parser.add_argument('--alpha', default=1e-1, type=float, help='unfair scale')
parser.add_argument('--u', default=(1.0, 1.0), type=float, nargs='+', help='unfair direction')
parser.add_argument('--n-train', default=200, type=int, help='number of training points')
parser.add_argument('--n-val', default=50, type=int, help='number of training points')
parser.add_argument('--n-test', default=50, type=int, help='number of training points')

# training
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('-lr', '--lr', default=1e-2, type=float)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-r', '--robust', action='store_true')
parser.add_argument('--radius', default=2e-1, type=float)
parser.add_argument('--robustOptimizer', default='trust', type=str)

# general
parser.add_argument('-p', '--plot', action='store_true')

# save
parser.add_argument('-s', '--save', action='store_true')

# parse
args = parser.parse_args()

print(args)

# args.epochs = 20
# args.verbose = True
# args.robust = True
# args.radius = .15
# args.plot = True




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
trainer = TrainerSGD(opt, robustOptimizer=args.robustOptimizer, max_epochs=args.epochs)

# train!
results_train = trainer.train(fctn, (x_train, y_train, s_train), (x_val, y_val, s_val), (x_test, y_test, s_test),
                              verbose=args.verbose, robust=args.robust, radius=args.radius)
print([row[1] for row in results_train['history']['values'][1:]])


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

    plt.figure()

    for j in ('full', 's = 0', 's = 1'):
        fpr, tpr, auc = itemgetter(*('fpr', 'tpr', 'auc'))(results_eval['train'][j])
        plt.plot(fpr, tpr, label=j + ': AUC = %0.4f' % auc)

    plt.plot(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), '--', label='rand')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

    plt.figure()

    visualize_unfair_data((x_train, y_train, s_train), fctn.net, show_orig=True, domain=(-2, 2, -2, 2))
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    plt.figure()


#%%

if args.verbose:
    from pprint import pprint
    pprint(results_eval['train']['fairness'])

#%% saving results
if args.save:
    import pickle
    import os
    dir_name = 'unfair_2d_results/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # make filenmae
    filename = ''

    if args.robust:
        filename += 'robust' + '--' + args.robustOptimizer + ('--r_%0.2e' % args.radius)
    else:
        filename += 'nonrobust'


    print('Saving as...')
    print(filename)

    with open(dir_name + filename + '.pkl', 'wb') as f:
        results = {'results_train': results_train, 'results_eval': results_eval, 'args': args}
        pickle.dump(results, f)





