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
parser.add_argument('--n-train', default=10000, help='number of training points')
parser.add_argument('--n-val', default=2000, help='number of training points')
parser.add_argument('--n-test', default=1000, help='number of training points')

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


args.epochs = 10
args.verbose = True
#args.robust = False
#args.radius = .3
#args.plot = True

print(args)



#%% generate data

# for reproducibility
torch.manual_seed(args.seed)

# number of data points
n_train, n_val, n_test = args.n_train, args.n_val, args.n_test

(x_train, y_train, s_train), (x_test, y_test, s_test) = generate_adult(load_dir='../data/raw/adult/', attr='sex')


x_val, y_val, s_val = x_train[n_train:], y_train[n_train:], s_train[n_train:]
x_train, y_train, s_train = x_train[:n_train], y_train[:n_train], s_train[:n_train]
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
trainer = TrainerSGD(opt, robustOptimizer=args.robustOptimizer, max_epochs=args.epochs, batch_size= 100)

# train!
results_train = trainer.train(fctn, (x_train, y_train, s_train), (x_val, y_val, s_val), (x_test, y_test, s_test),
                              verbose=args.verbose, robust=args.robust, radius=args.radius)
print("Weights and biases of the network:")
print(my_net.state_dict())

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
if args.verbose:
    from pprint import pprint
    pprint(results_eval['train']['fairness'])

#%% saving results
if args.save:
    import pickle
    import os
    from pprint import pprint
    dir_name = 'adult_results/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # make filename
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

    with open(dir_name + filename + '.txt', 'w') as f:
        f.write(str(args))
        f.write(str(results_eval))






