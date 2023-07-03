import tempfile
import os
import pandas as pd
import six.moves.urllib as urllib
import numpy as np
import torch
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


def load_lsat():
    # https: // www.tensorflow.org / responsible_ai / fairness_indicators / tutorials / Fairness_Indicators_Pandas_Case_Study
    _DATA_ROOT = tempfile.mkdtemp(prefix='lsat-data')
    _DATA_PATH = 'https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv'
    _DATA_FILEPATH = os.path.join(_DATA_ROOT, 'bar_pass_prediction.csv')

    data = urllib.request.urlopen(_DATA_PATH)

    _LSAT_DF = pd.read_csv(data)

    # To simpliy the case study, we will only use the columns that will be used for
    # our model.
    _COLUMN_NAMES = [
        'dnn_bar_pass_prediction',
        'gender',
        'lsat',
        'pass_bar',
        'race1',
        'ugpa',
    ]

    _LSAT_DF.dropna()
    _LSAT_DF['gender'] = _LSAT_DF['gender'].astype(str)
    _LSAT_DF['race1'] = _LSAT_DF['race1'].astype(str)
    _LSAT_DF = _LSAT_DF[_COLUMN_NAMES]

    return _LSAT_DF


def generate_lsat(p_train=0.8):
    df = load_lsat()

    x = np.array(df[['lsat', 'ugpa']])
    y = np.array(df['pass_bar'])

    # create binary attributes (white or not identified as white)
    s = np.array(1 * (df['race1'] == 'white'))

    # convert to tensors
    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.int64)
    s = torch.from_numpy(s).to(torch.int64)

    # split data for each group
    x_train, x_test = torch.empty(0, x.shape[1]), torch.empty(0, x.shape[1])
    y_train, y_test = torch.empty(0, dtype=y.dtype), torch.empty(0, dtype=y.dtype)
    s_train, s_test = torch.empty(0, dtype=s.dtype), torch.empty(0, dtype=s.dtype)

    idx = torch.randperm(x.shape[0])
    n = []
    for yi in [0, 1]:
        for sj in [0, 1]:
            n.append(((y == yi) * (s == sj)).sum().item())
    n_max = min(n)

    for yi in [0, 1]:
        for sj in [0, 1]:
            idx_ij = ((y == yi) * (s == sj))
            x_ij = x[idx_ij]
            y_ij = y[idx_ij]
            s_ij = s[idx_ij]
            nn = y_ij.numel()
            n_train = min(math.floor(p_train * nn), math.floor(p_train * n_max))

            idx = torch.randperm(nn)
            x_train = torch.cat((x_train, x_ij[idx[:n_train]]), dim=0)
            y_train = torch.cat((y_train, y_ij[idx[:n_train]]))
            s_train = torch.cat((s_train, s_ij[idx[:n_train]]))

            x_test = torch.cat((x_test, x_ij[idx[:n_train]]), dim=0)
            y_test = torch.cat((y_test, y_ij[idx[:n_train]]))
            s_test = torch.cat((s_test, s_ij[idx[:n_train]]))

    x_train, x_m, x_s = normalize_data(x_train)
    x_test -= x_m
    x_test /= x_s

    return (x_train, y_train, s_train), (x_test, y_test, s_test)


def normalize_data(x):
    # normalize entries to lie between 0 and 1
    opts = {'dim': 0, 'keepdim': True}
    # x = x / x.max(**opts)[0]
    # x = (x - x.min(**opts)[0]) / (x.max(**opts)[0] - x.min(**opts)[0])
    x_m = x.mean(**opts)
    x_s = x.std(**opts)
    x -= x_m
    x /= x_s
    return x, x_m, x_s


def visualize_lsat_data(data, net=None, domain=None):
    # input is a tuple of data
    x, y, s = data

    if domain is None:
        domain = (x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max())

    # plot everything
    x_grid, y_grid = torch.meshgrid(torch.linspace(domain[0], domain[1], 300),
                                    torch.linspace(domain[0], domain[1], 300), indexing='ij')
    xy_grid = torch.cat((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), dim=1)

    if net is not None:
        c_grid = 1 * (net(xy_grid)[0] > 0)

        # tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cmap = mpl.colors.ListedColormap(['r', 'b'])
        plt.contourf(x_grid, y_grid, c_grid.reshape(x_grid.shape), alpha=0.25, cmap=cmap)

    # create classifiers
    t = torch.linspace(domain[0], domain[1], 100)

    if net is not None:
        with torch.no_grad():
            tt2 = -(net[0].K[0] / net[0].K[1]) * t - (net[0].b / net[0].K[1])

        plt.plot(t, tt2, 'k', linewidth=4, label='predicted')

    cmap = mpl.colors.ListedColormap(['tab:red', 'tab:blue'])

    idx1 = (s == 0)
    plt.scatter(x[idx1, 0], x[idx1, 1], None, y[idx1], marker='$W$', cmap=cmap)

    idx2 = (s == 1)
    plt.scatter(x[idx2, 0], x[idx2, 1], None, y[idx2], marker='$N$', cmap=cmap)



if __name__ == "__main__":
    import pprint
    import matplotlib.pyplot as plt

    df = load_lsat()

    (x, y, s), (xt, yt, st) = generate_lsat(p_train=0.9)


    visualize_lsat_data((x, y, s))
    plt.show()

    visualize_lsat_data((xt, yt, st))
    plt.show()