import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_adult(load_dir='./raw/adult/'):
    headers = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
               'salary')
    data_train = pd.read_csv(load_dir + 'adult.data', sep=",")

    data_test = pd.read_csv(load_dir + 'adult.test', sep=",", header=1)

    return data_train, data_test, headers


def generate_adult(load_dir='./raw/adult/', attr='race'):
    data_train, data_test, headers = load_adult(load_dir=load_dir)

    # convert to numpy array
    data_train = np.array(data_train)
    data_test = np.array(data_test)

    # extract continuous variables (except 'fnlwgt', which is on a different scale)
    idx = []
    for name in ('age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'):
        idx.append(headers.index(name))

    # put data in format for us to learn
    x_train = torch.from_numpy(data_train[:, idx].astype(np.float32))
    x_train, x_m, x_s = normalize_data(x_train)
    y_train = torch.from_numpy(1 * (data_train[:, headers.index('salary')] == ' <=50K'))

    _, s = np.unique(data_train[:, headers.index(attr)], return_inverse=True)
    s_train = torch.from_numpy(s)

    x_test = torch.from_numpy(data_test[:, idx].astype(np.float32))
    # x_test = normalize_data(x_test)
    x_test -= x_m
    x_test /= x_s
    y_test = torch.from_numpy(1 * (data_test[:, headers.index('salary')] == ' <=50K.'))
    _, s = np.unique(data_test[:, headers.index(attr)], return_inverse=True)
    s_test = torch.from_numpy(s)

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


def visualize_adult_data(data, net=None, domain=None):
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


    data_train, data_test, headers = load_adult()

    (x_train, y_train, s_train), (x_test, y_test, s_test) = generate_adult(attr='sex')

    # plt.scatter(x_train[:, 0], x_train[:, 1], None, y_train)
    # plt.show()

    visualize_adult_data((x_train, y_train, s_train))
    plt.show()






