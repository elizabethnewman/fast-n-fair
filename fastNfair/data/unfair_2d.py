import torch
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


def generate_unfair_data(n_samples_per_class=100, p1=0.5, p2=0.5, push_unfair=0.05, v=(1.0, 1.0),
                         classifier=lambda z: -z[:, 0] - z[:, 1] + 1):
    # p1: percent of label = 0 (not hired) in group A
    # p2: percent of label = 1 (hired) in group A
    # push_unfair: amount of shift (group B is given advantage, group A is given disadvantage)
    # v : direction of push
    # classifier(x) > 0 or <= 0

    # generate 4x number of points
    x = torch.rand(4 * n_samples_per_class, 2)

    # select samples per class in [0, 1] above y = -x + 1
    x1 = x[classifier(x) > 0]

    # select samples per class in [0, 1] below y = -x + 1
    x2 = x[classifier(x) <= 0]

    # truncate and label class 0
    x1 = x1[:n_samples_per_class]
    y1 = torch.zeros(n_samples_per_class, dtype=torch.int64)

    # randomly select attribute A with probability p1
    s1 = torch.zeros_like(y1)
    n1 = max(1, math.floor(p1 * s1.numel()))
    idx1 = torch.randperm(s1.numel())[:n1]
    s1[idx1] = 1.0

    # truncate and label class 1
    x2 = x2[:n_samples_per_class]
    y2 = torch.ones(n_samples_per_class, dtype=torch.int64)

    # randomly select attribute A with probability p2
    s2 = torch.zeros_like(y2)
    n2 = max(1, math.floor(p2 * s2.numel()))
    idx2 = torch.randperm(s1.numel())[:n2]
    s2[idx2] = 1.0

    # concatenate
    x = torch.cat((x1, x2), dim=0)
    y = torch.cat((y1, y2))
    s = torch.cat((s1, s2))

    # shift the data to create overlapping
    # group A (s == 0) more likely not to be hired
    x[(s == 0) * (y == 1)] -= push_unfair * torch.tensor(v)

    # group B (s == 1) more likely to be hired
    x[(s == 1) * (y == 0)] += push_unfair * torch.tensor(v)

    # shuffle
    idx = torch.randperm(x.shape[0])
    x, y, s = x[idx], y[idx], s[idx]

    return x, y, s


def visualize_unfair_data(data, net=None, show_orig=True, domain=(0, 1, 0, 1)):
    # input is a tuple of data
    x, y, s = data

    # plot everything
    x_grid, y_grid = torch.meshgrid(torch.linspace(domain[0], domain[1], 300),
                                    torch.linspace(domain[0], domain[1], 300), indexing='ij')
    xy_grid = torch.cat((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), dim=1)

    if net is not None:
        c_grid = 1 * (net(xy_grid)[0] > 0)
    else:
        fctn_orig = lambda xy: 1 * (-xy[:, 0] + 1 < xy[:, 1])
        c_grid = fctn_orig(xy_grid)

    # tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cmap = mpl.colors.ListedColormap(['r', 'b'])
    plt.contourf(x_grid, y_grid, c_grid.reshape(x_grid.shape), alpha=0.25, cmap=cmap)

    # create classifiers
    t = torch.linspace(domain[0], domain[1], 100)

    # original line
    if show_orig or net is None:
        tt = -t.reshape(-1, 1) + 1
        plt.plot(t, tt, 'k--', linewidth=4, label='orig')

    if net is not None:
        with torch.no_grad():
            tt2 = -(net[0].K[0] / net[0].K[1]) * t - (net[0].b / net[0].K[1])

        plt.plot(t, tt2, 'k', linewidth=4, label='predicted')

    cmap = mpl.colors.ListedColormap(['tab:red', 'tab:blue'])

    idx1 = (s == 0)
    plt.scatter(x[idx1, 0], x[idx1, 1], None, y[idx1], marker='$A$', cmap=cmap)

    idx2 = (s == 1)
    plt.scatter(x[idx2, 0], x[idx2, 1], None, y[idx2], marker='$B$', cmap=cmap)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import fastNfair.utils.statistics as stats
    torch.manual_seed(42)

    x, y, s = generate_unfair_data(p1=0.75, p2=0.25, push_unfair=0.1, v=(1.0, 3.0))


    visualize_unfair_data((x, y, s))
    plt.show()


    # a bad model
    y_pred = 1 * (x[:, 1] > 0.5)
    results = stats.compute_statistics(y, y_pred)

    # results per group

    y_pred_A = 1 * (x[s == 0][:, 1] > 0.5)
    results_A = stats.compute_statistics(y[s == 0], y_pred[s == 0])

    y_pred_B = 1 * (x[s == 1][:, 1] > 0.5)
    results_B = stats.compute_statistics(y[s == 1], y_pred[s == 1])

