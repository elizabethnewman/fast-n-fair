import torch
import numpy as np


def generate_gaussian_normal_data(mu, sigma, label, attribute, n=100):
    x = 1.0 * torch.tensor(mu).reshape(-1, 2) + torch.randn(n, 2) @ (1.0 * torch.tensor(sigma))
    return x, np.array(n * [label]), np.array(n * [attribute])


def generate_binary_data():
    blue_label, red_label = 1, -1
    attr_A, attr_B = 'A', 'B'

    # generate cluster for each class
    # mu_red_A, sigma_red_A = [3, -3], [[2, 0], [0, 1]]  # class 0, attr A
    # mu_red_B, sigma_red_B = [-3, -3], [[1, 0], [0, 2]]  # class 0, attr B
    # mu_blue_A, sigma_blue_A = [3, 3], [[1, 0], [0, 2]]  # class 1, attr A
    # mu_blue_B, sigma_blue_B = [-3, 3], [[2, 0], [0, 1]]  # class 1, attr B

    mu_red_A, sigma_red_A = [3, 0], [[1, 0], [0, 1]]    # class 0, attr A
    mu_red_B, sigma_red_B = [0, -5], [[2, -0.7], [-0.7, 1]]  # class 0, attr B
    mu_blue_A, sigma_blue_A = [-3, 0], [[1, 0.5], [0.5, 2]]  # class 1, attr A
    mu_blue_B, sigma_blue_B = [0, 1], [[1, 0], [0, 3]]   # class 1, attr B
    mu_blue_B2, sigma_blue_B2 = [0, 0], [[1, 0], [0, 1]]  # class 1, attr B

    # from SearchFair
    # mu_red_A, sigma_red_A = [4.5, -1.5], [[1, 0], [0, 1]]  # negative class, positive sens attr (unprotected)
    # mu_red_B, sigma_red_B = [2, -2], [[1, 0], [0, 1]]  # negative class, negative sens attr (protected)
    #
    # mu_blue_A, sigma_blue_A = [3, -1], [[1, 0], [0, 1]]  # positive class, positive sens attr
    # mu_blue_A2, sigma_blue_A2 = [1, 4], [[0.5, 0], [0, 0.5]]  # positive class, positive sens attr
    # mu_blue_B, sigma_blue_B = [2.5, 2.5], [[1, 0], [0, 1]] # positive class, negative sens attr

    # red class, sensitive attribute A
    x_red_A, label_red_A, s_red_A = generate_gaussian_normal_data(mu_red_A, sigma_red_A, red_label, 'A')

    # blue class, sensitive attribute A
    x_blue_A, label_blue_A, s_blue_A = generate_gaussian_normal_data(mu_blue_A, sigma_blue_A, blue_label, 'A')

#    x_blue_A2, label_blue_A2, s_blue_A2 = generate_gaussian_normal_data(mu_blue_A2, sigma_blue_A2, blue_label, 'A')

    # red class, sensitive attribute B
    x_red_B, label_red_B, s_red_B = generate_gaussian_normal_data(mu_red_B, sigma_red_B, red_label, 'B')

    # blue class, sensitive attribute B
    x_blue_B, label_blue_B, s_blue_B = generate_gaussian_normal_data(mu_blue_B, sigma_blue_B, blue_label, 'B')

    # blue class, sensitive attribute B2
    # x_blue_B2, label_blue_B2, s_blue_B2 = generate_gaussian_normal_data(mu_blue_B2, sigma_blue_B2, blue_label, 'B')


    # store as one dataset
    x = torch.cat((x_red_A, x_blue_A, x_red_B, x_blue_B), dim=0)
    y = np.concatenate((label_red_A, label_blue_A, label_red_B, label_blue_B))
    s = np.concatenate((s_red_A, s_blue_A, s_red_B, s_blue_B))

    return x, y, s, (blue_label, red_label), (attr_A, attr_B)


def create_confidence_level(x):
    mu = x.mean(dim=0, keepdim=True)
    x -= mu
    C = (1 / (x.shape[0] - 1)) * (x.T @ x)
    d, v = torch.linalg.eig(C)
    d, v = torch.real(d), torch.real(v)
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(-1, 1)
    xx = torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)
    yy = xx @ v @ torch.diag(d) @ v.T + mu

    return yy


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x, y, s, (blue_label, red_label), (attr_A, attr_B) = generate_binary_data()

    # create plot
    plt.figure()

    # BLUE A
    idx_blue_A = (y == blue_label) * (s == attr_A)
    plt.plot(x[idx_blue_A, 0], x[idx_blue_A, 1], 'co', markersize=12, zorder=0)
    plt.scatter(x[idx_blue_A, 0], x[idx_blue_A, 1], c='white', marker='${' + attr_A + '}$', zorder=1)

    # draw approximate confidence levels
    yy = create_confidence_level(x[idx_blue_A])
    plt.plot(yy[:, 0], yy[:, 1], '--', linewidth=3)

    # RED A
    idx_red_A = (y == red_label) * (s == attr_A)
    plt.plot(x[idx_red_A, 0], x[idx_red_A, 1], 'mo', markersize=12, zorder=0)
    plt.scatter(x[idx_red_A, 0], x[idx_red_A, 1], c='white', marker='${' + attr_A + '}$')

    yy = create_confidence_level(x[idx_red_A])
    plt.plot(yy[:, 0], yy[:, 1], '--', linewidth=3)

    # BLUE B
    idx_blue_B = (y == blue_label) * (s == attr_B)
    plt.plot(x[idx_blue_B, 0], x[idx_blue_B, 1], 'cs', markersize=12, zorder=0)
    plt.scatter(x[idx_blue_B, 0], x[idx_blue_B, 1], c='black', marker='${' + attr_B + '}$')

    yy = create_confidence_level(x[idx_blue_B])
    plt.plot(yy[:, 0], yy[:, 1], '--', linewidth=3)

    # RED B
    idx_red_B = (y == red_label) * (s == attr_B)
    plt.plot(x[idx_red_B, 0], x[idx_red_B, 1], 'ms', markersize=12, zorder=0)
    plt.scatter(x[idx_red_B, 0], x[idx_red_B, 1], c='black', marker='${' + attr_B + '}$')

    yy = create_confidence_level(x[idx_red_B])
    plt.plot(yy[:, 0], yy[:, 1], '--', linewidth=3)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
