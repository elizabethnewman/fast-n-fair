import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl


def gray_to_color(x, y, colors, p):
    """
    x (torch.Tensor) : grayscale images of shape (# images, 1, height, width)
    y (torch.LongTensor) : labels of shape (# images,)
    colors (list)   : list of tuples of RGB color weights; each entry in a tuple must be between 0.0 and 1.0
    p (float) : probability (between 0.0 and 1.0) of flipping color label; smaller p means stronger correlation between color and digit
    """

    # number of colors
    n_colors = len(colors)

    # assign color label based on digit
    n_labels = len(torch.unique(y))
    color_labels = torch.zeros(y.shape).long()

    for i in range(1, n_colors):
        color_labels[y % n_colors == i] = i

    # swap label randomly with 25% probability
    # these are the attribute labels and they are highly correlated with the digit
    change_prob = torch.rand(x.shape[0])
    idx = change_prob < p
    shift = torch.randint(n_colors - 1, (sum(idx).item(),)) + 1
    color_labels[idx] = (color_labels[idx] + shift) % n_colors

    # concatenate images (subsample for computational convenience)
    x_color = torch.cat((x[..., ::2, ::2], x[..., ::2, ::2]), dim=1)

    # create false colors via weighting
    unique_color_labels = torch.unique(color_labels)
    for label in unique_color_labels:
        x_color[color_labels == label] = (x_color[color_labels == label]
                                          * torch.tensor(colors[label]).reshape(1, -1, 1, 1))

    return x_color, color_labels


def generate_mnist(n_train=1000, n_test=100, labels=(0, 1)):
    # choose number of images (per class)

    # download data
    mnist = datasets.MNIST('./raw/mnist', train=True, download=True)
    mnist_test = datasets.MNIST('./raw/mnist', train=False, download=True)

    # select labels
    x_train, y_train = torch.empty(0), torch.empty(0)
    x_test, y_test = torch.empty(0), torch.empty(0)
    for label in labels:
        mnist_data = mnist.data[mnist.targets == label]
        mnist_targets = mnist.targets[mnist.targets == label]

        idx = torch.randperm(mnist_data.shape[0])
        x_train = torch.cat((x_train, (mnist_data[idx[:n_train]].float() / 255.).unsqueeze(1)), dim=0)
        y_train = torch.cat((y_train, mnist_targets[idx[:n_train]]))

        mnist_test_data = mnist_test.data[mnist_test.targets == label]
        mnist_test_targets = mnist_test.targets[mnist_test.targets == label]

        idx_test = torch.randperm(mnist_test_data.shape[0])
        x_test = torch.cat((x_test, (mnist_test_data[idx_test[:n_test]].float() / 255.).unsqueeze(1)), dim=0)
        y_test = torch.cat((y_test, mnist_test_targets[idx_test[:n_test]]))

    # shuffle
    idx = torch.randperm(x_train.shape[0])
    x_train, y_train = x_train[idx], y_train[idx]

    idx = torch.randperm(x_test.shape[0])
    x_test, y_test = x_test[idx], y_test[idx]

    return (x_train, y_train), (x_test, y_test)


def bernoulli(p, n):
    # assume p is probability
    return 1 * (torch.rand(n) < p)


def xor(a, b):
    # assume a and b are 0 or 1
    return (a - b).abs()


def generate_color_mnist_binary(x, y, p=0.5):
    # x is (N, C, H, W)

    # get number of labels
    n = y.numel()

    # scale images down for efficiency
    x = x[..., ::2, ::2]

    # split into binary
    mid_digit = torch.unique(y).mean()
    labels = 1 * (y > mid_digit)

    # flip labels with probability 0.25
    # flip if bernoulli = 1
    # labels = xor(labels, bernoulli(0.25, n))

    # get color labels
    colors = xor(labels, bernoulli(p, n))

    # create color images
    x = torch.cat((x, x), dim=1)

    # zero out channel with wrong color
    x[torch.arange(n), (1 - colors).long()] *= 0

    # some historic data says color1 is more likely to be classified as 1 and color2 more likely classified as 0


    return x, labels, colors


def visualize_color_mnist(data, n_rows=2):
    x, y, c = data
    cmap = mpl.colormaps['cool']
    n_img = (x.shape[0] // n_rows) * n_rows
    for i in range(n_img):
        plt.subplot(n_rows, n_img // n_rows, i + 1)
        tmp = torch.cat((x[i], 0.0 * x[i][0:1]), dim=0)
        plt.imshow(tmp.permute(1, 2, 0).squeeze(), cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        # plt.title(digit)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = generate_mnist()

    x, y, s = generate_color_mnist_binary(x_train, y_train, 0.1)

    n = 64
    visualize_color_mnist(x[:n], y[:n], s[:n], n_rows=4)
    plt.show()




