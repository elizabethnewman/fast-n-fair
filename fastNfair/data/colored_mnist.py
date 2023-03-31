import torch
from torchvision import datasets
import matplotlib.pyplot as plt


def generate_colored_mnist(n_train=100, n_val=100, colors=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), p=0.5):

    n_val = min(n_val, 50000 - n_train)

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    x = mnist.data[:n_train + n_val]
    y = mnist.targets[:n_train + n_val]

    # normalize
    x = x.to(torch.float32)
    x = ((x - (x.min(dim=1, keepdim=True)[0]).min(dim=2, keepdim=True)[0])
         / ((x.max(dim=1, keepdim=True)[0]).max(dim=2, keepdim=True)[0] - (x.min(dim=1, keepdim=True)[0]).min(dim=2, keepdim=True)[0]))

    # number of colors
    n_colors = len(colors)

    # assign color label based on digit
    color_labels = torch.zeros(y.shape).long()

    for i in range(1, n_colors):
        color_labels[y % n_colors == i] = i

    # swap label randomly with p% probability
    # these are the attribute labels and they are highly correlated with the digit
    change_prob = torch.rand(x.shape[0])
    idx = change_prob < p
    shift = torch.randint(n_colors - 1, (sum(idx).item(),)) + 1
    color_labels[idx] = (color_labels[idx] + shift) % n_colors

    # concatenate images
    x = x.unsqueeze(1)
    x_color = torch.cat((x, x, x), dim=1)

    # create false colors via weighting
    unique_color_labels = torch.unique(color_labels)
    for label in unique_color_labels:
        x_color[color_labels == label] = (x_color[color_labels == label]
                                          * torch.tensor(colors[label]).reshape(1, -1, 1, 1))

    x_train, y_train, c_train = x_color[:n_train], y[:n_train], color_labels[:n_train]
    x_val, y_val, c_val = x_color[n_train:], y[n_train:], color_labels[n_train:]

    return (x_train, y_train, c_train), (x_val, y_val, c_val)


def plot_digits(x, y, digit, n_img=4, n_rows=1):
    """
    Plot the images

    x (torch.Tensor)        : images of shape (# images, # channels, height, width)
    y (torch.LongTensor)    : labels of shape (# images,)
    n_img (int)             : number of images to display
    n_rows (int, optional)  : number of rows of images to display (default=1)
    """
    x = x[y == digit]

    plt.figure()
    if x.ndim == 3 or x.shape[1] == 1:
        plt.set_cmap('gray')

    n_img = (n_img // n_rows) * n_rows
    for i in range(n_img):
        plt.subplot(n_rows, n_img // n_rows, i + 1)
        plt.imshow(x[i].permute(1, 2, 0).squeeze())
        plt.xticks([])
        plt.yticks([])
        # plt.title(digit)

    plt.show()


if __name__ == "__main__":

    d1, d2 = generate_colored_mnist(n_train=500, p=0.8)

    plt.figure()
    plot_digits(d1[0], d1[1], 0, n_img=16, n_rows=4)
    plt.show()


