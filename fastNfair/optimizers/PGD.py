import torch
from copy import deepcopy


def projection(s_x, radius):
    if s_x.norm() > radius:
        s_x = (radius / s_x.norm()) * s_x

    return s_x


class ProjectedGradientDescent:

    def __init__(self, max_iter = 10):
        super(ProjectedGradientDescent, self).__init__()
        self.max_iter = max_iter

    def solve(self, function, x, step_size, radius):
        x = x.clone().detach().requires_grad_(True)

        for _ in range(self.max_iter):
            y = function(x)
            y.backward()

            # taking step in the gradient direction
            dfc = x.grad
            x_1 = x - step_size * dfc

            while function(x_1) >= function(x):
                step_size = step_size / 1.5
                x_1 = x - step_size * dfc

            x_1 = projection(x_1, radius)
            # value = (x - x_1).norm()

            if (x - x_1).norm() < .001:
                break

            x.grad.zero_()  # reset the gradient for the next step
            x = x_1.clone().detach().requires_grad_(True)  # update x after zeroing gradients

        return x


if __name__ == "__main__":
    x = torch.tensor([4.0, 4.0], requires_grad=True)
    print("x:", x)

    dot_product_fn = lambda x: torch.dot(x, x)  # define the function
    opt = ProjectedGradientDescent(max_iter=10)
    x = opt.solve(dot_product_fn, torch.tensor([2.0, 10.0]), 1, 1)  # pass the function, not the result

    print(x)
