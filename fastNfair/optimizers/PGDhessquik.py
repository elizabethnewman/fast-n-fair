import torch
from copy import deepcopy



class ProjectedGradientDescentv3:

    def __init__(self, max_iter = 200, per_sample=False):
        super(ProjectedGradientDescentv3, self).__init__()
        self.max_iter = max_iter
        self.per_sample = per_sample

    def projection(self, s_x, radius, per_sample):
        if per_sample:
            norm_delta = torch.sqrt(torch.sum(s_x**2, dim=1))
            s_x = s_x / torch.reshape(norm_delta, (-1,1))

            vector_radius = torch.ones_like(s_x) * radius
            s_x = s_x * torch.minimum(vector_radius, torch.reshape(norm_delta, (-1,1)))
        else:
            if s_x.norm() > radius:
                s_x = (radius / s_x.norm()) * s_x

        return s_x

    def solve(self, function, x, step_size, radius):
        delta_x = torch.zeros_like(x)
        x = x.clone().detach().requires_grad_(True)

        for _ in range(self.max_iter):
            y, dfc = function(x + delta_x, do_gradient = True)[:2]
            #y = function(x)
            #y.backward()

            # taking step in the gradient direction
            #dfc = x.grad
            delta_x1 = delta_x - step_size * dfc.squeeze(-1)

            while function(x + delta_x1)[0] >= y:
                step_size = step_size / 1.5
                delta_x1 = delta_x - step_size * dfc.squeeze(-1)

            delta_x1 = self.projection(delta_x1, radius, self.per_sample)
            # value = (x - x_1).norm()

            if (delta_x - delta_x1).norm() < .000001:
                break

            #x.grad.zero_()  # reset the gradient for the next step
            delta_x = delta_x1.clone()

        return x + delta_x


if __name__ == "__main__":
    import torch.nn as nn
    import hessQuik.networks as net
    import hessQuik.layers as lay
    import hessQuik.activations as act
    from fastNfair.objective_functions import ObjectiveFunctionMSE
    import matplotlib.pyplot as plt

    class TestFunctionPGD(nn.Module):
        def __init__(self, fctn, y):
            super(TestFunctionPGD, self).__init__()
            self.fctn = fctn
            self.y = y

        def forward(self, x, do_gradient=False, do_Hessian=False):

            loss, dloss, d2loss, info = self.fctn(x, self.y, do_gradient=do_gradient, do_Hessian=do_Hessian)

            return loss, dloss, d2loss, info


    x = torch.tensor([[0.5, 1.0], [2.0, 3.0], [-1.0, -1.0]])
    y = torch.tensor([[1.0, -2.0], [1.0, 1.0], [-2.0, 0.5]])

    my_net = lay.singleLayer(2, 2, act.identityActivation(), bias=False)
    my_net.K = nn.Parameter(torch.eye(2))

    K_opt = torch.linalg.lstsq(x.T, y.T)[0]
    print(K_opt)

    fctn = ObjectiveFunctionMSE(my_net)
    test_fctn = TestFunctionPGD(fctn, y)

    tmp = test_fctn(x)
    opt = ProjectedGradientDescentv3(per_sample=True)

    delta = 1.5
    xt = opt.solve(test_fctn, x, 1, 1.5)
    print(xt)

    #x = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
    #print("x:", x)
    #print("star")

    #log_sum_exp_func = lambda x: torch.logsumexp(x, 0) # define the function
    #print(log_sum_exp_func(x))
    #opt = ProjectedGradientDescent(max_iter=10)
    #x = opt.solve(log_sum_exp_func, torch.tensor([2.0, 3.0, 4.0]), 1, 1)  # pass the function, not the result

    #print(x)
