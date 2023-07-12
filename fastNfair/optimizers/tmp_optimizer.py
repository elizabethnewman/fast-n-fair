import torch
from fastNfair.optimizers import BisectionMethod
from copy import deepcopy
import matplotlib.pyplot as plt


class TrustRegionSubproblem:
    r"""
    Our goal is to solve the minimization problem
        min_{p} f(x + p)    subject to      ||p||_2 <= r
    where x is one sample and p is the perturbation.

    To do this, we first approximate f by a quadratic model q
        f(x + p) \approx q(p) = f(x) + p^T @ grad_x f(x) + 0.5 * p^T @ hess_x f(x) @ p

    We reformulate the problem of minimizing the quadratic model
        min_{p} q(p)    subject to      ||p||_2 <= r

    """
    def __init__(self, max_iter=100, tol=1e-8, per_sample=False):
        self.opt = BisectionMethod(max_iter, tol)
        self.per_sample = per_sample

    def solve(self, fctn, x, delta=None):

        # compute value, gradient, and Hessian at current point
        fc, dfc, d2fc = fctn(x, do_gradient=True, do_Hessian=True)[:3]

        # check if Hessian has any NaNs or Infs
        if torch.isnan(d2fc).any() or torch.isinf(d2fc).any():
            print('here')
        # pre-compute svd per sample
        # u, s, _ = torch.linalg.svd(d2fc.squeeze(-1), full_matrices=False)
        s, u = torch.linalg.eig(d2fc.squeeze(-1))
        s, u = torch.real(s), torch.real(u)

        # per sample way
        p = torch.empty(x.shape[0], x.numel() // x.shape[0])
        for i in range(x.shape[0]):

            # find number of nonzero eigenvalues
            tol = 1e-8
            idx = (s[i].abs() > tol)
            si = s[i][idx]
            ui = u[i][:, idx]
            # solve unconstrained problem for sample i
            pi_a = lambda a: torch.diag(1 / (si + a)) @ (ui.T @ (-dfc[i]))
            # pi_a = lambda a: (d2fc[i].squeeze() + a * torch.eye(d2fc[i].squeeze().shape[0])).pinverse() @ (-dfc[i])

            alpha = 0.0
            pi = pi_a(alpha)

            r = delta
            if r is None or r == 0:
                r = pi.norm()

            if pi.norm() > r:
                # 1D function to find point on boundary
                g = lambda a: pi_a(a).norm() - r
                # g = lambda a: 1.0 / pi_a(a).norm().item() - 1.0 / r

                # form bracket between the eigenvalues of the Hessian
                # alpha_low = 0
                # g_low = g(alpha_low)
                # si = torch.sort(s[i])[0]
                # for j in range(si.numel() - 1, -1, -1):
                #     alpha_high = si[j]
                #     g_high = g(alpha_high)
                #
                #     if g_low * g_high < 0:
                #         break
                #     else:
                #         alpha_low = deepcopy(alpha_high)
                #         g_low = deepcopy(g_high)

                alpha_low, alpha_high = 0, ((dfc[i].numel() * dfc[i].norm()) / r + si.abs().min()).item()

                # if g(alpha_low) * g(alpha_high) > 0:
                #     print('here')
                # solve for alpha
                alpha = self.opt.solve(g, alpha_low, alpha_high)

                # get search direction (before application of u)
                pi = pi_a(alpha)

            p[i] = (ui @ pi.reshape(ui.shape[1], -1)).reshape(-1)

            # # # check optimality conditions
            # _, dfi = fctn(x + p, do_gradient=True)[:2]
            # #
            # print('stationarity f:', (dfi[i].reshape(-1) + alpha * p[i].reshape(-1)).norm().item())
            # print('stationarity q:', (dfc[i].reshape(-1) + (d2fc[i].squeeze() @ p[i].reshape(-1, 1)).reshape(-1)
            #                           + alpha * p[i].reshape(-1)).norm().item())
            #
            # print('comp. slackness:', alpha * (pi.norm().item() - r))
            # print('dual:', alpha)
            # print('primal:', pi.norm().item() - r)

        # update
        xt = x + p.reshape(x.shape)
        # print(p.reshape(x.shape))
        # print(fctn.fctn.net[0].K)
        return xt, {'s': s.clone(), 'delta': delta}


def plot_g(g, alpha, alpha_low, alpha_high, n=100):

    a = torch.linspace(alpha_low, alpha_high, n)

    y = []
    for i in range(n):
        y.append(g(a[i]))

    plt.plot(a, y)

    # plot horizontal line at y = 0
    plt.plot(a, torch.zeros_like(a), color='k')

    # plot solution
    plt.plot(alpha, g(alpha), 'o', color='r')

    return torch.tensor(y)


class TestFunctionTrustRegion(torch.nn.Module):
    def __init__(self, fctn, y):
        super(TestFunctionTrustRegion, self).__init__()
        self.fctn = fctn
        self.y = y

    def forward(self, x, do_gradient=False, do_Hessian=False):

        loss, dloss, d2loss, info = self.fctn(x, self.y, do_gradient=do_gradient, do_Hessian=do_Hessian)

        return loss, dloss, d2loss, info


if __name__ == "__main__":
    import torch.nn as nn
    import hessQuik.networks as net
    import hessQuik.layers as lay
    import hessQuik.activations as act
    from fastNfair.objective_functions import ObjectiveFunctionMSE
    from fastNfair.training.adversarial_training import ObjectiveFunctionMaximize
    import matplotlib.pyplot as plt

    # reproducibility and numerical stability
    torch.manual_seed(1234)
    torch.set_default_dtype(torch.float64)

    with torch.no_grad():

        # setup least squares problem
        K = torch.randn(3, 2)
        y = torch.randn(K.shape[0], 1)
        x_opt = torch.linalg.lstsq(K, y)[0].reshape(-1, 1)

        my_net = lay.singleLayer(K.shape[1], y.shape[0], bias=False)
        my_net.K = nn.Parameter(K.T)

        fctn = ObjectiveFunctionMSE(my_net)
        opt = TrustRegionSubproblem()
        # test_fctn = TestFunctionTrustRegion(fctn, y.T)
        test_fctn = ObjectiveFunctionMaximize(fctn, y.T)

        # get landscape
        n = 100
        xy = torch.linspace(-5, 5, n)
        x_grid, y_grid = torch.meshgrid(xy, xy, indexing='ij')
        xy_grid = torch.concatenate((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), dim=1)

        z_grid = torch.zeros(xy_grid.shape[0], 1)
        for i in range(xy_grid.shape[0]):
            z_grid[i] = test_fctn(xy_grid[i].reshape(1, -1))[0]

        # sanity check - make sure we solve a least squares problem with large trust region radius
        delta = 1e3
        plt.contourf(x_grid, y_grid, z_grid.reshape(x_grid.shape))
        for i in range(3):
            x0 = 2 * torch.randn_like(x_opt)
            xt = opt.solve(test_fctn, x0.T, delta=delta)[0]
            plt.plot(x0[0], x0[1], 'cs', markersize=10)
            plt.plot(xt[:, 0], xt[:, 1], 'ro', markersize=10)

        plt.plot(x_opt[0], x_opt[1], 'y*', markersize=10)
        plt.axis('square')
        plt.show()

        # check with smaller radius
        delta = 1e0
        plt.contourf(x_grid, y_grid, z_grid.reshape(x_grid.shape))
        theta = torch.linspace(0, 2 * torch.pi, 100)
        for i in range(3):
            x0 = 2 * torch.randn_like(x_opt)
            xt = opt.solve(test_fctn, x0.T, delta=delta)[0]
            plt.plot(x0[0], x0[1], 'cs', markersize=10)

            # plot circle around x0
            plt.plot(x0[0] + delta * torch.cos(theta), x0[1] + delta * torch.sin(theta), 'k')

            plt.plot(xt[:, 0], xt[:, 1], 'wo', markersize=10)

        plt.plot(x_opt[0], x_opt[1], 'y*', markersize=10)
        plt.axis('square')
        plt.show()




