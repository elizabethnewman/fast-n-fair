import torch.nn as nn
import numpy as np
from fastNfair.linesearch import ConstantLineSearch, ArmijoLineSearch


class GradientDescent:

    def __init__(self,
                 max_iter: int = 10,
                 abs_tol: float = 1e-8,
                 rel_tol: float = 1e-8,
                 linesearch=ArmijoLineSearch()):

        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.linesearch = linesearch
        self.headers = ('iter', 'f', '|df|', '|df|/|df0|', 'alphaLS', 'iterLS', 'flagLS')
        self.frmt = ('{:<15d}{:<15.2e}{:<15.2e}{:<15.2e}{:<15.2e}{:<15d}{:<15d}')

    def solve(self, net, loss, x, y, verbose=False, log_interval=1):

        # initial evaluation
        f, df_dx, _ = net(x, do_gradient=True, do_Hessian=False)
        df_dx_nrm0 = torch.norm(df_dx)

        info = {'headers': self.headers,
                'frmt': self.frmt,
                'values': [[-1, f.item(), df_dx_nrm0.item(), 1.0, 0.0, 0, 0]],
                'x': torch.clone(x).reshape(1, -1).detach(),
                'df': torch.clone(df_dx).reshape(1, -1).detach()}
        if verbose:
            print((len(self.headers) * '{:<15s}').format(*self.headers))
            print(self.frmt.format(*info['values'][-1]))

        i = 0
        while i < self.max_iter:
            x, infoLS = self.linesearch.step(net, x, -df_dx.view(x.shape), f, df_dx.view(x.shape))

            f, df_dx, _ = net(x, do_gradient=True, do_Hessian=False)
            df_dx_nrm = torch.norm(df_dx)

            if self.stopping_criteria(df_dx_nrm, df_dx_nrm0):
                print('stopping criteria satisfied!')
                break

            if infoLS['flagLS'] < 0:
                print('linesearch break')
                break

            info['values'].append([i, f.item(), df_dx_nrm.item(), (df_dx_nrm / df_dx_nrm0).item()] + list(infoLS.values()))
            info['x'] = torch.cat((info['x'], torch.clone(x.reshape(1, -1).detach())), dim=0)
            info['df'] = torch.cat((info['df'], torch.clone(df_dx.reshape(1, -1).detach())), dim=0)

            if verbose and (i % log_interval == 0):
                print(self.frmt.format(*info['values'][-1]))

            i += 1

        return x, info

    def stopping_criteria(self, df_dx_nrm, df_dx0_nrm):
        flag_abs, flag_rel = False, False

        if df_dx_nrm < self.abs_tol:
            flag_abs = True

        if df_dx_nrm / df_dx0_nrm < self.rel_tol:
            flag_rel = True

        flag = (flag_abs and flag_rel)
        return flag


if __name__ == "__main__":
    import torch
    from hessQuik.networks import NN
    from hessQuik.layers import quadraticLayer
    import matplotlib.pyplot as plt

    torch.manual_seed(123)
    torch.set_default_dtype(torch.float64)

    # setup quadratic problem 0.5 * x.T @ A @ x + b.T @ x
    m, n = 3, 2
    A = torch.cat((torch.randn(m, n), torch.eye(n)), dim=0)
    AtA = A.T @ A
    b = torch.randn(n, 1)

    # solution
    x_true = torch.linalg.solve(AtA, -b)

    fctn = lambda x: 0.5 * x.T @ (AtA @ x) + b.T @ x
    f_true = fctn(x_true)
    print(f_true)
    x0 = 10 * torch.randn_like(x_true)

    # setup network
    layer = quadraticLayer(m, n)
    layer.A = torch.nn.Parameter(A)
    layer.v = torch.nn.Parameter(b.reshape(-1))
    layer.mu = torch.nn.Parameter(torch.zeros(1))

    # check evaluation
    d, dd, _ = layer(x0.reshape(1, -1), do_gradient=True)
    print(layer(x0.reshape(1, -1))[0].item(), fctn(x0).item())

    # create plot

    loss = lambda x, y: x

    opt = GradientDescent(max_iter=50)
    x, info = opt.solve(layer, loss, x0.reshape(1, -1), torch.zeros_like(x0), verbose=True, log_interval=1)
    print((torch.norm(x.view(-1) - x_true.view(-1)) / torch.norm(x_true)).item())

    xx, yy = torch.meshgrid(torch.linspace(-6, 6, 100), torch.linspace(-3, 3, 100), indexing='ij')
    zz = layer(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1))[0]

    plt.figure()
    plt.contourf(xx, yy, zz.detach().reshape(xx.shape))
    plt.plot(x_true[0], x_true[1], 'y*', markersize=20)
    plt.plot(info['x'][:, 0], info['x'][:, 1], 'k-o', markersize=8)
    plt.colorbar()
    plt.show()


