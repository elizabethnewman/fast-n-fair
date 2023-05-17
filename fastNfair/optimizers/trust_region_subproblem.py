import torch
from fastNfair.optimizers import BisectionMethod
from copy import deepcopy


class TrustRegionSubproblem:
    # TODO: add documentation

    def __init__(self, max_iter=100, tol=1e-8, per_sample=False):
        self.opt = BisectionMethod(max_iter, tol)
        self.per_sample = per_sample

    def solve(self, fctn, x, delta=None):
        with torch.no_grad():
            fc, dfc, d2fc = fctn(x, do_gradient=True, do_Hessian=True)[:3]

            s = torch.linalg.solve(d2fc.squeeze(-1), -dfc.squeeze(-1))
            # s = s.unsqueeze(-1)
            # s = s.permute(0, 2, 1)

            if delta is None or delta == 0:
                delta = torch.norm(s)

            # per sample way
            if self.per_sample:
                for i in range(x.shape[0]):

                    if s[i].norm() > delta:
                        d, v = torch.linalg.eig(d2fc[i].squeeze(-1))
                        d, v = torch.real(d), torch.real(v)

                        s_a = lambda alpha: (-v.T @ dfc[i].squeeze()) / (d + alpha)
                        g = lambda alpha: s_a(alpha).norm() - delta
                        alpha_high = dfc[i].norm() / delta

                        alpha_low = 0
                        if not g(alpha_low) > 0 or g(alpha_low) == torch.inf:
                            alpha_low = 1e-16

                        if not g(alpha_high) < 0:
                            alpha_high = max(1e3, 1e3 * alpha_high)

                        if g(alpha_low) * g(alpha_high) > 0:
                            s[i] = delta * (s[i] / s[i].norm())
                        else:
                            alpha = self.opt.solve(g, alpha_low, alpha_high)
                            s[i] = s_a(alpha).squeeze()
            else:
                # global way
                if s.norm() > delta:
                    # solve the regularized problem for Lagrange multiplier
                    d, v = torch.linalg.eig(d2fc.squeeze(-1))
                    d, v = torch.real(d), torch.real(v)

                    s_a = lambda alpha: (-v.permute(0, 2, 1) @ dfc) / (d + alpha).unsqueeze(-1)
                    g = lambda alpha: s_a(alpha).norm() - delta
                    alpha_high = dfc.norm() / delta

                    alpha_low = 0
                    if not g(alpha_low) > 0 or g(alpha_low) == torch.inf:
                        alpha_low = 1e-16

                    if not g(alpha_high) < 0:
                        alpha_high = max(1e3, 1e3 * alpha_high)

                    alpha = self.opt.solve(g, alpha_low, alpha_high)
                    s = (v @ s_a(alpha)).squeeze()

            xt = x + s.reshape(x.shape)

        return xt, {'s': deepcopy(s), 'delta': delta}


if __name__ == "__main__":
    import torch.nn as nn
    import hessQuik.networks as net
    import hessQuik.layers as lay
    import hessQuik.activations as act
    from fastNfair.objective_function import ObjectiveFunctionMSE
    import matplotlib.pyplot as plt

    class TestFunctionTrustRegion(nn.Module):
        def __init__(self, fctn, y):
            super(TestFunctionTrustRegion, self).__init__()
            self.fctn = fctn
            self.y = y

        def forward(self, x, do_gradient=False, do_Hessian=False):

            loss, dloss, d2loss, info = self.fctn(x, self.y, do_gradient=do_gradient, do_Hessian=do_Hessian)

            return loss, dloss, d2loss, info

    x = torch.tensor([[0.5, 1.0], [0.0, 0.0], [-1.0, -1.0]])
    y = torch.tensor([[1.0, -2.0], [1.0, 1.0], [-2.0, 0.5]])

    my_net = lay.singleLayer(2, 2, act.identityActivation(), bias=False)
    my_net.K = nn.Parameter(torch.eye(2))

    # def fctn_special(x, **kwargs):
    #     f = 10 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2
    #     df1 = -40 * (x[:, 1] - x[:, 0] ** 2) * x[:, 0] - 2 * (1 - x[:, 0])
    #     df2 = 20 * (x[:, 1] - x[:, 0] ** 2)
    #     df = torch.cat((df1.reshape(-1, 1), df2.reshape(-1, 1)), dim=1)
    #
    #     d2f11 = -40 * (x[:, 1] - x[:, 0] ** 2) + 80 * (x[:, 0] ** 2) + 2
    #     d2f12 = -40 * x[:, 0]
    #     d2f22 = 20 + 0 * x[:, 1]
    #     d2f1 = torch.cat((d2f11.reshape(-1, 1), d2f12.reshape(-1, 1)), dim=1)
    #     d2f2 = torch.cat((d2f12.reshape(-1, 1), d2f22.reshape(-1, 1)), dim=1)
    #     d2f = torch.cat((d2f1, d2f2), dim=1).reshape(-1, 2, 2)
    #
    #     return f.unsqueeze(-1), df.unsqueeze(-1), d2f.unsqueeze(-1)
    #
    # test_fctn = lambda x, **kwargs: fctn_special(x, **kwargs)

    fctn = ObjectiveFunctionMSE(my_net)

    test_fctn = TestFunctionTrustRegion(fctn, y)
    tmp = test_fctn(x)

    opt = TrustRegionSubproblem(per_sample=True)

    delta = 1.5
    xt, info = opt.solve(test_fctn, x, delta=delta)

    theta = torch.linspace(0, 2 * torch.pi, 100)

    for i in range(x.shape[0]):
        plt.plot(x[i, 0] + delta * torch.cos(theta), x[i, 1] + delta * torch.sin(theta), 'k')

    plt.plot(x[:, 0], x[:, 1], 'kx', markersize=10, label='x^{(0)}')
    plt.plot(y[:, 0], y[:, 1], 'bs', markersize=10, label='y')
    plt.plot(xt[:, 0], xt[:, 1], 'ro', markersize=10, label='xt')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('square')
    plt.show()

    # xx, yy = torch.meshgrid(torch.linspace(-6, 6, 100), torch.linspace(-3, 3, 100), indexing='ij')
    # zz = test_fctn(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1))[0]
    # plt.figure()
    # plt.contourf(xx, yy, zz.detach().reshape(xx.shape))
    # plt.plot(1.0, 1.0, 'y*', markersize=20)
    # plt.plot(x[:, 0], x[:, 1], 'kx', markersize=10, label='x^{(0)}')
    # plt.plot(xt[:, 0], xt[:, 1], 'ro', markersize=10, label='xt')
    #
    # # plt.plot(info['x'][:, 0], info['x'][:, 1], 'k-o', markersize=8)
    # plt.colorbar()
    # plt.show()

