
import torch
import torch.nn as nn


# http://proceedings.mlr.press/v119/lohaus20a/lohaus20a.pdf


class DDP(nn.Module):

    def __init__(self):
        super(DDP, self).__init__()
        # Difference in Demographic Parity
        # we create a convex approximation
        # let's see if we can find a resource for this

    def forward(self, p, s, y=None, do_gradient=False, do_Hessian=False):
        p, s = p.squeeze(), s.squeeze()

        attr_1 = ((p > 0.5) * (s == 1)).sum()
        attr_0 = ((p > 0.5) * (s == 0)).sum()

        beta = 1.0 / p.shape[0]
        reg = beta * (attr_1 - attr_0)

        dreg, d2reg = None, None

        if do_gradient:
            dreg = 0 * beta * (1.0 * (s == 1) - 1.0 * (s == 0))
            dreg = dreg.unsqueeze(-1).unsqueeze(-1)

        if do_Hessian:
            d2reg = torch.zeros_like(p).reshape(-1, 1, 1)
            d2reg = d2reg.unsqueeze(-1)

        return reg, dreg, d2reg, {}


class TikhonovRegularizer(nn.Module):

    def __init__(self):
        super(TikhonovRegularizer, self).__init__()

    def forward(self, p, s, y=None, do_gradient=False, do_Hessian=False):
        beta = 1.0 / p.shape[0]
        reg = (0.5 * beta) * torch.norm(p) ** 2

        dreg, d2reg = None, None

        if do_gradient:
            dreg = beta * p
            dreg = dreg.unsqueeze(-1)

        if do_Hessian:
            d2reg = torch.diag_embed(beta * torch.ones_like(p))
            d2reg = d2reg.unsqueeze(-1)

        return reg, dreg, d2reg, {}


if __name__ == "__main__":
    from fastNfair.utils.derivative_check import objective_function_derivative_check
    torch.set_default_dtype(torch.float64)

    p = torch.rand(10, 1)
    s = torch.randint(0, 2, (p.shape[0],))
    s = s.to(torch.float32)

    reg = DDP()

    reg0, dreg0, d2reg0, info = reg(p, s, do_gradient=True, do_Hessian=True)

    objective_function_derivative_check(reg, p, s, verbose=True, dx=torch.rand_like(p))


    p = torch.randn(10, 1)
    s = torch.randint(0, 2, (p.shape[0],))
    s = s.to(torch.float32)

    reg = TikhonovRegularizer()

    reg0, dreg0, d2reg0, info = reg(p, s, do_gradient=True, do_Hessian=True)

    objective_function_derivative_check(reg, p, s, verbose=True)


