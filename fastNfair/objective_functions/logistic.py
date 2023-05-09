import torch
from fastNfair.objective_functions import ObjectiveFunction


class ObjectiveFunctionLogisticRegression(ObjectiveFunction):

    def __init__(self, net):
        super(ObjectiveFunctionLogisticRegression, self).__init__(net)

    def forward(self, x, y, do_gradient=False, do_Hessian=False):
        n = x.shape[0]

        f, df, d2f = self.net(x, do_gradient=do_gradient, do_Hessian=do_Hessian)

        y = y.view(f.shape)
        info = {'num_correct': torch.sum((f > 0.5) == (y > 0.5)).item()}

        beta = 1.0 / n
        p = torch.sigmoid(f)
        # loss = -beta * (y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).sum()

        # loss = beta * torch.nn.functional.nll_loss(p, y, reduction='sum')
        # # more stable implementation
        m = f.max()
        tmp = torch.log(torch.exp(f - m) + torch.exp(-m)) + m
        loss = -beta * (y * (f - tmp) - (1.0 - y) * tmp).sum()
        #
        # even more stable implementation
        # m = f.max()
        # tmp = torch.log(torch.exp(f - m) + torch.exp(-m)) + m
        # loss = -beta * ((f[y == 1] - tmp[y == 1]).sum() - tmp[y == 0].sum())

        if torch.isnan(loss):
            print('here')
        dloss, d2loss = None, None
        if do_gradient:
            res = -y + p
            dloss = df @ (beta * res.unsqueeze(-1))

            if do_Hessian:
                h = torch.diag_embed(p * (1 - p))
                d2loss = beta * (d2f @ res.unsqueeze(1).unsqueeze(-1)
                                 + (df.unsqueeze(1)
                                    @ (h.unsqueeze(1)
                                       @ df.permute(0, 2, 1).unsqueeze(1))).permute(0, 2, 3, 1))

        return loss, dloss, d2loss, info



if __name__ == '__main__':
    import torch
    from hessQuik.networks import fullyConnectedNN
    from fastNfair.utils import objective_function_derivative_check

    torch.manual_seed(123)
    torch.set_default_dtype(torch.float64)

    print('=========== LOGISTIC REGRESSION ===========')
    x = torch.randn(10, 2)
    # y = torch.rand(10, 1)
    # y /= y.sum(dim=1, keepdim=True)
    y = torch.randint(0, 2, (x.shape[0],))
    y = y.to(torch.float32)

    net = fullyConnectedNN([2, 3, 1])
    fctn = ObjectiveFunctionLogisticRegression(net)

    f, df, d2f, info = fctn(x, y, do_gradient=True, do_Hessian=True)

    objective_function_derivative_check(fctn, x, y, verbose=True)
