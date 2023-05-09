import torch
from fastNfair.objective_functions import ObjectiveFunction


class ObjectiveFunctionCrossEntropy(ObjectiveFunction):

    def __init__(self, net):
        super(ObjectiveFunctionCrossEntropy, self).__init__(net)

    def forward(self, x, y, do_gradient=False, do_Hessian=False):
        n = x.shape[0]

        f, df, d2f = self.net(x, do_gradient=do_gradient, do_Hessian=do_Hessian)

        info = {'num_correct': torch.sum(1 * f.argmax(dim=-1) == y).item()}

        c = torch.zeros_like(f)
        c[torch.arange(n), y] = 1.0

        beta = 1.0 / n
        p = torch.softmax(f, dim=1)
        loss = -beta * (c * torch.log(p)).sum()

        dloss, d2loss = None, None
        if do_gradient:
            res = -c + p
            dloss = df @ (beta * res.unsqueeze(-1))

            if do_Hessian:
                h = torch.diag_embed(p) - p.unsqueeze(-1) @ p.unsqueeze(1)
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

    print('=========== CROSS ENTROPY ===========')
    x = torch.randn(10, 2)
    y = torch.randint(0, 5, (x.shape[0],))

    net = fullyConnectedNN([2, 3, 5])
    fctn = ObjectiveFunctionCrossEntropy(net)

    f, df, d2f, info = fctn(x, y, do_gradient=True, do_Hessian=True)

    objective_function_derivative_check(fctn, x, y, verbose=True)



