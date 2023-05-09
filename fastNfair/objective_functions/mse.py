import torch
from fastNfair.objective_functions import ObjectiveFunction


class ObjectiveFunctionMSE(ObjectiveFunction):

    def __init__(self, net, regularizer=None, alpha=0.0):
        super(ObjectiveFunctionMSE, self).__init__(net, regularizer, alpha)

    def forward(self, x, y, do_gradient=False, do_Hessian=False):
        n = x.shape[0]

        f, df, d2f = self.net(x, do_gradient=do_gradient, do_Hessian=do_Hessian)

        beta = 1.0 / n
        res = f - y
        loss = (0.5 * beta) * torch.norm(res) ** 2

        dloss, d2loss = None, None
        if do_gradient:
            dloss = df @ (beta * res.unsqueeze(-1))

        if do_Hessian:
            d2loss = beta * (d2f @ res.unsqueeze(1).unsqueeze(-1)
                             + (df.unsqueeze(1) @ df.unsqueeze(1).permute(0, 1, 3, 2)).permute(0, 2, 3, 1))
        return loss, dloss, d2loss, {}


if __name__ == '__main__':
    import torch
    from hessQuik.networks import fullyConnectedNN
    from fastNfair.utils import objective_function_derivative_check

    torch.manual_seed(123)
    torch.set_default_dtype(torch.float64)

    print('=========== MEAN SQUARED ERROR ===========')
    x = torch.randn(10, 2)
    y = torch.randn(10, 5)

    net = fullyConnectedNN([2, 3, 5])
    fctn = ObjectiveFunctionMSE(net)

    f, df, d2f, info = fctn(x, y, do_gradient=True, do_Hessian=True)

    objective_function_derivative_check(fctn, x, y, verbose=True)

    # print('=========== REGULARIZER ===========')
    # from fastNfair.regularizers import RegularizerTikhonov
    #
    # x = torch.randn(10, 2)
    # y = torch.randn(10, 5)
    #
    # net = fullyConnectedNN([2, 3, 5])
    # fctn_no_reg = ObjectiveFunctionMSE(net)
    #
    # out_no_reg = fctn(x, y)[0]
    #
    # reg = RegularizerTikhonov()
    #
    # fctn_reg_0 = ObjectiveFunctionMSE(net, reg, alpha=0.0)
    # out_reg_0 = fctn_reg_0(x, y)[0]
    #
    # fctn_reg = ObjectiveFunctionMSE(net, reg, alpha=1e-2)
    # out_reg = fctn_reg(x, y)[0]
    #
    # print(out_no_reg, out_reg_0, out_reg)
    #
