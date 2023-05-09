import torch.nn as nn
import torch


class Regularizer(nn.Module):

    def __init__(self, alpha=1e0):
        super(Regularizer, self).__init__()
        self.alpha = alpha

    def forward(self, fctn=None, x=None, y=None, s=None, do_gradient=False, do_Hessian=False):
        raise NotImplementedError


class CombinedRegularizer(nn.Module):
    def __init__(self, *args):
        super(CombinedRegularizer, self).__init__()

        self.regularizers = list()
        for a in args:
            self.regularizers.append(a)

    def forward(self, fctn=None, x=None, y=None, s=None, do_gradient=False, do_Hessian=False):
        out = torch.zeros(1, requires_grad=True)
        for reg in self.regularizers:
            out = out + reg(fctn=fctn, x=x, y=y, s=s, do_gradient=do_gradient, do_Hessian=do_Hessian)[0]

        return out


if __name__ == "__main__":
    import hessQuik.activations as act
    import hessQuik.layers as lay
    import hessQuik.networks as net
    from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
    from fastNfair.regularizers import RegularizerTikhonov, RegularizerInvariantRisk, RegularizerSeparation

    torch.manual_seed(42)

    my_net = net.NN(lay.singleLayer(2, 1, act=act.tanhActivation(), bias=True))

    fctn = ObjectiveFunctionLogisticRegression(my_net)

    reg = CombinedRegularizer(RegularizerTikhonov(alpha=1e-4), RegularizerInvariantRisk(alpha=1e0), RegularizerSeparation(alpha=1e0))

    x = torch.randn(11, 2)
    y = 1 * (torch.rand(11) > 0.5).view(-1)
    s = 1 * (torch.rand(11) > 0.5).view(-1)
    out = reg(fctn, x, y, s)

    print(out)
