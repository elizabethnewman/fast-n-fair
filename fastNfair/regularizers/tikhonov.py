import torch
from fastNfair.regularizers import Regularizer


class RegularizerTikhonov(Regularizer):

    def __init__(self, alpha=1e0):
        super(RegularizerTikhonov, self).__init__(alpha=alpha)

    def forward(self, fctn=None, x=None, y=None, s=None, do_gradient=False, do_Hessian=False):

        dreg, d2reg = None, None

        reg = torch.zeros(1, requires_grad=True)
        for p in fctn.parameters():
            reg = reg + p.norm() ** 2

        return (0.5 * self.alpha) * reg, dreg, d2reg, {}


if __name__ == "__main__":
    import hessQuik.activations as act
    import hessQuik.layers as lay
    import hessQuik.networks as net
    from fastNfair.objective_functions import ObjectiveFunctionMSE
    torch.manual_seed(42)

    my_net = net.NN(lay.singleLayer(2, 1, act=act.tanhActivation(), bias=True))

    fctn = ObjectiveFunctionMSE(my_net)

    reg = RegularizerTikhonov()
    out = reg(fctn=fctn)

    print(out[0])



