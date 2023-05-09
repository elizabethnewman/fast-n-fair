
import torch
from fastNfair.regularizers import Regularizer


class RegularizerInvariantRisk(Regularizer):

    def __init__(self, alpha=1e0):
        super(RegularizerInvariantRisk, self).__init__(alpha=alpha)

    def forward(self, fctn=None, x=None, y=None, s=None, do_gradient=False, do_Hessian=False):

        dreg, d2reg = None, None

        dloss = fctn(x, y, do_gradient=True)[1]

        return self.alpha * (dloss.norm() ** 2), dreg, d2reg, {}


if __name__ == "__main__":
    import hessQuik.activations as act
    import hessQuik.layers as lay
    import hessQuik.networks as net
    from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
    torch.manual_seed(42)

    my_net = net.NN(lay.singleLayer(2, 1, act=act.tanhActivation(), bias=True))

    fctn = ObjectiveFunctionLogisticRegression(my_net)

    reg = RegularizerInvariantRisk()

    x = torch.randn(11, 2)
    y = 1 * (torch.rand(11) > 0.5)
    out = reg(fctn, x, y)

    print(out[0])





