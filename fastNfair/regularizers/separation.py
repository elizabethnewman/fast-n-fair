
import torch
from fastNfair.regularizers import Regularizer


class RegularizerSeparation(Regularizer):

    def __init__(self, alpha=1e0, threshold=0.0):
        super(RegularizerSeparation, self).__init__(alpha=alpha)
        self.threshold = threshold

    def forward(self, fctn=None, x=None, y=None, s=None, do_gradient=False, do_Hessian=False):

        dreg, d2reg = None, None

        # y_pred = 1.0 * (fctn.net(x)[0] > self.threshold).view(-1)
        y_pred = torch.sigmoid(fctn.net(x)[0])

        p0 = y_pred[(y == 1) * (s == 0)].sum() / ((y == 1) * (s == 0)).sum()
        p1 = y_pred[(y == 1) * (s == 1)].sum() / ((y == 1) * (s == 1)).sum()
        reg = self.alpha * torch.abs(p0 - p1)

        p0 = y_pred[(y == 0) * (s == 0)].sum() / ((y == 0) * (s == 0)).sum()
        p1 = y_pred[(y == 0) * (s == 1)].sum() / ((y == 0) * (s == 1)).sum()
        reg = reg + self.alpha * torch.abs(p0 - p1)


        return reg, dreg, d2reg, {}


if __name__ == "__main__":
    import hessQuik.activations as act
    import hessQuik.layers as lay
    import hessQuik.networks as net
    from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
    torch.manual_seed(42)

    my_net = net.NN(lay.singleLayer(2, 1, act=act.tanhActivation(), bias=True))

    fctn = ObjectiveFunctionLogisticRegression(my_net)

    reg = RegularizerSeparation()

    x = torch.randn(11, 2)
    y = 1 * (torch.rand(11) > 0.5).view(-1)
    s = 1 * (torch.rand(11) > 0.5).view(-1)
    out = reg(fctn, x, y, s)

    print(out[0])





