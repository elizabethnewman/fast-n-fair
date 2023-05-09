import torch.nn as nn


class ObjectiveFunction(nn.Module):

    def __init__(self, net, regularizer=None, alpha=0.0):
        super(ObjectiveFunction, self).__init__()
        # net must be a hessQuik network
        self.net = net
        self.regularizer = regularizer
        self.alpha = alpha

    def forward(self, x, y, do_gradient=False, do_Hessian=False):
        raise NotImplementedError

