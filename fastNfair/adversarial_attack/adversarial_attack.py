import torch


class AdversarialAttack(torch.nn.Module):

    def __init__(self):
        super(AdversarialAttack, self).__init__()

    def forward(self, x):
        raise NotImplementedError
