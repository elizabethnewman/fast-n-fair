import torch
from fastNfair.adversarial_attack import AdversarialAttack


class FastSignGradient(AdversarialAttack):

    def __init__(self, epsilon=1e-4):
        super(FastSignGradient, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return self.epsilon * torch.sign(torch.randn_like(x))



