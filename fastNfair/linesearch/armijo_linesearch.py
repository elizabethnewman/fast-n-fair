import torch
from fastNfair.linesearch import LineSearch


class ArmijoLineSearch(LineSearch):

    def __init__(self, *args,
                 max_iters: int = 40,
                 beta: float = 0.5,
                 gamma: float = 1e-3,
                 c: float = 1e-4,
                 **kwargs):
        super(ArmijoLineSearch, self).__init__(*args, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.c = c
        self.max_iters = max_iters

    def step(self, f, x0, s, *args):
        f0, g0, x, flag = args[0], args[1], x0, -1


        tau = torch.dot(g0.view(-1), s.view(-1))

        mu = self.alpha

        if tau > 0:
            print('here')

        i = 0
        while i < self.max_iters:
            x = x0 + mu * s
            ft = f(x)[0]

            if ft < f0 - mu * self.gamma * tau:
                flag = 1
                break

            mu *= self.beta
            i += 1

        if i == 0:
            self.alpha *= 1.05
        else:
            self.alpha = mu

        info = {'alphaLS': mu, 'iterLS': i, 'flagLS': flag}
        return x, info

