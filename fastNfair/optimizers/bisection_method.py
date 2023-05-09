
import torch
from copy import deepcopy


class BisectionMethod:

    def __init__(self, max_iter=10, tol=1e-8):
        super(BisectionMethod, self).__init__()
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, g, a, b):
        ga, gb, x = g(a), g(b), deepcopy(a)

        if ga * gb > 0:
            if torch.abs(ga) < 1e-8:
                return a
            elif torch.abs(gb) < 1e-8:
                return b
            else:
                raise ValueError('BisectionMethod: g(a) and g(b) must be different signs')

        if torch.sign(ga) != 0 and torch.sign(gb) != 0:
            for n in range(self.max_iter):
                c = (a + b) / 2.0
                x = deepcopy(c)
                gc = g(c)

                if torch.sign(gc) == 0:
                    x = deepcopy(c)
                    break

                if torch.sign(ga) != torch.sign(gc):
                    b = deepcopy(c)
                    # gb = deepcopy(gc)
                else:
                    a = deepcopy(c)
                    ga = deepcopy(gc)

                if abs(b - a) <= self.tol * max(abs(a), abs(b)):
                    break
        else:
            if torch.sign(ga) == 0:
                x = deepcopy(a)
            else:
                x = deepcopy(b)

        return x


if __name__ == "__main__":

    g = lambda x: (x - 3) * (x - 5)

    opt = BisectionMethod(max_iter=100)
    x = opt.solve(g, torch.tensor(4), torch.tensor(6))
    print(x)

    x = opt.solve(g, torch.tensor(0), torch.tensor(4.5))
    print(x)


