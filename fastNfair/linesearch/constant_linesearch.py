from fastNfair.linesearch import LineSearch


class ConstantLineSearch(LineSearch):

    def __init__(self, *args, **kwargs):
        super(ConstantLineSearch, self).__init__(*args, **kwargs)

    def step(self, f, x0, s, *args):
        info = {'alphaLS': self.alpha, 'iterLS': 0, 'flagLS': 1}
        return x0 + self.alpha * s.view(x0.shape), info

