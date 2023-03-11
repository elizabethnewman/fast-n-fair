

class LineSearch:

    def __init__(self, alpha: float = 1.0,
                 abs_tol: float = 1e-8,
                 rel_tol: float = 1e-8):
        self.alpha = alpha
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.header = ('alphaLS', 'iterLS', 'flagLS')
        self.frmt = '{:<15e}{:<15d}{:<15d}'

    def initialize_info(self):
        return {'header': self.header,
                'frmt': self.frmt,
                'values': len(self.header) * [0.0]}

    def step(self, f, x0, s, *args):
        raise NotImplementedError
