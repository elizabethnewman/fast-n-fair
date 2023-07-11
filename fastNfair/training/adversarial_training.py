import torch
from copy import deepcopy
# from fastNfair.optimizers.tmp_optimizer import TrustRegionSubproblem
from fastNfair.optimizers.tmp_optimizer import TrustRegionSubproblem
from fastNfair.optimizers.PGDhessquik import ProjectedGradientDescentv3
from fastNfair.optimizers.rand_perturbation import RandomPerturbation

class ObjectiveFunctionMaximize(torch.nn.Module):
    def __init__(self, fctn, y):
        super(ObjectiveFunctionMaximize, self).__init__()
        self.fctn = fctn
        self.y = y

    def forward(self, x, do_gradient=False, do_Hessian=False):

        f, df, d2f, *_ = self.fctn(x, self.y, do_gradient=do_gradient, do_Hessian=do_Hessian)

        # negate so we maximize
        f *= -1.0
        if df is not None:
            df *= -1.0
        if d2f is not None:
            d2f *= -1.0
        return f, df, d2f


def loss_landscape(fctn_perturb, delta1, delta2, n=20, a=-1, b=1):
    with torch.no_grad():
        tmp = torch.zeros(n, n)
        for i, alpha in enumerate(torch.linspace(a, b, n)):
            for j, beta in enumerate(torch.linspace(a, b, n)):
                tmp[i, j] = fctn_perturb(alpha * delta1 + beta * delta2)[0]

    return tmp


def predict_labels(out):
    return out.argmax(dim=-1)


def train_one_epoch(fctn, optimizer, x, y, s, robustOptimizer='trust', regularizer=None, batch_size=32, robust=True, radius=2e-1, device='cpu'):
    fctn.train()
    n = x.shape[0]
    b = batch_size
    n_batch = n // b

    running_loss = 0.0
    running_acc = 0

    # shuffle
    idx = torch.randperm(n)

    count = 0
    for i in range(n_batch):
        idxb = idx[i * b:(i + 1) * b]
        xb, yb, sb = x[idxb].to(device), y[idxb].to(device), s[idxb].to(device)

        # find perturbation
        if not robust or radius < 1e-10:
            optimizer.zero_grad()
            loss, _, _, info = fctn(xb, yb)
        else:
            with torch.no_grad():
                fctn_max = ObjectiveFunctionMaximize(fctn, yb)
                if robustOptimizer == 'trust':
                    opt = TrustRegionSubproblem(max_iter=100, per_sample=True)
                    xt, info = opt.solve(fctn_max, xb, radius)
                elif robustOptimizer == 'pgd':
                    opt = ProjectedGradientDescentv3(max_iter = 5000, per_sample = True)
                    xt = opt.solve(fctn_max, xb, 0.1, radius)
                elif robustOptimizer == 'rand':
                    opt = RandomPerturbation(per_sample = True)
                    xt = opt.solve(xb, radius)

            optimizer.zero_grad()
            loss, _, _, info = fctn(xt, yb)

        # add fairness regularization
        if regularizer is not None:
            reg = regularizer(fctn, xb, yb, sb)[0]
            loss = loss + reg

        # update network weights
        loss.backward()
        optimizer.step()

        # store results
        running_loss += b * loss.item()
        running_acc += info['num_correct']
        count += b

    output = (running_loss / count, 100 * running_acc / count)

    return output


def test(fctn, x, y, device='cpu', batch_size=10):
    fctn.eval()

    with torch.no_grad():
        n = x.shape[0]
        b = batch_size
        n_batch = n // b

        count, running_loss, running_acc = 0, 0.0, 0.0
        idx = torch.arange(n_batch * batch_size)
        for i in range(n_batch):
            idxb = idx[i * b:(i + 1) * b]
            xb, yb = x[idxb].to(device), y[idxb].to(device)
            loss, _, _, info = fctn(xb, yb)

            # store results
            running_loss += b * loss.item()
            running_acc += info['num_correct']
            count += b

    return running_loss / count, 100 * running_acc / count

