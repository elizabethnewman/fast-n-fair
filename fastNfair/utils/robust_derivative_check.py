import torch
from math import log2, floor
import hessQuik
from hessQuik.utils import convert_to_base, extract_data, insert_data
from typing import Callable, Tuple, Optional, Union
from fastNfair.optimizers import TrustRegionSubproblem
from fastNfair.training.adversarial_training import ObjectiveFunctionMaximize


def robust_network_derivative_check(fctn, x, y, radius=1e-2,
                                    num_test: int = 15, base: float = 2.0, tol: float = 0.1,
                                    verbose: bool = False) -> Optional[bool]:

    # initial evaluation
    with torch.no_grad():
        opt = TrustRegionSubproblem(max_iter=100, per_sample=True)
        fctn_max = ObjectiveFunctionMaximize(fctn, y)
        xt, info = opt.solve(fctn_max, x, radius)

    # compute loss
    loss, _, _, info = fctn(xt, y)
    loss.backward()

    loss0 = loss.detach()
    theta0 = extract_data(fctn, 'data')
    grad_theta0 = extract_data(fctn, 'grad')

    # perturbation
    dtheta = torch.randn_like(theta0)
    dtheta = dtheta / torch.norm(dtheta)

    # directional derivative
    dfdtheta = (grad_theta0 * dtheta).sum()

    # ---------------------------------------------------------------------------------------------------------------- #
    # derivative check
    if verbose:
        headers = ('h', 'E0', 'E1')
        print(('{:<20s}' * len(headers)).format(*headers))

    # with torch.no_grad():
    E0, E1 = [], []
    loss_dft, loss_d2ft = 0.0, 0.0
    for k in range(num_test):
        h = base ** (-k)
        insert_data(fctn, theta0 + h * dtheta)
        with torch.no_grad():
            opt = TrustRegionSubproblem(max_iter=10000, per_sample=True)
            fctn_max = ObjectiveFunctionMaximize(fctn, y)
            xt, info = opt.solve(fctn_max, x, radius)

        # compute loss
        losst, _, _, info = fctn(xt, y)

        E0.append(torch.norm(loss0 - losst.detach()).item())
        E1.append(torch.norm(loss0 + h * dfdtheta - losst.detach()).item())

        printouts = convert_to_base((E0[-1], E1[-1]))

        if verbose:
            print(((1 + len(printouts) // 2) * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))

    E0, E1 = torch.tensor(E0), torch.tensor(E1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # check if order is 2 at least half of the time
    eps = torch.finfo(x.dtype).eps
    grad_check = (sum((torch.log2(E1[:-1] / E1[1:]) / log2(base)) > (2 - tol)) > 3)
    grad_check = (grad_check or (torch.kthvalue(E1, num_test // 3)[0] < (100 * eps)))

    if verbose:
        if grad_check:
            print('Gradient PASSED!')
        else:
            print('Gradient FAILED.')

    return grad_check


if __name__ == "__main__":
    # import torch
    # import hessQuik.networks as net
    # import hessQuik.layers as lay
    # import hessQuik.activations as act
    #
    # torch.set_default_dtype(torch.float64)
    #
    # nex = 11  # no. of examples
    # d = 2  # no. of input features
    #
    # x = torch.randn(nex, d)
    # dx = torch.randn_like(x)
    #
    # # f = net.NN(lay.singleLayer(d, 7, act=act.softplusActivation()),
    # #            lay.singleLayer(7, 5, act=act.identityActivation()))
    #
    # # width = 8
    # # depth = 8
    # # f = net.NN(lay.singleLayer(d, width, act=act.tanhActivation()),
    # #            net.resnetNN(width, depth, h=1.0, act=act.tanhActivation()),
    # #            lay.singleLayer(width, 1, act=act.identityActivation()))
    #
    # # width = 7
    # # f = net.NN(lay.singleLayer(d, width, act=act.tanhActivation()),
    # #            net.resnetNN(width, 4, act=act.softplusActivation()),
    # #            net.fullyConnectedNN([width, 13, 5], act=act.quadraticActivation()),
    # #            lay.singleLayer(5, 3, act=act.identityActivation()),
    # #            lay.quadraticLayer(3, 2)
    # #            )

    import torch
    from hessQuik.networks import fullyConnectedNN
    from fastNfair.utils import objective_function_derivative_check
    from fastNfair.objective_functions import ObjectiveFunctionMSE

    torch.manual_seed(123)
    torch.set_default_dtype(torch.float64)

    print('=========== LOGISTIC REGRESSION ===========')
    x = torch.randn(10, 2)
    # y = torch.rand(10, 1)
    # y /= y.sum(dim=1, keepdim=True)
    y = torch.randn(x.shape[0], 5)
    # y = torch.randint(0, 2, (x.shape[0],))
    # y = y.to(torch.float32)

    net = fullyConnectedNN([2, 5])
    fctn = ObjectiveFunctionMSE(net)

    robust_network_derivative_check(fctn, x, y, verbose=True, radius=1e3)
