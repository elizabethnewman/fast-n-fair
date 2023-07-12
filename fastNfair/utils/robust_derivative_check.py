import torch
from math import log2, floor
import hessQuik
from hessQuik.utils import convert_to_base, extract_data, insert_data
from typing import Callable, Tuple, Optional, Union
from fastNfair.optimizers import TrustRegionSubproblem
from fastNfair.optimizers import ProjectedGradientDescentv1
from fastNfair.optimizers.tmp_optimizer import TrustRegionSubproblem
from fastNfair.optimizers.PGDhessquik import ProjectedGradientDescentv3
from fastNfair.training.adversarial_training import ObjectiveFunctionMaximize

def solve_inner_problem(sol_type, fctn, x,y,radius):
    # evaluate inner problem
    with torch.no_grad():
        if sol_type == 'PGD':
            opt = ProjectedGradientDescentv3(max_iter=1)
            fctn_max = ObjectiveFunctionMaximize(fctn, y)
            xt = opt.solve(fctn_max, x, 1, radius)
        elif sol_type == 'TRS':
            opt = TrustRegionSubproblem(max_iter=10000, per_sample=True)
            fctn_max = ObjectiveFunctionMaximize(fctn, y)
            xt, info = opt.solve(fctn_max, x, radius)

    # compute loss
    loss, _, _, info = fctn(xt,y)
    return loss

def robust_network_derivative_check(fctn, x, y, sol_type='TRS', radius=1e-2,
                                    num_test: int = 15, base: float = 2.0, tol: float = 0.1,
                                    verbose: bool = False) -> Optional[bool]:
    fctn.zero_grad()

    # initial solution of inner optimization problem
    loss = solve_inner_problem(sol_type,fctn,x,y,radius)
    loss.backward()

    # extract initial values
    loss0 = loss.detach()
    theta0 = extract_data(fctn, 'data')
    grad_theta0 = extract_data(fctn, 'grad')
    # print(grad_theta0.norm())

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
    for k in range(num_test):
        # update perturbation scale
        h = base ** (-k)

        insert_data(fctn, theta0 + h * dtheta)

        # solve inner optimization problem for perturbation
        losst = solve_inner_problem(sol_type,fctn,x,y,radius)

        # compute 0th and 1st order errors
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
    import hessQuik.networks as net
    import hessQuik.layers as lay
    import hessQuik.activations as act
    from fastNfair.utils import objective_function_derivative_check
    from fastNfair.objective_functions import ObjectiveFunctionMSE

    torch.manual_seed(123)
    torch.set_default_dtype(torch.float64)

    # sanity check - does this work for a quadratic function
    n = 3
    x = torch.randn(n, 2)
    y = torch.randn(n, 3)

    # my_net = lay.singleLayer(x.shape[1], y.shape[1], bias=True)
    # my_net = net.NN(lay.singleLayer(x.shape[1], 7, act=act.softplusActivation()),
    #                 lay.singleLayer(7, y.shape[1], act=act.identityActivation()))

    width = 7
    my_net = net.NN(lay.singleLayer(x.shape[1], width, act=act.tanhActivation()),
                    net.resnetNN(width, 4, act=act.softplusActivation()),
                    net.fullyConnectedNN([width, 13, 5], act=act.quadraticActivation()),
                    lay.singleLayer(5, y.shape[1], act=act.identityActivation())
                    )

    fctn = ObjectiveFunctionMSE(my_net)
    theta0 = extract_data(fctn, 'data')

    # sol_type: 'PDG' for projected gradient descent method, 'TRS' for trust region
    # (default is 'TRS' if left unspecified)
    robust_network_derivative_check(fctn, x, y, sol_type='PGD', verbose=True, radius=1e3)

    insert_data(fctn, theta0)
    robust_network_derivative_check(fctn, x, y, sol_type='PGD', verbose=True, radius=1e-1)

