import torch
from math import log2, floor, ceil
import hessQuik
from hessQuik.utils import convert_to_base
from typing import Callable, Tuple, Optional, Union
import math


def objective_function_derivative_check(fctn, x, y, num_test: int = 15, base: float = 2.0, tol: float = 0.1,
                           verbose: float = False, dx: torch.Tensor = None) -> Tuple[Optional[bool], Optional[bool]]:

    # initial evaluation
    f0, df0, d2f0, _ = fctn(x, y, do_gradient=True, do_Hessian=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # directional derivatives
    if dx is None:
        dx = torch.randn_like(x)
        dx = dx / torch.norm(x)

    dfdx = (df0.transpose(1, 2) @ dx.unsqueeze(-1)).sum()
    curvx = (dx.unsqueeze(1) @ (d2f0.squeeze(-1) @ dx.unsqueeze(-1))).sum()

    # ---------------------------------------------------------------------------------------------------------------- #
    # derivative check
    E0, E1, E2 = [], [], []

    if verbose:
        headers = ('h', 'E0', 'E1', 'E2')
        print(('{:<20s}' * len(headers)).format(*headers))

    for k in range(num_test):
        h = base ** (-k)
        ft, *_ = fctn(x + h * dx, y, do_gradient=False, do_Hessian=False)

        E0.append(torch.norm(f0 - ft).item())
        E1.append(torch.norm(f0 + h * dfdx - ft).item())
        E2.append(torch.norm(f0 + h * dfdx + 0.5 * (h ** 2) * curvx - ft).item())
        printouts = convert_to_base((E0[-1], E1[-1], E2[-1]))

        if verbose:
            print(((1 + len(printouts) // 2) * '%0.2f x 2^(%0.2d)\t\t') % ((1, -k) + printouts))

    E0, E1, E2 = torch.tensor(E0), torch.tensor(E1), torch.tensor(E2)

    # ---------------------------------------------------------------------------------------------------------------- #
    # check if order is 2 enough of the time
    eps = torch.finfo(x.dtype).eps

    grad_check = (sum((torch.log2(E1[:-1] / E1[1:]) / log2(base)) > (2 - tol)) > num_test // 3)
    grad_check = (grad_check or (torch.kthvalue(E1, num_test // 4)[0] < (100 * eps)))

    hess_check = (sum((torch.log2(E2[:-1] / E2[1:]) / log2(base)) > (3 - tol)) > num_test // 3)
    hess_check = (hess_check or (torch.kthvalue(E2, num_test // 4)[0] < (100 * eps)))

    if verbose:
        if grad_check:
            print('Gradient PASSED!')
        else:
            print('Gradient FAILED.')

        if hess_check:
            print('Hessian PASSED!')
        else:
            print('Hessian FAILED.')

    return grad_check, hess_check


