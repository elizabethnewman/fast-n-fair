import torch
from copy import deepcopy
from fastNfair.optimizers import TrustRegionSubproblem


class TrustRegionNewton:

    def __init__(self,
                 max_iter: int = 10,
                 abs_tol: float = 1e-3,
                 rel_tol: float = 1e-3,
                 delta_tol: float = 1e-5,
                 delta: float = 0.0,
                 delta_max: float = 10.0,
                 eta: float = 0.3,
                 rho0: float = 1e-4,
                 rhoL: float = 0.1,
                 rhoH: float = 0.75,
                 wdown: float = 0.5,
                 wup: float = 1.5,
                 C: float = 1e-4):

        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.delta_tol = delta_tol
        self.delta = delta
        self.delta_max = delta_max
        self.eta = eta
        self.rho0 = rho0
        self.rhoL = rhoL
        self.rhoH = rhoH
        self.wdown = wdown
        self.wup = wup
        self.C = C
        self.sub_opt = TrustRegionSubproblem(per_sample=False)

        self.headers = ('iter', 'f', '|x-x_old|', '|df|', '|df|/|df0|', '|s|', 'delta', 'flagTR', 'ared / pred')
        self.frmt = ('{:<15d}{:<15.2e}{:<15.2e}{:<15.2e}{:<15.2e}{:<15.2e}{:<15.2e}{:<15d}{:<15.2e}')

    def solve(self, fctn, x, verbose=False, log_interval=1):

        # initial evaluation
        f, df_dx, d2f_dx2 = fctn(x, do_gradient=True, do_Hessian=True)[:3]
        df_dx_nrm0 = torch.norm(df_dx)

        info = {'headers': self.headers,
                'frmt': self.frmt,
                'values': [[-1, f.item(), 0.0, df_dx_nrm0.item(), 1.0, 0.0, self.delta, 0, 0.0]],
                'x': torch.clone(x).reshape(1, -1).detach(),
                'df': torch.clone(df_dx).reshape(1, -1).detach()}

        if verbose:
            print((len(self.headers) * '{:<15s}').format(*self.headers))
            print(self.frmt.format(*info['values'][-1]))

        i = 0
        while i < self.max_iter:

            # compute search direction
            xt, infoTR = self.sub_opt.solve(fctn, x, self.delta)
            self.delta = infoTR['delta']

            x, infoTRupdate = self.trupdate(x, xt, fctn)

            # re-evaluate
            f, df_dx, d2f_dx2 = fctn(x, do_gradient=True, do_Hessian=True)[:3]
            df_dx_nrm = torch.norm(df_dx)

            # store progress
            info['values'].append([i, f.item(), 0.0, df_dx_nrm.item(),
                                   (df_dx_nrm / df_dx_nrm0).item()] + [torch.norm(infoTR['s']).item(),
                                                                       self.delta, infoTRupdate['flag'],
                                                                       infoTRupdate['rho']])
            info['x'] = torch.cat((info['x'], torch.clone(x.reshape(1, -1).detach())), dim=0)
            info['df'] = torch.cat((info['df'], torch.clone(df_dx.reshape(1, -1).detach())), dim=0)

            if verbose and (i % log_interval == 0):
                print(self.frmt.format(*info['values'][-1]))

            if self.stopping_criteria(df_dx_nrm, df_dx_nrm0):
                # print('stopping criteria satisfied!')
                break

            i += 1

        return x, info

    def trupdate(self, xc, xt, fctn):

        flag = 0
        fc, dfc, d2fc = fctn(xc, do_gradient=True, do_Hessian=True)[:3]
        ft = fctn(xt)[0]

        # step (xt = xc + s)
        st = xt - xc
        st = st.reshape(dfc.shape)

        # actual reduction
        ared = fc - ft

        # predicted reduction
        # pred = -torch.sum(dfc * st).squeeze() - 0.5 * (st.transpose(-1, -2) @ (d2fc.squeeze(-1) @ st)).squeeze()
        # pred = pred.mean()  # TODO: check this
        # pred2 = -torch.dot(dfc.view(-1), st.view(-1)) - 0.5 * (st.transpose(-1, -2) @ (d2fc.squeeze(-1) @ st)).squeeze()
        pred = -(dfc * st).sum() - 0.5 * (st.transpose(-1, -2) @ (d2fc.squeeze(-1) @ st)).sum()

        # ratio of actual to predicted
        rho = ared / pred

        if rho < 0.25:
            self.delta *= 0.25
            flag = -1
        else:
            if rho > 0.75 and (torch.norm(st) - self.delta) / self.delta < 1e-5:
                self.delta = min(2 * self.delta, self.delta_max)
                flag = 2

        if rho > self.eta:
            # z = xc + st.squeeze(-1)
            z = xt
            flag = 3
        else:
            z = xc

        info = {'flag': flag, 'rho': rho.item()}
        return z, info

    def stopping_criteria(self, df_dx_nrm, df_dx0_nrm):
        flag_abs, flag_rel = False, False

        if df_dx_nrm < self.abs_tol:
            flag_abs = True

        if df_dx_nrm / df_dx0_nrm < self.rel_tol:
            flag_rel = True

        flag = (flag_abs and flag_rel)
        return flag


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from hessQuik.utils import input_derivative_check

    torch.manual_seed(123)
    torch.set_default_dtype(torch.float64)

    def fctn(x, **kwargs):
        f = 10 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2
        df1 = -40 * (x[:, 1] - x[:, 0] ** 2) * x[:, 0] - 2 * (1 - x[:, 0])
        df2 = 20 * (x[:, 1] - x[:, 0] ** 2)
        df = torch.cat((df1.reshape(-1, 1), df2.reshape(-1, 1)), dim=1)

        d2f11 = -40 * (x[:, 1] - x[:, 0] ** 2) + 80 * (x[:, 0] ** 2) + 2
        d2f12 = -40 * x[:, 0]
        d2f22 = 20 + 0 * x[:, 1]
        d2f1 = torch.cat((d2f11.reshape(-1, 1), d2f12.reshape(-1, 1)), dim=1)
        d2f2 = torch.cat((d2f12.reshape(-1, 1), d2f22.reshape(-1, 1)), dim=1)
        d2f = torch.cat((d2f1, d2f2), dim=1).reshape(-1, 2, 2)

        return f.unsqueeze(-1), df.unsqueeze(-1), d2f.unsqueeze(-1)

    x0 = torch.randn(1, 2)
    f0, df0, d2f0 = fctn(x0)
    input_derivative_check(fctn, x0, verbose=True)

    x_true = torch.ones_like(x0)
    f_true = fctn(x_true)[0]
    print(f_true.squeeze())

    x0 = torch.randn_like(x_true)

    # create plot
    opt = TrustRegionNewton(max_iter=50)
    x, info = opt.solve(fctn, x0.reshape(1, -1), verbose=True, log_interval=1)
    print((torch.norm(x.view(-1) - x_true.view(-1)) / torch.norm(x_true)).item())

    xx, yy = torch.meshgrid(torch.linspace(-6, 6, 100), torch.linspace(-3, 3, 100), indexing='ij')
    zz = fctn(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1))[0]

    plt.figure()
    plt.contourf(xx, yy, zz.detach().reshape(xx.shape))
    plt.plot(x_true[0, 0], x_true[0, 1], 'y*', markersize=20)
    plt.plot(info['x'][:, 0], info['x'][:, 1], 'k-o', markersize=8)
    plt.colorbar()
    plt.show()