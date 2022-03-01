from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb
import torch


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0, v0):
        #print(y0.shape, v0.shape)
        f0 = func(t0, torch.cat((y0, v0), dim=-1))
        return dt * f0, f0


class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0, v0):
        half_dt = 0.5 * dt
        f0 = func(t0, torch.cat((y0, v0), dim=-1), perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        return dt * func(t0 + half_dt, torch.cat((y_mid, v0), dim=-1)), f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0, v0):
        f0 = func(t0, torch.cat((y0, v0), dim=-1), perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, v0, f0=f0, perturb=self.perturb), f0
