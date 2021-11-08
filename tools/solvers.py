import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import Callable



class ODESolver(metaclass=ABCMeta):
    """
    Абстрактный класс для решения ОДУ 
    :param func: nn.Module – функция, аппроксимирующая производную (на вход принимает z (nD-тензор) и t (1D-тензор))
    :param z0: torch.Tensor – начальное значение ОДУ
    :param step_size: float – шаг дискретизации t
    :param grid_constructor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] – способ создания сетки интегрирования (значения ti в которых происходит подсчет интеграла)
    :param interp: str – способ интерполяции результата интегрирования
    :return z: torch.Tensor – значения функции z(t) на интервале t в кажды момент ti 
    """
    order: int

    def __init__(self, 
                 func: nn.Module, 
                 z0: torch.Tensor, 
                 step_size: float = None, 
                 grid_constructor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]=None,
                 interp="linear"):

        self.func = func
        self.z0 = z0
        self.dtype = z0.dtype
        self.device = z0.device
        self.step_size = step_size
        self.interp = interp

        if step_size is None and grid_constructor is None:
            self.grid_constructor = lambda f, z0, t: t
        elif grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(
                step_size)
        elif step_size is None:
            self.grid_constructor = grid_constructor
        else:
            raise ValueError(
                "Один из парамеров step_size или grid_constructor должен отсутствовать")

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, z0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(
                0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abstractmethod
    def _step_func(self, func: nn.Module, t0, dt, t1, z0: torch.Tensor):
        """
        Метод, отвечающий за шаг интегрирования метода.
        Таким образом этот метод определяет сам алгоритм решения, 
        например Эйлер или РК4
        """
        pass

    def __call__(self, t: torch.Tensor):
        """
        Функция интегрирования. 
        Вызыввает решение ОДУ на интервале t
        """
        time_grid = self.grid_constructor(self.func, self.z0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1], "Неправильный метод grid_constructor"
        assert t.dim() == 1, "t – 1D-тензор"

        solution = torch.empty(len(t), *self.z0.shape, dtype=self.dtype, device=self.device)
        solution[0] = self.z0

        j = 1
        z0 = self.z0

        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            dz, f0 = self._step_func(self.func, t0, dt, t1, z0)
            z1 = z0 + dz

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, z0, z1, t[j])
                elif self.interp == "cubic":
                    f1 = self.func(t1, z1)
                    solution[j] = self._cubic_hermite_interp(t0, z0, f0, t1, z1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
            
                j += 1

            z0 = z1
        
        return solution
    
    def _cubic_hermite_interp(self, t0, z0, f0, t1, z1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * z0 + h10 * dt * f0 + h01 * z1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, z0, z1, t):
        if t == t0:
            return z0
        if t == t1:
            return z1
        slope = (t - t0) / (t1 - t0)
        return z0 + slope * (z1 - z0)

    def __str__(self):
        return f"Метод численного решения ОДУ {self.order}-го порядка"



class Euler(ODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, z0):
        f0 = func(t0, z0)
        return dt * f0, f0


_one_third = 1/3
_two_thirds = 2/3

class RK4(ODESolver):
    order = 4
    
    def _step_func(self, func, t0, dt, t1, z0):
        
        k1 = func(t0, z0)
        k2 = func(t0 + dt * _one_third, z0 + dt * k1 * _one_third)
        k3 = func(t0 + dt * _two_thirds, z0 + dt * (k2 - k1 * _one_third))
        k4 = func(t1, z0 + dt * (k1 - k2 + k3))

        return dt * (k1 + 3 * (k2 + k3) + k4) * 0.125, k1


