import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.solvers import *
from tools.misc import _check_inputs, _flat_to_shape


SOLVERS = {
    'Euler': Euler,
    'RK4': RK4
}


def odeint(func, z0, t, method='Euler', options={}):

    shapes, func, z0, t, method, options, t_is_reversed = _check_inputs(func, z0, t, method, options, SOLVERS)
    solver = SOLVERS[method](func=func, z0=z0, **options)
    solution = solver(t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
        
    return solution


class ODEAdjointMethod(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, func, z0, t, method, options, adjoint_method, adjoint_options, t_requires_grad, *adjoint_params):

        ctx.func = func
        ctx.adjoint_method = adjoint_method
        ctx.t_requires_grad = t_requires_grad
        ctx.adjoint_options = adjoint_options

        with torch.no_grad():
            z = odeint(func, z0, t, method=method, options=options)
            ctx.save_for_backward(t, z, *adjoint_params)
            
        return z

    @staticmethod
    def backward(ctx, *grad_z):
        with torch.no_grad():
            func = ctx.func
            adjoint_method = ctx.adjoint_method
            t_requires_grad = ctx.t_requires_grad
            adjoint_options = ctx.adjoint_options

            t, z, *adjoint_params = ctx.saved_tensors
            grad_z = grad_z[0]

            adjoint_params = tuple(adjoint_params)

            aug_state = [torch.zeros((), dtype=z.dtype, device=z.device), z[-1], grad_z[-1]]
            aug_state.extend([torch.zeros_like(param) for param in adjoint_params])

            def augmented_dynamics(t, z_aug):
                z = z_aug[1]
                adj_z = z_aug[2]

                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    z = z.detach().requires_grad_(True)

                    func_eval = func(t if t_requires_grad else t_, z)

                    _t = torch.as_strided(t, (), ()) 
                    _z = torch.as_strided(z, (), ())
                    _params = tuple(torch.as_strided(param, (), ())
                                    for param in adjoint_params)
                    
                    vjp_t, vjp_z, *vjp_params = torch.autograd.grad(
                        func_eval, (t, z) + adjoint_params, -adj_z,
                        allow_unused=True, retain_graph=True
                    )

                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_z = torch.zeros_like(z) if vjp_z is None else vjp_z
                vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)]

                return (vjp_t, func_eval, vjp_z, *vjp_params)

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    func_eval = func(t[i], z[i])
                    dLd_cur_t = func_eval.reshape(
                        -1).dot(grad_z[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t
                aug_state = odeint(augmented_dynamics, tuple(aug_state), t[i - 1:i + 1].flip(0), method=adjoint_method, options=adjoint_options)
                
                aug_state = [a[1] for a in aug_state]
                aug_state[1] = z[i - 1]
                aug_state[2] += grad_z[i - 1]
            
            if t_requires_grad:
                time_vjps[0] = aug_state[0]

            adj_z = aug_state[2]
            adj_params = aug_state[3:]

            return (None, None, adj_z, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)


def find_parameters(module):

    assert isinstance(module, nn.Module)

    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items()
                      if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def odeint_adjoint(func, z0, t, method='Euler', options={}, adjoint_method=None, adjoint_options=None, adjoint_params=None):

    if adjoint_method is None:
        adjoint_method = method

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))

    if adjoint_options is None:
        adjoint_options = options

    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    shapes, func, z0, t, method, options, t_is_reversed = _check_inputs(func, z0, t, method, options, SOLVERS)

    solution = ODEAdjointMethod.apply(
        func, z0, t, method, options, adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    return solution
