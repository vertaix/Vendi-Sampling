import torch
from typing import Callable

def ReplicaSamp(E: Callable, steps: int, x_init: torch.Tensor, eta: float=1e-3):
    replicas = x_init.size()[0]
    dim = x_init.size()[1]
    samples = torch.zeros((steps, replicas, dim))
    const = torch.sqrt(torch.tensor(2 * eta))

    for i in range(steps):
        x_init.requires_grad_()
        e = E(x_init)
        g = torch.autograd.grad(e, x_init,  grad_outputs=[torch.ones_like(e)])[0]
        x_init = x_init.detach() - eta * g + const * torch.randn((replicas, dim))
        samples[i] = x_init.detach()

    return samples.numpy()

def VendiSamp(E: Callable, score: Callable, steps: int, x_init: torch.Tensor, eta: float=1e-3, nu: float=4e-3, stop: int = 100):
    replicas = x_init.size()[0]
    dim = x_init.size()[1]
    samples = torch.zeros((steps, replicas, dim))
    weights = torch.ones((steps, replicas))
    
    const = torch.sqrt(torch.tensor(2 * eta))
    
    zeros = torch.zeros_like(x_init) #avoid having to reinitialize this 

    for i in range(steps):
        x_init.requires_grad_()
        if i < stop:
            vl = score(x_init) # vendi score
            gv = torch.autograd.grad(vl, x_init)[0] # vendi force
            vs = nu * gv * (1-i/stop) # vendi step
            weights[i] = 1. / torch.exp(vl).detach() * weights[i] # vendi weight
        else:
            vs = zeros
        e = E(x_init)
        g = torch.autograd.grad(e, x_init,  grad_outputs=[torch.ones_like(e)])[0]
        x_init = x_init.detach() - eta * g + const * torch.randn((replicas, dim)) + vs
        samples[i] = x_init.detach()

    return samples.numpy(), weights.numpy()