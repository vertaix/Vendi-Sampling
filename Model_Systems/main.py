import typer
import torch
from model_systems import PrinzEnergy, DoubleWell
import numpy as np
import matplotlib.pyplot as plt
import time

from samplers import ReplicaSamp, VendiSamp
from helper import *
import pickle as pkl


def main(E: str = 'PrinzEnergy', seed: int = 100, dim: int = 1, eta: float = 5e-5,
         hyperopt: bool = False, nu: float  = 4e-3, stop: int = 100, replicas: int = 32, steps: int = 10000, 
        reproductions: int = 10):
    assert E in ['PrinzEnergy', 'DoubleWell'], 'Unknown Energy Function.'
    
    '''
    Example Calls:
    python main.py --e DoubleWell --seed 2 --dim 2 --eta 1e-2 --hyperopt --replicas 32 --steps 2500000 --reproductions 10
    python main.py --e PrinzEnergy --seed 1 --eta 1e-4 --hyperopt --replicas 8 --steps 400000 --reproductions 10

    E: Energy Function
    
    dim: Dimension of the Energy Function
    
    eta: step-size used in simulations
    hyperopt: Boolean flag for doing hyperparameter optimization
    nu: Coefficient of the Vendi Force (will be set automatically if hyperopt is True)
    stop: How long vendi force will be applied
    replicas: Number of simulation replicas
    steps: Number of steps in the simulation  
    Reproductions: Number of trials
    '''
    
    # init seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # init energy function
    Ename = None
    if E == 'PrinzEnergy':
        Ename = 'PrinzEnergy'
        E = PrinzEnergy()
        x_init = torch.rand((replicas, dim), requires_grad=True) #U[0,1]
    elif E == 'DoubleWell':
        Ename = 'DoubleWell'
        E = DoubleWell()
        x_init = 5*torch.rand((replicas, dim), requires_grad=True)-2.5  # U[-2.5,2.5]^2
    print(Ename)

    # ReplicaSampling
    replica = []
    for i in range(reproductions):
        start = time.time()
        replica_res = ReplicaSamp(E.energy, steps, x_init, eta=eta)
        replica.append(replica_res)
        print(f'Replica Run-Time (iter {i+1} / {reproductions}): {time.time() - start:.2f}')

    # re-init seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Vendi Sampling
    if hyperopt:
        print('Starting Hyperparameter Search')
        best_score, best_stop, best_nu =-np.inf,None,None
        start = time.time()
        nu_coeff = [10,40,100]
        if dim==1:
            stops = [100, 500, 1000]
        else:
            stops = [10000,50000]
        for s in stops:
            for b in nu_coeff:
                res, weights = VendiSamp(E.energy, logvendi_loss, min(2000,steps), x_init, eta=eta, nu=eta*b, stop=s)
                vs, avg_energy = resample(res, weights, E.energy)
                if vs-2*avg_energy>best_score:
                    best_stop,best_nu =s,eta*b
        nu = best_nu
        stop = best_stop
        print(f'Finished Hyperparameter Search After {time.time()-start:.2f}, choosing nu={nu}, stopping VS after {stop} steps')
        
    vendi_s = []
    ws = []
    for i in range(reproductions):
        start = time.time()
        res, weights = VendiSamp(E.energy, logvendi_loss, steps, x_init, eta=eta, nu=nu, stop=stop)
        vendi_s.append(res)
        ws.append(weights)
        print(f'Vendi Run-Time (iter {i+1} / {reproductions}): {time.time() - start:.2f}')
    
    burn = stop #discard all samples where the vendi score is active (so we only use unbiased samples)
    #Note: Using the proposed reweighting scheme gives identical results.
    
    if Ename=='PrinzEnergy':
        fig, ax = createPrinzPlot(E, x_init, replicas, replica, vendi_s, ws, burn, snapshot_step=10000+burn, steps=steps)
        fig.savefig('PrinzPlot.pdf')
    elif Ename=='DoubleWell':
        fig, ax = createDWPlot(E, x_init, replica, vendi_s, ws, burn, steps)
        fig.savefig('DWPlot.pdf')


if __name__ == '__main__':
    typer.run(main)