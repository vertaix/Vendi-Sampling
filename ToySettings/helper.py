import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable

from vendi import log_score
import pickle as pkl


def logvendi_loss(x):
    def sim(samples):
        #choice of similarity measure - using broadcasting to speed things up
        if samples.dim()==1:
            #if we have one-dimensional inputs, use following code
            samples1 = torch.unsqueeze(samples,0)
            samples2 = torch.unsqueeze(samples,1)

            K = 1. - torch.abs(samples1-samples2)/(torch.abs(samples1)+torch.abs(samples2))
            return K
        else:
            samples1 = torch.unsqueeze(samples,0)
            samples2 = torch.unsqueeze(samples,1)
            #samples-samples2 is of the shape (nsamples nsamples, dimension of input).
            #We want the l1 norm over the last part - the input dimension (so use dim=2)
            K = 1. - torch.norm(samples1-samples2, p=1, dim=2)/(torch.norm(samples1, p=1, dim=2)+torch.norm(samples2, p=1, dim=2))
            return K
    return log_score(x, sim)

def resample(samples, weights, energy):
    weights = weights.flatten()
    samples = samples.reshape(-1, samples.shape[-1])
    weights = weights/weights.sum()
    new_samp = np.random.choice(np.arange(len(samples)), size=1000, p=weights)
    samps = torch.tensor(samples[new_samp])
    return logvendi_loss(samps).detach().numpy(), torch.mean(energy(samps)).detach().numpy()


def createPrinzPlot(E, x_init, replicas, replica, vendi_s, ws, burn, snapshot_step, steps):
    fig, ax = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1.5, 1.5, 2]})
    
    replicaFE_Diff = []
    replica_steps = []
    
    vendiFE_Diff = []
    vendiFE_steps = []
    for i in range(len(replica)):
        diff, diff_steps = FreeEnergyBoundary_steps(E.energy, replica[i], boundary=0., weights=None, burn=burn, skip = 5000., lb = [-1], ub = [1.])
        replicaFE_Diff.append(diff)
        replica_steps.append(diff_steps)


        diff, diff_steps = FreeEnergyBoundary_steps(E.energy, vendi_s[i], boundary=0., weights=ws[i], burn=burn, skip = 5000., lb = [-1], ub = [1.])
        vendiFE_Diff.append(diff)
        vendiFE_steps.append(diff_steps)

    trueDiff, fx = getBoundaryDiff(E.energy, 0., [-1.], [1.])
    plot_differences_step(ax[-1], np.array(replicaFE_Diff), replica_steps, np.array(vendiFE_Diff), vendiFE_steps, trueDiff)

    ax[-1].set_xlabel('Steps')
    ax[-1].set_ylabel('Free Energy Difference')

    ax[-1].set_ylim(bottom=-3.5,top=0.5)

    ax[-1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[-1].set_xlim(left=0., right=steps)        
    
    plt.tight_layout()

    x=np.linspace(-1, 1, 101)
    Ex = E.energy(torch.from_numpy(x))
    px = torch.exp(-Ex)
    norm_const = 0.02*torch.sum(px).detach().numpy()

    ax[-2].scatter(x_init.detach().numpy(), E.energy(x_init).detach().numpy(), label='Initial Sample', color='green')
    ax[-3].scatter(x_init.detach().numpy(), torch.exp(-E.energy(x_init)).detach().numpy()/norm_const, label='Initial Sample', color='green')
        
    ax[-3].set_xlabel('Position')
    ax[-2].set_xlabel('Position')
    ax[-3].set_ylabel('Density')
    ax[-2].set_ylabel('Energy (KT)')
    
    densities = []
    energies = []
    centers = []
    for j in range(len(replica)):
        y, e_y, bc = get_1Denergy(replica[j][burn:snapshot_step], weights=None)
        densities.append(y)
        energies.append(e_y)
        centers.append(bc)
            
    x, new_densities, new_energies = align_bins(densities, energies, centers)
    plot_aligned(ax[-2], x, new_energies, label='Replica Sampling', color='red', alpha=0.8)

    plot_aligned(ax[-3], x, new_densities, label='Replica Sampling', color='red', alpha=0.8)

    energies = []
    centers = []
    densities = []

    for j in range(len(replica)):
        y, e_y, bc = get_1Denergy(vendi_s[j][burn:snapshot_step], weights=ws[j][burn:snapshot_step])
        densities.append(y)
        energies.append(e_y)
        centers.append(bc)
            
    x, new_densities, new_energies = align_bins(densities, energies, centers)
    plot_aligned(ax[-2], x, new_energies, label='Vendi Sampling', color='blue', alpha=0.8)

    plot_aligned(ax[-3], x, new_densities, label='Vendi Sampling', color='blue', alpha=0.8)
        
    ax[-2].plot(x, E.energy(torch.from_numpy(x)), label='Target: Prinz Pot.', color='black')
    ax[-2].legend()
    ax[-2].set_title('Replica-Size ' + str(replicas))
        
    ax[-3].plot(x, torch.exp(-E.energy(torch.from_numpy(x)))/norm_const, label='Target: Prinz Pot.', color='black')
    ax[-3].legend()

    return fig, ax
# TO-DO : make prettier?

def createDWPlot(E, x_init, replica, vendi_s, ws, burn, steps):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={'width_ratios': [3, 1, 2]})
    plt.subplots_adjust(wspace=0.25)
    
    prec = 100
    values = np.zeros((prec,prec))
    x = np.linspace(-2.5,2.5,prec)
    y = np.linspace(-4.,4.,prec)
    for i in range(len(x)):
        for j in range(len(y)):
            values[i,j] = E.energy(torch.tensor([x[i],y[j]]))

    ax[0].set_title('Energy Surface')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    pos = ax[0].imshow(values.T, cmap='viridis', interpolation=None, extent=[-2.5, 2.5, -4., 4.])

    fig.colorbar(pos, pad=0.04, fraction=0.046)
    
    
    trueDiff, fx = getBoundaryDiff(E.energy, 0., [-2.5, -4.], [2.5, 4.])

    x_init = x_init.detach().numpy()
    x = np.linspace(-2.5, 2.5, 100)
    x_init_energy = np.zeros(len(x_init))
    for i in range(len(x_init)):
        x_init_energy[i] = np.abs(x-x_init[i,0]).argmin()
    ax[1].scatter(x_init[:,0], fx[x_init_energy], label='Initial Sample')
    ax[1].plot(x, fx, color='black')
    ax[1].set_ylabel('Free Energy')
    ax[1].set_xlabel('x')
    ax[1].axvline(0., label='Boundary', linestyle='--', alpha=0.2, c='k')
    ax[1].legend()
    
    
    
    replicaFE_Diff = []
    replica_steps = []
    
    vendiFE_Diff = []
    vendiFE_steps = []

    skip_size = 50000
    for j in range(len(replica)):
        print(j)
        diff, diff_steps = FreeEnergyBoundary_steps(E.energy, replica[j], boundary=0., weights=None, burn=burn, skip = skip_size, lb = [-2.5, -4.], ub = [2.5, 4.])
        replicaFE_Diff.append(diff)
        replica_steps.append(diff_steps)

        diff, diff_steps = FreeEnergyBoundary_steps(E.energy, vendi_s[j], boundary=0., weights=ws[j], burn=burn, skip = skip_size, lb = [-2.5,-4.], ub = [2.5, 4.])

        vendiFE_Diff.append(diff)
        vendiFE_steps.append(diff_steps)

    plot_differences_step(ax[2], np.array(replicaFE_Diff), replica_steps, np.array(vendiFE_Diff), vendiFE_steps, trueDiff)
    ax[2].legend()
    ax[2].set_xlabel('Steps')
    ax[2].set_ylabel('Free Energy Difference')
    ax[2].set_ylim(bottom=0, top=4.0)
    ax[2].set_xlim(left=0, right=steps)
    
    plt.tight_layout()

    return fig, ax
def plot_1Denergy(ax, f, X, weights=None, x_min=-1., x_max=1., color='red', label='Sampler', plot_pot=True, plot_prob=True):
    x = X.flatten()

    if weights is None:
        weights = np.ones_like(x)
    weights = weights.flatten()
    y, binEdges = np.histogram(x, bins=100, weights=weights, density=True)

    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    if plot_prob:
        ax[0].plot(bincenters, y, '-', alpha=0.8, color=color, label='Density '+label)
        ax[1].plot(bincenters, -np.log(y), '-', alpha=0.8, color=color, label='Energy '+label)
    else:
        ax.plot(bincenters, -np.log(y), '-', alpha=0.8, color=color, label='Energy '+label)
    x = np.linspace(x_min, x_max, 100)
    if plot_pot: #plot potential
        if plot_prob:
            ax[0].plot(x, f(torch.from_numpy(x)), label='Target: Prinz Pot.', color='black')
            ax[1].plot(x, f(torch.from_numpy(x)), label='Target: Prinz Pot.', color='black')
        else:
            ax.plot(x, f(torch.from_numpy(x)), label='Target: Prinz Pot.', color='black')
def plot_2Denergy(ax, f, X, weights=None, x_min=-2.5, x_max=2.5, y_min=-4., y_max=4., color='red', label='Sampler', plot_pot=True):
    '''
    Assuming first dimension of sample is the 'slow' dimension
    '''
    x = X.reshape(-1, X.shape[-1])

    #if weights is not None:
    #    weights = weights.flatten()[np.abs(x[:,1])<eps]
    #x = x[np.abs(x[:,1])<eps]
    if weights is None:
        weights = np.ones(len(x))
    weights = weights.flatten()
    
    #y, binEdges = np.histogram(x[:,0], bins=100, weights=weights, density=True)
    
    H, xEdges, yEdges = np.histogram2d(x[:,0], x[:,1], bins=100, weights=weights, density=True)
    xCenter = 0.5*(xEdges[1:]+xEdges[:-1])
    yCenter = 0.5*(yEdges[1:]+yEdges[:-1])
    
    dy = np.diff(yEdges)
    Px = np.matmul(H, dy)
    #Px = np.sum(H.T, axis=0)
    
    #bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax[0].plot(xCenter, Px, '-', alpha=0.8, color=color, label='Density '+label)
    ax[1].plot(xCenter, -np.log(Px), '-', alpha=0.8, color=color, label='Energy '+label)
    
   
    if plot_pot: #plot potential
        x0 = np.linspace(x_min, x_max, len(xCenter))
        _, fx = getBoundaryDiff(f, 0, [x_min, y_min], [x_max, y_max], precision=len(xCenter))
        ax[0].plot(x0, fx, label='Target: Prinz Pot.', color='black')
        ax[1].plot(x0, fx, label='Target: Prinz Pot.', color='black')

def get_1Denergy(X, weights=None):
    x = X.flatten()

    if weights is None:
        weights = np.ones_like(x)
    weights = weights.flatten()
    y, binEdges = np.histogram(x, bins=100, weights=weights, density=True)

    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    
    return y, -np.log(y), bincenters

def align_bins(densities, energies, centers):
    left_end = np.inf
    right_end = -np.inf
    for i in range(len(centers)):
        left_end = min(centers[i][0],left_end)
        right_end = max(centers[i][-1],right_end)
    x = np.linspace(left_end, right_end, 90)
    new_energies = np.zeros((len(centers), len(x)))
    new_densities = np.zeros((len(centers), len(x)))
    for i in range(len(x)):
        for j in range(len(centers)):
            new_energies[j,i] = energies[j][np.abs(centers[j]-x[i]).argmin()]
            new_densities[j,i] = densities[j][np.abs(centers[j]-x[i]).argmin()]
    return x, new_densities, new_energies

def plot_aligned(ax, x, new_energies, label, color, alpha=0.8):
    mu = np.mean(new_energies, axis=0)
    std = np.std(new_energies, axis=0)
    ax.plot(x, mu, '-', color=color, label=label, alpha=alpha)
    ax.fill_between(x, mu - std, mu + std,
                     color=color, alpha=0.2)

def plot_differences_step(ax, ula, ula_t, vula, vula_t, truediff):
    #fig, ax = plt.subplots()
    ax.axhline(truediff, label=r'True Free Energy Difference', linestyle='--', alpha=0.2, c='k')
    
    #Plot until one of the samplers finishes (for both types)
    
    first_end = len(ula_t[0])
    
    
    #t = [i for i in range(first_end)]
    t = ula_t[0]
    #print(ula)

    # ULA
    mu = ula.mean(axis=0)
    std = ula.std(axis=0)

    ax.plot(t, mu, 'red', label='Replica Sampling')
    ax.fill_between(t, mu - std, mu + std,
                     color='red', alpha=0.2)

    # VULA
    mu = vula.mean(axis=0)
    std = vula.std(axis=0)
    ax.plot(t, mu, 'blue', label='Vendi Sampling')
    ax.fill_between(t, mu - std, mu + std,
                     color='blue', alpha=0.2)
    return ax







def getBoundaryDiff(f: Callable, boundary, lb, ub, precision: int =100):
    #Get the energy difference for the energy function at the boundary, using a lower and upper bound to define integration bounds
    #Assuming boundary is an integer representing a boundary at x=boundary (i.e. x=0, x=-1, etc.)
    #lb and ub should be floats

    dim = len(lb)
    if dim==1:
        x = torch.linspace(lb[0], boundary, precision)
        y = torch.exp(-f(x))
        lowerE = torch.trapezoid(y, x)
        
        x = torch.linspace(boundary, ub[0], precision)
        y = torch.exp(-f(x))
        upperE = torch.trapezoid(y, x)
        return -torch.log(upperE/lowerE), None
    
    elif dim==2:
        x = torch.linspace(lb[0], ub[0], precision)
        y = torch.linspace(lb[1], ub[1], precision)
        y = torch.unsqueeze(y,0)
        y = torch.transpose(y,0,1)
        fx = torch.zeros_like(x)
        for i in range(len(x)):
            coord = x[i]*torch.ones_like(y)
            coord = torch.cat((coord, y),dim=1)

            fx[i] = -torch.log(torch.trapz(torch.exp(-f(coord)), coord[:,1].flatten()))

        lower = x<=boundary
        upper = x>boundary
        lowerE = torch.trapz(torch.exp(-fx[lower]), x[lower])
        upperE = torch.trapz(torch.exp(-fx[upper]), x[upper])
        return -torch.log(upperE/lowerE), fx

def FreeEnergyBoundary_steps(f: Callable, X, boundary, weights=None, burn: int=100., skip: int=1000., lb = [-1], ub = [1.], calcTrue: bool=False):
    dim = X.shape[-1]
    tot_steps = X.shape[0]
    replica = X.shape[1]
    if calcTrue:
        trueDiff, fx = getBoundaryDiff(f, boundary, lb, ub)
    
    if X.ndim>2:
        x = X.reshape(-1, X.shape[-1])
    else:
        x = X.flatten()
       
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = weights.flatten()
    
    energyDiff = []

    steps = np.arange(skip+burn, tot_steps+1, skip)
    burn = int(burn*replica)
    #print(x.shape, burn)
    if x.shape[-1]==1:
        for i in steps:
            ind = int(i*replica)
            y, binEdges = np.histogram(x[burn:ind].flatten(), bins=50, weights=weights[burn:ind], density=True)
            binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            energyDiff.append(samplerEnergy1D(y, binCenters, binEdges, boundary))
        return energyDiff, steps
    else:
        for i in steps:
            ind = int(i*replica)  
            rel_x = x[burn:ind,0]
            lt = (rel_x<0).sum()/len(rel_x)
            gt = (rel_x>0).sum()/len(rel_x)
            energyDiff.append(-np.log(gt)+np.log(lt))
            #H, xEdges, yEdges = np.histogram2d(x[burn:ind,0], x[burn:ind,1], bins=250, weights=weights[burn:ind], density=True)
            #xCenter = 0.5*(xEdges[1:]+xEdges[:-1])
            #yCenter = 0.5*(yEdges[1:]+yEdges[:-1])
            #energyDiff.append(samplerEnergy2D(H, xCenter, yCenter, xEdges, yEdges, boundary))
        return energyDiff, steps

def FreeEnergyBoundary(f: Callable, X, T, boundary, weights=None, burn: float = 0.1, skip: float = 10., lb = [-1], ub = [1.], calcTrue: bool=False, eps: float=1e-1):
    #Numerically calculate what the free energy is on either side of the boundary
    dim = X.shape[-1]
    if calcTrue:
        trueDiff, fx = getBoundaryDiff(f, boundary, lb, ub)
    t = T.flatten()

    burn_tidx =np.abs(t-burn).argmin()
    
    if X.ndim>2:
        x = X.reshape(-1, X.shape[-1])
    else:
        x = X.flatten()
    t = t[burn_tidx:]
    x = x[burn_tidx:]
        
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = weights.flatten()
        weights = weights[burn_tidx:]
    #Code for 1D and 2D cases: (Will need KDE for higher dimension)
    energyDiff = []
    timesteps = np.arange(skip, np.max(t), skip)
    if x.shape[-1]==1:
        for i in timesteps:
            ind = np.abs(t-i).argmin()
            y, binEdges = np.histogram(x[:ind].flatten(), bins=50, weights=weights[:ind], density=True)
            binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            energyDiff.append(samplerEnergy1D(y, binCenters, binEdges, boundary))
    elif x.shape[-1]==2:

        for i in timesteps:
            ind = np.abs(t-i).argmin()         

            H, xEdges, yEdges = np.histogram2d(x[:ind,0], x[:ind,1], bins=1000, weights=weights[:ind], density=True)
            xCenter = 0.5*(xEdges[1:]+xEdges[:-1])
            yCenter = 0.5*(yEdges[1:]+yEdges[:-1])
            energyDiff.append(samplerEnergy2D(H, xCenter, yCenter, xEdges, yEdges, boundary))

    energyDiff= np.array(energyDiff)
    if calcTrue:  
        return energyDiff, timesteps, trueDiff
    else:
        return energyDiff, timesteps
    
def samplerEnergy1D(y, bincenters, binedges, boundary):

    lower = bincenters<boundary
    upper = ~lower

    dx = np.diff(binedges)
    lowerP = np.sum(y[lower]* dx[lower])
    upperP = np.sum(y[upper]*dx[upper]) 
    return -np.log(upperP) + np.log(lowerP)
    
def samplerEnergy2D(H, xCenters, yCenters, xEdges, yEdges, boundary):
    #Each row in H corresponds to a fixed x-value
    lower = xCenters<boundary

    upper = ~lower
    dx = np.diff(xEdges)
    dy = np.diff(yEdges)

    lowerP = np.sum(np.matmul(np.matmul(dx[lower],H[lower,:]),dy))
    upperP = np.sum(np.matmul(np.matmul(dx[upper],H[upper,:]),dy))

    return -np.log(upperP) + np.log(lowerP)
