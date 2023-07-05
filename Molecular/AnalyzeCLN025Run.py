import torch
import pickle as pkl
import matplotlib
import matplotlib.transforms as mtransforms
from pyemma.coordinates.transform import TICA
import pyemma as pe
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

def compile_parts(path_prefix, parts=51):
    file0 = path_prefix+'0.npy'
    samp1 = np.load(file0)

    for i in range(1, parts):
        print(i)
        filen = np.load(path_prefix+str(i)+'.npy')
        samp1 = np.hstack((samp1, filen))
    samples = samp1
    return samples

def snapshotHeatMap(ax, phi, psi, t, t_per_step=2e-3, replica=32, weights=None):
    end = int(replica*t/t_per_step)

    t_phi = phi[:end]
    t_psi = psi[:end]
    if weights is not None:
        t_w = weights[:end]
    else:
        t_w = None
    H, xEdges, yEdges = np.histogram2d(t_phi, t_psi, bins=150, weights=t_w, density=True)
    xCenter = 0.5*(xEdges[1:]+xEdges[:-1])
    yCenter = 0.5*(yEdges[1:]+yEdges[:-1])
    
    H = H.T
    E = -np.log(H)
    pos = ax.imshow(E-np.min(E), interpolation=None, origin='lower',
        extent=[xEdges[0], xEdges[-1], yEdges[0], yEdges[-1]], vmin=0, vmax=9)

    return pos

def mdtraj_obj(coordinates, pdb):
    top = md.load(pdb).topology
    return md.Trajectory(coordinates.reshape((-1, top.n_atoms, 3)), top)

def tica_project(trajectory, tica_param_filename):
    tica_sovs = TICA(lag=5)
    tica_params = np.load(tica_param_filename)
    tica_sovs.model.update_model_params(mean=tica_params["tica_mean"],
                                cov=tica_params["tica_c0"],
                                cov_tau=tica_params["tica_ctau"] )

    featur = pe.coordinates.featurizer(trajectory)
    featur.add_distances_ca()
    Y = featur.transform(trajectory)

    return tica_sovs.transform(Y)

if __name__ == "__main__":
    burn = 80000 #Hard-coded value, change as you see fit.
    replicas = 32
    burn = int(burn*replicas/500) #/500 since we record 1/500 steps (true ts in simulation is 4 fs/step, we record each step every 2 ps)

    
    #Samples is of shape # of Replicas x # of Recoderded Steps x # Number of atoms x 3 
    #Also define the prefix directory for where the output files are saved
    samples = compile_parts('Vendi_CLN025/outputPart', parts=4)
    vendi_samp = samples.transpose(1,0,2,3).reshape(-1, samples.shape[-2], samples.shape[-1])
    
    traj = mdtraj_obj(vendi_samp, 'system/structure_vaccum.pdb')
    features = tica_project(traj, 'system/cln025_tica_parameters.npz')

    ang1, ang2 = features[:,0], features[:,1]
    
    fig, ax = plt.subplots()
    
    pos = snapshotHeatMap(ax, ang1, ang2, t=4) #Can set this to be the end of the simulation
    fig.colorbar(pos, ax=ax, label="$k_B T$", pad=0.08)
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\psi$')