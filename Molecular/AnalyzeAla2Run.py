import torch
import pickle as pkl
import matplotlib
import matplotlib.transforms as mtransforms
from pyemma.coordinates.transform import TICA
import pyemma as pe
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

def compute_ala2_phipsi(samples, ala2_top="system/alanine-dipeptide.pdb"):
    """
    Compute Ala2 Ramachandran angles
    Parameters
    ----------
    traj : mdtraj.Trajectory
    """

    topology = md.load(ala2_top).topology
    traj = md.Trajectory(xyz=samples, topology=topology)

    phi_atoms_idx = [4, 6, 8, 14]
    phi = md.compute_dihedrals(traj, indices=[phi_atoms_idx])[:, 0]
    psi_atoms_idx = [6, 8, 14, 16]
    psi = md.compute_dihedrals(traj, indices=[psi_atoms_idx])[:, 0]

    return phi, psi

def snapshotHeatMap(ax, phi, psi, t, t_per_step=1e-4, replica=32, weights=None, show_bar=False):
    end = int(replica*t/t_per_step)

    t_phi = phi[:end]
    t_psi = psi[:end]
    if weights is not None:
        t_w = weights[:end]
    else:
        t_w = None
    H, xEdges, yEdges = np.histogram2d(t_phi, t_psi, bins=150, range=[[-np.pi,np.pi],[-np.pi,np.pi]], weights=t_w, density=True)
    xCenter = 0.5*(xEdges[1:]+xEdges[:-1])
    yCenter = 0.5*(yEdges[1:]+yEdges[:-1])
    
    H = H.T
    E = -np.log(H)
    pos = ax.imshow(E-np.min(E), interpolation=None, origin='lower',
        extent=[xEdges[0], xEdges[-1], yEdges[0], yEdges[-1]], vmin=0, aspect='auto')
    
    ax.set_ylim(bottom=-np.pi, top=np.pi)
    ax.set_xlim(left=-np.pi, right=np.pi)
    
    return pos

def compile_parts(path_prefix, parts=31):
    file0 = path_prefix+'0.npy'
    samp1 = np.load(file0)

    for i in range(1, parts):
        print(i)
        filen = np.load(path_prefix+str(i)+'.npy')
        samp1 = np.hstack((samp1, filen))
    samples = samp1
    return samples

if __name__ == "__main__":
    burn = 7500 #Hard-coded value, change as you see fit.
    replicas = 32
    burn = int(burn*replicas/50) #/50 since we record 1/50 steps (true ts in simulation is 2 fs/step, we record 100 fs/step)

    
    #Samples is of shape # of Replicas x # of Recoderded Steps x # Number of atoms x 3 
    samples = compile_parts('Vendi_Ala2/outputPart', parts=31)
    print(samples.shape)

    #Reshape into a 3D Numpy array: The first dimension contains each recorded sample. Blocks of 32 correspond to all replicas at given time-step
    vendi_samp = samples.transpose(1,0,2,3).reshape(-1, samples.shape[-2], samples.shape[-1])

    
    phi, psi = compute_ala2_phipsi(vendi_samp, 'system/alanine-dipeptide.pdb')
    
    #Note: Use the above angles to compute any kind of metric of interest such as Free Energy differences over axes
    
    fig, ax = plt.subplots()
    
    pos = snapshotHeatMap(ax, phi, psi, 1) 
    fig.colorbar(pos, ax=ax, label="$k_B T$", pad=0.08)
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\psi$')