import typer
from reform import simu_utils
from openmm import app, unit
from openmmtorch import TorchForce
import numpy as np
import torch
import pickle as pkl
import openmm
from timeit import default_timer as timer

class VendiModule(torch.nn.Module):

    def __init__(self, n_replicas, n_particles, device, nu, stop=1000, gamma=1.,offset=0, q=1.):
        super().__init__()
        self.n_replicas = n_replicas
        self.n_particles = n_particles
        self.nu = nu

        self.device = device
        self.current_step = torch.tensor(0.0)
        self.stop = torch.tensor(stop)
        self.gamma = gamma
        self.offset = offset
        self.q=q
        print(q)
    def getInvariant(self, positions):
        new_pos = torch.zeros(self.n_replicas, 1, self.n_particles*3)
        for i in range(len(positions)):
            inp = positions[i].view(-1, self.n_particles, 3)-self.offset
            com = torch.mean(inp, dim=1)
            centered = inp-com[:,None,:]
            
            top = centered[:,-1]
            A = top / torch.norm(top, dim=1)[:, None]

            B = torch.tensor([1., 0., 0.], device=A.device)

            B = B.type(A.dtype)

            dot = (A @ B)
            cross = torch.cross(A, B.repeat(A.size(0), 1))
            cross_norm = torch.norm(cross, dim=1)

            zeros = torch.zeros_like(dot, device=dot.device)
            ones = torch.ones_like(dot, device=dot.device)
            G1 = torch.vstack((dot, -cross_norm, zeros)).T
            G2 = torch.vstack((cross_norm, dot, zeros)).T
            G3 = torch.vstack((zeros, zeros, ones)).T
            G = torch.stack((G1, G2, G3), dim=2)

            u = A
            v = (B - (dot[:, None] * A)) / torch.norm((B - (dot[:, None] * A)), dim=1)[:, None]
            w = torch.cross(B.repeat(A.size(0), 1), A)
            F_inv = torch.stack([u, v, w], dim=2)
            F = torch.inverse(F_inv)
            U = torch.matmul(torch.matmul(F_inv, G), F)
            rotated = torch.matmul(centered, U)
            new_pos[i] = rotated.view((1, self.n_particles*3))
        return new_pos
    
    def RBF(self, positions):
        positions = positions.reshape(self.n_replicas, self.n_particles * 3)
        samples1 = torch.unsqueeze(positions, 0)
        samples2 = torch.unsqueeze(positions, 1)
        return torch.exp(-self.gamma*torch.norm(samples1 - samples2, p=2, dim=2) ** 2)
    
    
    def VS(self, positions):
        new_pos = self.getInvariant(positions)
        K = self.RBF(new_pos)
        K_ = K / K.shape[0]
        
        w, _ = torch.linalg.eigh(K_)
        
        p_ = w[w>0]
        if self.q==1:
            entropy_q = -(p_ * torch.log(p_)).sum()
        elif self.q == -1: #-1 q corresponds to q=inf
            entropy_q = -torch.log(torch.max(p_))
        else:
            entropy_q = torch.log((p_ ** self.q).sum()) / (1 - self.q)

        return entropy_q

    def forward(self, positions):
        #nu is the step-size for the vendi-force relative to the energy force
        positions = positions.reshape(self.n_replicas, -1, 3)

        # anneal for next time step
        self.current_step += torch.tensor(1.0)

        if self.stop < self.current_step:
            return torch.sum(positions * 0.0)
        else:
            return -self.nu*self.VS(positions)

def main(simu_time: float = 50, recording_interval: float = 0.2,
        temperature: float = 350, nu: float = 100., replicas: int = 32, stop : int = 1000, gamma : float=1.,
        vendi: bool = False, verbose: bool = True, 
        output_path: str = './output', part: int=0, q: float=1.):
    '''
    Code for generating simulations of Chignolin (CLN025) in implicit solvent
    New systems in implicit solvent can be simulated by adding system files to 'system' directory
    
    Script to generate a simulation are included
    
    simu-time: (in picoseconds) length of simulation
    recording-interval: (in picoseconds) the frequency at which we record states
   
    temperature: Tempurature in Kelvin 
    nu: Vendi Force Coefficient 
    replicas: Number of parallel replicas used in simulation
    gamma: RBF bandwidth parameter

    vendi: boolean flag for if the Vendi Force can be applied
    
    output_path: Defines prefix for where files are saved
    part: If nonzero, loads the initial position from last sample in an earlier simulation (searches in output_path directory)
    '''

    system = 'CLN025'

    exchange_interval = 0.0 
    stop = int(stop)
    if torch.cuda.is_available(): 
        print('Using GPU')
        dev = "cuda:0" 
        platform = "CUDA"
    else: 
        dev = "cpu" 
        platform = "CPU"
    device = torch.device(dev) 
    print('Using Device ', dev)
    # --- OpenMM system setup ---
    
    if system == 'CLN025':
        if verbose:
            print(f'Setting up the {system} system...')
        params = app.CharmmParameterSet("system/top_all22star_prot.rtf", "system/top_water_ions.rtf", 'system/parameters_ak_dihefix.prm')
        solv = app.CharmmPsfFile('system/structure_vacuum.psf')
        crds = app.PDBFile("system/structure_vaccum.pdb")
        system = solv.createSystem(params, nonbondedMethod=app.NoCutoff,
                          constraints=app.HBonds,
                          hydrogenMass=4. * unit.amu,
                          implicitSolvent=app.OBC2,
                          )
        pos = crds.positions
    else:
        exit() #Add other implicit solvent system here!
 
    time_step_in_fs = 4.0
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 0.1, "time_step_in_fs": time_step_in_fs}
    ts = [temperature for _ in range(replicas)]
    
    start = timer()

    name='Part'+str(part)

    offset=100
    if part==0 and vendi:
        print('Using Vendi Force')
        n_particles = len(pos)
        # Render the compute graph to a TorchScript module

        module = torch.jit.script(VendiModule(replicas, n_particles, nu=torch.tensor(nu), device=device, stop=stop, gamma=gamma, offset=offset, q=q))
        # Serialize the compute graph to a file
        module.save(MODEL_PATH)
        force = TorchForce(MODEL_PATH)

        simu = simu_utils.MultiTSimulation(system, ts, interface=interface,
                                       integrator_params=integrator_params, verbose=verbose, platform=platform,
                                       replicated_system_additional_forces=[force])
        
        pos = np.array([pos]*replicas)
        
        #We define an offset to ensure atoms are not overlapping in parallel replicas - 
        #OpenMM customGBforces do not work with parell replicas (even if you add exclusions)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                for k in range(pos.shape[2]):
                    if i%2==0:
                        pos[i,j,k] = pos[i,j,k] - openmm.unit.Quantity(offset*i/2, unit=unit.nanometer)
                    else:
                        pos[i,j,k] = pos[i,j,k] + openmm.unit.Quantity(offset*(i+1)/2, unit=unit.nanometer)
        
        simu.set_positions(pos)
        print('compiling energy')
        simu.minimize_energy()
        print('finished compiling energy')

        simu.set_velocities_to_temp()
        steps = simu_utils.recording_hook_setup(simu=simu, simu_time=simu_time,
                                                 recording_interval=recording_interval,
                                                 output_path=output_path+name,
                                                 exchange_interval=exchange_interval)
        simu.run(steps)
    else:
        simu = simu_utils.MultiTSimulation(system, ts, interface=interface,
                                           integrator_params=integrator_params, verbose=verbose, platform=platform)
        
        pos = np.array([pos]*replicas)
        
        
        if part==0:
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    for k in range(pos.shape[2]):
                        if i%2==0:
                            pos[i,j,k] = pos[i,j,k] - openmm.unit.Quantity(offset*i/2, unit=unit.nanometer)
                        else:
                            pos[i,j,k] = pos[i,j,k] + openmm.unit.Quantity(offset*(i+1)/2, unit=unit.nanometer)
            simu.set_positions(pos)

        else:
            prev_name = output_path + 'Part'+str(part-1)+'.npy'
            samples = np.load(prev_name)
            pos = samples[:,-1,:,:]

            simu.set_positions(pos)

        if part == 0:
            simu.minimize_energy()
        
        simu.set_velocities_to_temp()

       

        steps = simu_utils.recording_hook_setup(simu=simu, simu_time=simu_time,
                                             recording_interval=recording_interval,
                                             output_path=output_path+name,
                                             exchange_interval=exchange_interval)

            
        simu.run(steps)

        #print(system.getForces())
        #for i, f in enumerate(system.getForces()):
        #    state = simu._context.get_rep_state(getEnergy=True, groups={i})
        #    print(f.getName(), state.getPotentialEnergy())

    end = timer()
    print(f'Simulation Took {end-start} s')
if __name__ == '__main__':
    interface = 'replicated_system'

    MODEL_PATH = './2model.pt'

    typer.run(main)