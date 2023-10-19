import typer
from reform import simu_utils
from openmm import app, unit
from openmmtorch import TorchForce
import numpy as np
import torch
import pickle as pkl

from timeit import default_timer as timer

class VendiModule(torch.nn.Module):

    def __init__(self, n_replicas, n_particles, device, nu, stop=1000, gamma=1., q=1.):
        super().__init__()
        self.n_replicas = n_replicas
        self.n_particles = n_particles
        self.nu = nu

        self.device = device
        self.current_step = torch.tensor(0.0)
        self.stop = torch.tensor(stop)
        self.gamma = gamma
        self.q = q
        print(self.q)
    def getInvariant(self, positions):
        new_pos = torch.zeros(self.n_replicas, 1, self.n_particles*3)
        for i in range(len(positions)):
            inp = positions[i].view(-1, self.n_particles, 3)
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
        # tracking time step
        self.current_step += torch.tensor(1.0)
        if self.stop <= self.current_step:
            return torch.sum(positions * 0.0) # returns grad = 0
        else:
            return -self.nu*self.VS(positions)

def main(simu_time: float = 50, recording_interval: float = 0.1,
        exchange_interval: float = 0.0, temperature: float = 300.15, nu: float = 100.,
        replicas: int = 32, output_path: str = './output', stop : int = 1000, gamma: float=1.,
        vendi: bool = False, verbose: bool = True, part: int=0, q: float=1.):
    '''
    Code for generating simulations of Alanine Dipeptide (Ala2) in vacuum
    New systems in vacuum can be simulated by adding system files to 'system' directory
    
    Script to generate a simulation are included
    
    simu-time: (in picoseconds) length of simulation
    recording-interval: (in picoseconds) the frequency at which we record states
   
    temperature: Tempurature in Kelvin 
    nu: Vendi Force Coefficient 
    stop: # of steps the Vendi Force Applied for
    gamma: RBF bandwidth parameter
    
    replicas: Number of parallel replicas used in simulation
    
    vendi: boolean flag for if the Vendi Force can be applied
    q: Order of Vendi Score (Use q=-1 if you want q='inf')
    
    output_path: Defines prefix for where files are saved
    part: If nonzero, loads the initial position from last sample in an earlier simulation (searches in output_path directory)
    
    '''

    system = 'alanine-dipeptide'

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
    if system == 'alanine-dipeptide':
        if verbose:
            print(f'Setting up the {system} system...')
        prmtop = app.AmberPrmtopFile("system/alanine-dipeptide.prmtop")
        inpcrd = app.AmberInpcrdFile("system/alanine-dipeptide.crd")
        pos = inpcrd.getPositions()
        system = prmtop.createSystem(implicitSolvent=None, constraints=app.HBonds,
                                     nonbondedCutoff=None, hydrogenMass=None)
    else:
        exit() # TO-DO : add other system

    time_step_in_fs = 2.0
    integrator_params = {"integrator": "Langevin", "friction_in_inv_ps": 1.0, "time_step_in_fs": time_step_in_fs}
    ts = [temperature for _ in range(replicas)]
    
    start = timer()

    name='Part'+str(part)

    if part==0 and vendi:
        print('Using Vendi Force')
        n_particles = len(pos)
        # Render the compute graph to a TorchScript module

        module = torch.jit.script(VendiModule(replicas, n_particles, nu=torch.tensor(nu), device=device, stop=stop, gamma=gamma, q=q))
        # Serialize the compute graph to a file
        module.save(MODEL_PATH)
        force = TorchForce(MODEL_PATH)

        simu = simu_utils.MultiTSimulation(system, ts, interface=interface,
                                       integrator_params=integrator_params, verbose=verbose, platform=platform,
                                       replicated_system_additional_forces=[force])
        simu.set_positions([pos] * replicas)
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
        if part==0:
            simu.set_positions([pos] * replicas)
        else:
            prev_name = output_path + 'Part'+str(part-1)+'.npy'
            samples = np.load(prev_name)
            pos = samples[:,-1,:,:]

            simu.set_positions(pos)

        if part==0: #Optional?
            simu.minimize_energy()

        simu.set_velocities_to_temp()

        steps = simu_utils.recording_hook_setup(simu=simu, simu_time=simu_time,
                                             recording_interval=recording_interval,
                                             output_path=output_path+name,
                                             exchange_interval=exchange_interval)

        simu.run(steps)

    end = timer()
    print(f'Simulation Took {end-start} s')
if __name__ == '__main__':
    interface = 'replicated_system'
    
    MODEL_PATH = './model.pt'

    typer.run(main)