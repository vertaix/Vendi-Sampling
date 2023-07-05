import torch

class PrinzEnergy:
    #Boundary at x=0
    def __init__(self, temperature=300.15):
        self.temperature = temperature
    def energy(self, x):
        return 4. * (x ** 8. + 0.8 * torch.exp(-80. * x ** 2.) +
                     0.2 * torch.exp(-80. * (x - 0.5) ** 2.) +
                     0.5 * torch.exp(-40. * (x + 0.5) ** 2.))

class DoubleWell(object):

    params_default = {'a4' : 1.0,
                      'a2' : 6.0,
                      'a1' : 1.0,
                      'k' : 1.0,
                      'dim' : 2}

    def __init__(self, params=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.dim = self.params['dim']


    def energy(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        dimer_energy = 10.7524+self.params['a4'] * x[:, 0] ** 4 - self.params['a2'] * x[:, 0] ** 2 + self.params['a1'] * x[:, 0]
        oscillator_energy = 0.0
        if self.dim == 2:
            oscillator_energy = (self.params['k'] / 2.0) * x[:, 1] ** 2
        if self.dim > 2:
            oscillator_energy = torch.sum((self.params['k'] / 2.0) * x[:, 1:] ** 2, axis=1)
        return  dimer_energy + oscillator_energy