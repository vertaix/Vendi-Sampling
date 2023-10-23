# Vendi Sampling for Molecular Dynamics

## Overview

This repository contains the implementation of Vendi Sampling, a method for 
increasing the efficiency and efficacy of the exploration of molecular conformation spaces. In Vendi sampling, molecular replicas are simulated in parallel and coupled via a global statistical measure, the Vendi Score, to enhance diversity.

![](PrinzPotential.gif)

<p align="center">
<em>Vendi Sampling enables efficient exploration of molecular conformation spaces</em>
</p>

For more information, see [our publication in JCP](https://pubs.aip.org/aip/jcp/article/159/14/144108/2916208/Vendi-sampling-for-molecular-simulations-Diversity)

## Installation

All libaries can be installed in a conda virtual environment from the given requirements.txt file

```
conda create --name <environment_name> --file requirements.txt
```

Note: We use a modified version of the [REFORM repository](https://github.com/noegroup/reform/tree/master) for our molecular simulations. Our version is included in our repository and does not require any additional installation. 

## Usage

We provide implementations of Vendi Sampling both in Model systems (with User-defined Energy functions) and Molecular Settings. 

### Vendi Sampler for Model Systems

First, define an Energy callable function.

`Model_Systems/model_systems.py` contains classes for the Prinz & Double Well Potential functions. 

```python
Prinz = PrinzEnergy()
E = Prinz.energy
```

Then, just initialize the positions of the particles, pass in a function for computing the log vendi-score, and then Vendi Sampling can be run!

```python
logvendi_loss = lv_loss(q=1.).loss
replicas = 8
dim = 1
x_init = torch.rand((replicas, dim), requires_grad=True) 
samples, weights = VendiSamp(E, logvendi_loss, steps=10000, x_init=x_init)
```

The choice of $q$ determines the order of the Vendi Score. See [our pre-print](https://arxiv.org/abs/2310.12952) describing the behavior of the Vendi Score with different orders $q$ 

In `Model_Systems/main.py` we perform hyperparameter optimization. 

### Vendi Sampling for Molecular Simulations

We provide two files: `Molecular/runAla2.py` & `Molecular/runCLN025.py` for performing Vendi & Replica Sampling systems on the two systems described in our paper.

We also provide two bash scripts for running these simulations with the conditions and hyperparameters provided in our paper. We run each simulation as a series of 1 ns simulations due to memory considerations. 

We also provide two python scripts for analyzing the results from simulations `Molecular/AnalyzeAla2.py` and `Molecular/AnalyzeCLN025.py`. These scripts produce the dihedral angle & TICA parameter Free Energy surfaces shown in our paper. 

Applying Vendi Sampling on additional Molecular systems outside of those in our paper is also simple. Systems in vacuum can use the Alanine Dipeptide simulation setup, whereas those in implicit solvent can use the Chignolin setup. All one would need to do is define the OpenMM system in either file in the marked location. For example, any system in implicit solvent can be handled using a setup such as 

```python
solv = app.CharmmPsfFile('system/structure_vacuum.psf')
crds = app.PDBFile("system/structure_vaccum.pdb")
system = solv.createSystem(params, nonbondedMethod=app.NoCutoff,
                  constraints=app.HBonds,
                  hydrogenMass=4. * unit.amu,
                  implicitSolvent=app.OBC2,
                  )
```

## Citation 

```bibtex
@article{pasarkar2023vendi,
  title={Vendi sampling for molecular simulations: Diversity as a force for faster convergence and better exploration},
  author={Pasarkar, Amey P and Bencomo, Gianluca M and Olsson, Simon and Dieng, Adji Bousso},
  journal={The Journal of Chemical Physics},
  volume={159},
  number={14},
  year={2023},
  publisher={AIP Publishing}
}
```

```bibtex
@article{pasarkar2023cousins,
      title={Cousins Of The Vendi Score: A Family Of Similarity-Based Diversity Metrics For Science And Machine Learning}, 
      author={Pasarkar, Amey P and Dieng, Adji Bousso},
      journal={arXiv preprint arXiv:2310.12952},
      year={2023},
}
```
