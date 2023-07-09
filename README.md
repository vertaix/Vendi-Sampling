# Vendi Sampling for Molecular Dynamics

## Overview

This repository contains the implementation of Vendi Sampling, a method for 
increasing the efficiency and efficacy of the exploration of molecular conformation spaces. In Vendi sampling, molecular replicas are simulated in parallel and coupled via a global statistical measure, the Vendi Score, to enhance diversity.

![](PrinzPotential.gif)

<p style="text-align:center">
<em>Vendi Sampling enables faster exploration than Replica Sampling</em>
</p>

For more information, see [our pre-print on ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/64a2f0abba3e99daef73a144)

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

```
Prinz = PrinzEnergy()
E = Prinz.energy
```

Then, just initialize the positions of the particles, pass in a function for computing the log vendi-score, and then Vendi Sampling can be run!

```
replicas = 8
dim = 1
x_init = torch.rand((replicas, dim), requires_grad=True) 
samples, weights = VendiSamp(E, logvendi_loss, steps=10000, x_init=x_init)
```

In `Model_Systems/main.py` we perform hyperparameter optimization. 

### Vendi Sampling for Molecular Simulations

We provide two files: `Molecular/runAla2.py` & `Molecular/runCLN025.py` for performing Vendi & Replica Sampling systems on the two systems described in our paper.

We also provide two bash scripts for running these simulations with the conditions and hyperparameters provided in our paper. We run each simulation as a series of 1 ns simulations due to memory considerations. 

We also provide two python scripts for analyzing the results from simulations `Molecular/AnalyzeAla2.py` and `Molecular/AnalyzeCLN025.py`. These scripts produce the dihedral angle & TICA parameter Free Energy surfaces shown in our paper. 

Applying Vendi Sampling on additional Molecular systems outside of those in our paper is also simple. Systems in vacuum can use the Alanine Dipeptide simulation setup, whereas those in implicit solvent can use the Chignolin setup. All one would need to do is define the OpenMM system in either file in the marked location. For example, any system in implicit solvent can be handled using a setup such as 

```
solv = app.CharmmPsfFile('system/structure_vacuum.psf')
crds = app.PDBFile("system/structure_vaccum.pdb")
system = solv.createSystem(params, nonbondedMethod=app.NoCutoff,
                  constraints=app.HBonds,
                  hydrogenMass=4. * unit.amu,
                  implicitSolvent=app.OBC2,
                  )
```

## Citation 

```
@article{pasarkar2023vendi,
  title={Vendi Sampling For Molecular Simulations: Diversity As A Force For Faster Convergence And Better Exploration},
  author={Pasarkar, Amey and Bencomo, Gianluca and Olsson, Simon and Dieng, Adji Bousso},
  year={2023}
}
```