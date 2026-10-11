# GSBL-BNNs

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JanHab/paper-2026-gsbl-bnns/blob/main/LICENSE)
ToDo: DOI, docs??
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23184383.svg)](https://doi.org/10.5281/zenodo.23184383)

Reproducibility repository for the article *Edge-preserving Bayesian neural networks via hierarchical sparsity-promoting priors*

## Introduction

Recovering piecewise-smooth functions with jump discontinuities from noisy, indirect, and incomplete data is a central task in computational inverse problems.
Bayesian neural networks (BNNs) with heavy-tailed weight priors have emerged as effective edge-preserving priors for such problems.
However, posterior inference for these models remains computationally challenging, and existing approaches settle for point estimates obtained by directly optimizing heavy-tailed log-densities.
We propose GSBL-BNNs, which model the neural network parameters hierarchically using sparsity-promoting priors from generalized sparse Bayesian learning (GSBL). 
The hierarchical formulation recovers several popular marginal priors, including Laplace and Student's $t$ (including Cauchy).
A key modeling ingredient, novel among edge-preserving BNNs and native to the hierarchical formulation, couples each neuron's bias to its weights through a shared hyperparameter, so that the induced jump discontinuities are free to occur anywhere in the domain while the conditionally Gaussian structure underlying efficient inference is preserved.
Furthermore, we develop a block-coordinate descent (BCD) algorithm for maximum a posteriori (MAP) estimation. 
This BCD approach alternates between standard network training with iteratively reweighted $\ell^2$-regularization and exact hyperparameter updates that are available in closed form for gamma and inverse gamma hyperpriors. 
Building on the BCD algorithm, we perform approximate Bayesian inference and uncertainty quantification (UQ) using deep ensembles. 
Computational experiments---including signal denoising, deblurring, and inverting a two-dimensional hyperbolic conservation law---demonstrate that GSBL-BNNs deliver accurate edge-preserving reconstructions together with built-in UQ.
Compared with Gaussian- and Cauchy-BNNs, they yield more accurate predictive means and avoid overconfident uncertainty estimates.

## Installation

The code was developed and tested for Python/3.12.3 and Python/3.13.5.

```bash
git clone git@github.com:JanHab/Paper-2026-GSBL-BNNs.git
cd Paper-2026-GSBL-BNNs
pip install -r requirements.txt
```

## Usage

### Data

The trained models and all results (`code/out/`, about 10 GB) are not part of this repository. They are archived on Zenodo [10.5281/zenodo.23184383](https://doi.org/10.5281/zenodo.23184383). Download the archive and unpack it so that the folder `code/out/` exists. The notebooks and the figure scripts below read from there.

### Generating the results

We executed our calculations on the [Arrhenius](https://www.naiss.se/resource/arrhenius/) supercomputer.
Therefore, we provide code to automatically generate Slurm scripts, which are then executed.
The files `code/SlurmController_1D.py`and `code/SlurmController_KPP.py` generate all our data.
If you don't have access to any Slurm-based setup, you can easily comment out the `createsbatch()` executions in `code/SlurmController_1D.py`, and `code/SlurmController_KPP.py`, and uncomment the `os.system()` lines.
Then, all calculations will be executed on your local computer.
While the 2D example might be too heavy for a Laptop, the 1D examples should work fine.
However, we ran calculations for many different parameters.
You might want to reduce the number of different parameters.

The controllers call the three run scripts with positional arguments. You can also call them directly.

**1D examples** (signal denoising and deblurring):

```bash
python code/Run_PiecewiseConstant_Noisy.py <N_ENSEMBLES> <MODEL> <OPTIMIZER> <KAPPA> <VARTHETA1> <INITIALIZER>
python code/Run_PiecewiseSmooth_Blur.py    <N_ENSEMBLES> <MODEL> <OPTIMIZER> <KAPPA> <VARTHETA1> <INITIALIZER>
```

| Argument | Meaning |
|---|---|
| `N_ENSEMBLES` | Number of ensemble members (50 in the paper) |
| `MODEL` | `Vanilla`, `Gaussian`, `Cauchy`, `Cauchy_Extended`, `GSBL_Independent`, `GSBL_Shared`, or `GSBL_Shared_CombinedOptimization` |
| `OPTIMIZER` | Pre-training optimizer: `muon` or `adam` (L-BFGS follows in both cases) |
| `KAPPA`, `VARTHETA1` | GSBL hyperparameters $\kappa$ and $\vartheta^{(1)}$ (must be given, ignored by non-GSBL models) |
| `INITIALIZER` | `Default` (Equinox default), `Gaussian` or `Cauchy` (draw from the respective prior), `GG` (draw from the generalized gamma hierarchy, GSBL models) |

Each call loops over the noise levels 0.01, 0.05, 0.1, 0.2 and, for the GSBL models, over $\beta \in \{0.5, 1.0\}$. Example used in the paper:

```bash
python code/Run_PiecewiseConstant_Noisy.py 50 GSBL_Shared muon 10.0 100.0 GG
```

## Citation

If you find these results useful or use the implementations provided here, please cite the corresponding article.

```bibtex
@article{habscheid_2026_gsblbnns,
  author       = {Habscheid, Jan and
                  Glaubitz, Jan},
  title        = {Edge-preserving Bayesian neural networks via hierarchical sparsity-promoting priors},
  month        = oct,
  year         = 2026,
}
```

If you use the implementations provided here, please also cite this repository as

```bibtex
@software{jan_habscheid_2026_23184383,
  author       = {Jan Habscheid and
                  Glaubitz, Jan},
  title        = {Reproducibility repository for the article "Edge-
                   preserving Bayesian neural networks via
                   hierarchical sparsity-promoting priors"
                  },
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.23184383},
  url          = {https://doi.org/10.5281/zenodo.23184383},
}
```

## Contact

- **Jan Habscheid** 
    - TIMA - Applied Mathematics
    - Linköping University
    - [Jan.Habscheid@liu.se](mailto:Jan.Habscheid@liu.se)
- **Jan Glaubitz**
    - TIMA - Applied Mathematics
    - Linköping University
    - [Jan.Glaubitz@liu.se](mailto:Jan.Glaubitz@liu.se)