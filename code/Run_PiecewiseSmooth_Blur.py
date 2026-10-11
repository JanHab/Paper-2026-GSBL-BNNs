
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import optax
import matplotlib.pyplot as plt

import sys # for command line arguments
import os

from src.bnn import Problem, build_F
from src.vanilla import Vanilla
from src.gaussian import GaussianBNN
from src.cauchy import CauchyBNN, CauchyBNNExtended
from src.gsbl import GSBLBNN, GSBLBNN_CombinedOptimization



######################
# Problem setup ######
######################
def f(x):
    test_fcn_smooth = lambda x: jnp.sin(x * 2 * jnp.pi)
    y = jnp.where(x < 0.2, test_fcn_smooth(x), 
            jnp.where(x < 0.7, test_fcn_smooth(x+0.55), -0.5)
        )
    return y

# Parameter
M = 50
n_disc = 128
kernel_width = 0.03
noise_level = [0.01, 0.05, 0.1, 0.2]

for noise in noise_level:

    SEED = 42

    # Data generation -- the data key is INDEPENDENT of the network init key
    data_key = jr.PRNGKey(SEED + 1000)

    x_disc = jnp.linspace(0.0, 1.0, n_disc).reshape(-1, 1)
    x_obs = jnp.linspace(0.0, 1.0, M).reshape(-1, 1)

    F = build_F(x_obs, x_disc, kernel_width)
    y_true = f(x_disc)
    y_obs = F @ y_true + noise * jr.normal(data_key, (M, 1))

    prob = Problem(x_disc=x_disc, y_obs=y_obs, noise=noise, F=F)

    # For plotting and evaluation only
    x_fine = jnp.linspace(0.0, 1.0, 1000).reshape(-1, 1)
    y_fine = f(x_fine)
    y_blur_fine = build_F(x_fine, x_disc, kernel_width) @ y_true




    #######################
    # Hyperparameter ######
    #######################
    # Common
    width = [1, 64, 64, 64, 64, 1]
    activation = jax.nn.tanh
    lr_schedule = 1e-3
    optimizer_pre = sys.argv[3]
    if optimizer_pre == 'muon':
        optimizer_pretraining = optax.contrib.muon(lr_schedule)
    elif optimizer_pre == 'adam':
        optimizer_pretraining = optax.adam(lr_schedule)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_pre}. Must be one of ['muon', 'adam'].")
    N_EPOCHS_PRE = 500
    N_EPOCHS_LBFGS = 1_500
    N_ENSEMBLES = int(sys.argv[1])  # Number of ensembles passed as command line argument

    common = dict(
        width=width, activation=activation, #seed=SEED,
        optimizer_pretraining=optimizer_pretraining,
        n_epochs_pre_training=N_EPOCHS_PRE, n_epochs_lbfgs=N_EPOCHS_LBFGS
    )

    # GSBL-specific
    vartheta1=float(sys.argv[5])
    kappa=float(sys.argv[4])
    c_bias=2.0
    n_inner=10
    r=-1.0
    beta_vec = [0.5, 1.0]
    INITIALIZER=sys.argv[6]  # Initializer passed as command line argument

    # Save directory
    dir = f'code/out/PiecewiseSmooth_Blur/'
    # ! save_every = 100

    model_string = sys.argv[2]  # Model name passed as command line argument
    for beta in beta_vec:
        if model_string not in ['Vanilla', 'Gaussian', 'Cauchy', 'Cauchy_Extended', 'GSBL_Independent', 'GSBL_Shared', 'GSBL_Shared_CombinedOptimization']:
            raise ValueError(f"Invalid model name: {model_string}. Must be one of ['Vanilla', 'Gaussian', 'Cauchy', 'Cauchy_Extended', 'GSBL_Independent', 'GSBL_Shared', 'GSBL_Shared_CombinedOptimization'].")
        elif beta == 1.0 and model_string not in ['GSBL_Independent', 'GSBL_Shared']:
            continue # Skip beta=1.0 for non-GSBL models
        elif model_string == 'Vanilla':
            models = [Vanilla(**{**common}, seed=SEED+i) for i in range(N_ENSEMBLES)]
        elif model_string == 'Gaussian':
            models = [GaussianBNN(**{**common}, gauss_std=1.0, initializer=INITIALIZER, seed=SEED+i) for i in range(N_ENSEMBLES)]
        elif model_string == 'Cauchy':
            models = [CauchyBNN(**{**common}, cauchy_scale=0.1, initializer=INITIALIZER, seed=SEED+i) for i in range(N_ENSEMBLES)]
        elif model_string == 'Cauchy_Extended':
            models = [CauchyBNNExtended(**{**common}, vartheta1=vartheta1, kappa=kappa, r=r, beta=beta, initializer=INITIALIZER, seed=SEED+i) for i in range(N_ENSEMBLES)]
        elif model_string == 'GSBL_Independent':
            models = [GSBLBNN(**{**common},
                                vartheta1=vartheta1,
                                kappa=kappa,
                                c_bias=c_bias, shared_bias=False, n_inner=n_inner,
                                r=r, beta=beta, seed=SEED+i, initializer=INITIALIZER) for i in range(N_ENSEMBLES)]
        elif model_string == 'GSBL_Shared':
            models = [GSBLBNN(**{**common},
                                vartheta1=vartheta1,
                                kappa=kappa,
                                c_bias=c_bias, shared_bias=True, n_inner=n_inner,
                                r=r, beta=beta, seed=SEED+i, initializer=INITIALIZER) for i in range(N_ENSEMBLES)]
        elif model_string == 'GSBL_Shared_CombinedOptimization':
            models = [GSBLBNN_CombinedOptimization(**{**common},
                                vartheta1=vartheta1,
                                kappa=kappa,
                                c_bias=c_bias, shared_bias=True, n_inner=1,
                                initializer=INITIALIZER,
                                r=r, beta=beta, seed=SEED+i) for i in range(N_ENSEMBLES)]

        for i, model in enumerate(models):
            if model_string in ['GSBL_Independent', 'GSBL_Shared']:
                dir_ensemble = os.path.join(dir, f'{model_string}/{INITIALIZER}/{optimizer_pre}_kappa{kappa}_beta{beta}_vartheta1_{vartheta1}_noise_{noise}/Ensemble_{i}/')
            else:
                dir_ensemble = os.path.join(dir, f'{model_string}/{INITIALIZER}/{optimizer_pre}_noise_{noise}/Ensemble_{i}/')
            if not os.path.exists(dir_ensemble):
                os.makedirs(dir_ensemble)

            print(f'Training {model_string} model {i+1}/{N_ENSEMBLES}...')
            model.train(prob) #!, save_every=save_every, save_name=dir_ensemble)
            model.save(os.path.join(dir_ensemble, f'{model_string}_model_{i}'))
            jax.clear_caches() # to avoid memory issues on the cluster
            