import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

from src.scalar2d_dg import ScalarDG2D, KPP, burgers2d
from src.bnn import Problem
from src.gsbl import GSBLBNN

print("jax", jax.__version__, "| device:", jax.devices()[0])



NX = NY = int(sys.argv[5])  # number of cells in x and y direction
REFINE_OBSERVATIONS = 8 
T_END = 1.0
CFL = 0.9

M = int(sys.argv[4])  # number of observations´
NOISE = 0.5                 # 4.9% of the data range, cf. 0.05 on 1D Burgers
SEED = int(sys.argv[1])
OBSERVATION_SEED = 1234

U_LO, U_HI = 0.25 * np.pi, 3.5 * np.pi   # network output is mapped affinely onto this

# flux=burgers2d(u_ref=...) switches the equation and nothing else.
solver = ScalarDG2D(
    flux=KPP, nx=NX, ny=NY, t_end=T_END, cfl=CFL,
    bc="transmissive", remat=True
)

solver_fine = ScalarDG2D(
    flux=KPP, nx=NX*REFINE_OBSERVATIONS, ny=NY*REFINE_OBSERVATIONS, t_end=T_END, cfl=CFL,
    bc="transmissive", remat=True
)

XY_NODES_J = jnp.asarray(solver.xy_flat)                 # (4*NX*NY, 2) network input
OBS_IDX, x_obs, y_obs_pts = solver.observation_indices(M)

OBS_IDX_FINE, x_obs_fine, y_obs_pts_fine = solver_fine.observation_indices(M, nx_ref=NX, ny_ref=NY)

ic_true = solver_fine.disk()          # 14 pi / 4 on the unit disk, pi / 4 outside

solver_fine.solve(ic_true).block_until_ready()      # compile before timing
t0 = time.perf_counter()
d_true = solver_fine.solve(ic_true)
d_true.block_until_ready()
FORWARD_SECONDS = time.perf_counter() - t0

u_init = np.asarray(solver_fine.image(solver_fine.limit(ic_true)))
u_sol = np.asarray(solver_fine.image(d_true))

y_obs = jnp.asarray(u_sol.ravel()[OBS_IDX_FINE], dtype=jnp.float32) \
        + NOISE * jr.normal(jr.PRNGKey(OBSERVATION_SEED), (M,), dtype=jnp.float32)

xs, ys = solver.x_sample, solver.y_sample
xs_f, ys_f = solver_fine.x_sample, solver_fine.y_sample
levels = np.linspace(U_LO, U_HI, 31)

np.testing.assert_array_equal(x_obs, x_obs_fine)
np.testing.assert_array_equal(y_obs_pts, y_obs_pts_fine)


# Inverse Problem
# width = [2, 64, 64, 64, 64, 1]      # <- 2 inputs (x, y)
net_arc_identifier = int(sys.argv[7])
if net_arc_identifier == 0:
    width = [2, 64, 64, 64, 64, 1]
elif net_arc_identifier == 1:
    width = [2, 128, 128, 128, 128, 1]
elif net_arc_identifier == 2:
    width = [2, 80, 80, 1000, 1]
elif net_arc_identifier == 3:
    width = [2, 100, 100, 1200, 1]
elif net_arc_identifier == 4:
    width = [2, 256, 256, 256, 256, 1]
activation = jax.nn.tanh

N_EPOCHS_PRE = 2_500
N_EPOCHS_LBFGS = 12_500
n_inner = 10
lr_schedule = 1e-3

optimizer_pretraining = optax.contrib.muon(lr_schedule)

vartheta1 = float(sys.argv[6])
kappa = float(sys.argv[2])
c_bias = 2.0
r = -1.0
beta = float(sys.argv[3])
initializer = 'GG'

# Saving
dir = f'code/out/KPP2D/NX{NX}_M{M}_vartheta1_{vartheta1}_beta_{beta}_kappa_{kappa}_net_arc_{net_arc_identifier}/Ensemble_{SEED}/'
if not os.path.exists(dir):
    os.makedirs(dir)
save_every = N_EPOCHS_PRE

model = GSBLBNN(
    width=width, activation=activation, seed=SEED,
    optimizer_pretraining=optimizer_pretraining,
    n_epochs_pre_training=N_EPOCHS_PRE, n_epochs_lbfgs=N_EPOCHS_LBFGS,
    vartheta1=vartheta1, kappa=kappa, c_bias=c_bias,
    shared_bias=True, n_inner=n_inner,
    r=r, beta=beta,
    initializer=initializer,
)



@dataclass(frozen=True)
class InverseKPPProblem(Problem):
    """y_obs = observe(solve(u_net(x, y))) + noise * eps.

    The network is evaluated on the flattened (4*NX*NY, 2) node coordinates and
    reshaped to the solver's (NY, NX, 2, 2) layout, then mapped affinely onto
    [U_LO, U_HI]. Because the nodes either side of an interface share a coordinate,
    a continuous network produces a continuous initial condition -- and the truth
    here is a disk-shaped jump, which is exactly what the GSBL prior is meant to
    let the network resolve.
    """

    def apply(self, net):
        u = jax.vmap(net)(XY_NODES_J).reshape(NY, NX, 2, 2)
        ic = (U_LO + (U_HI - U_LO) * u).astype(jnp.float32)
        return solver.image(solver.solve(ic)).ravel()[OBS_IDX]


prob = InverseKPPProblem(x_disc=XY_NODES_J, y_obs=y_obs, noise=NOISE)



t0 = time.time()
model.train(prob, save_every=save_every, save_name=dir)
model.save(dir + "model")
TRAIN_SECONDS = time.time() - t0
print(
    f"{TRAIN_SECONDS:.1f} s for {len(model.loss_history)} epochs "
    f"({TRAIN_SECONDS / len(model.loss_history):.2f} s/epoch)"
)



epochs = np.arange(len(model.loss_history))

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].semilogy(epochs, model.loss_history)
axs[1].plot(epochs, model.psi_norm_history)
axs[0].set_ylabel("loss"); axs[1].set_ylabel(r"$\parallel \psi \parallel$")
for ax in axs:
    ax.set_xlabel("epochs"); ax.grid()
fig.tight_layout(); fig.savefig(dir + "training_curves.pdf")

print(f"loss {model.loss_history[0]:.1f} -> {model.loss_history[-1]:.1f}")



ic_rec = (U_LO + (U_HI - U_LO)
          * jax.vmap(model.net)(XY_NODES_J).reshape(NY, NX, 2, 2)).astype(jnp.float32)
d_rec = solver.solve(ic_rec)
u_ic_rec = np.asarray(solver.image(ic_rec))
u_rec = np.asarray(solver.image(d_rec))
y_rec = u_rec.ravel()[OBS_IDX]

print("data misfit:", float(jnp.sum((y_obs - y_rec) ** 2) / (2 * NOISE ** 2)))

fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
for ax, gx, gy, field, title in (
    (axs[0], xs_f, ys_f, u_init, "true initial condition"),
    (axs[1], xs,   ys,   u_ic_rec, "recovered initial condition"),
    (axs[2], xs_f, ys_f, u_sol, f"true $u(\\cdot,{T_END})$"),
    (axs[3], xs,   ys,   u_rec, f"recovered $u(\\cdot,{T_END})$")
    ):
    m = ax.pcolormesh(
        gx, gy, field, cmap="viridis", shading="nearest",
        vmin=U_LO, vmax=U_HI, rasterized=True
    )
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_title(title)
axs[0].set_ylabel("$y$")
axs[2].scatter(
    x_obs, y_obs_pts, facecolors="none", edgecolors="red",
    s=6, linewidths=0.4, zorder=5
)
fig.colorbar(m, ax=axs, fraction=0.02, pad=0.01, label="$u$")
fig.savefig(dir + "recovered_fields.pdf")