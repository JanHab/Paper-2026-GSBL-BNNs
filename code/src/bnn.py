"""
bnn.py -- neural-network priors for edge-preserving inversion.

One base class holds everything the four models share (architecture, the two-stage
Adam -> L-BFGS loop, the forward operator, evaluation). A model differs from its
siblings only in its PENALTY, so a subclass overrides `penalty` and nothing else.

    Vanilla      no penalty  (plain least squares)
    GaussianBNN  ||theta||^2 / (2 s^2)
    CauchyBNN    sum log(1 + (theta/s)^2)
    GSBLBNN      hierarchical prior with a block-coordinate psi update

    from bnn import Vanilla, GaussianBNN, CauchyBNN, GSBLBNN, build_F, Problem

    prob  = Problem(x_disc, y_obs, noise, F)
    model = Vanilla(width=[1, 50, 50, 100, 1], seed=42)
    model.train(prob)
    y = model.evaluate(x_fine)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import equinox as eqx
import optax
from tqdm import tqdm


# --------------------------------------------------------------------------
# problem definition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    """The inverse problem: observe y_obs = F u(x_disc) + noise * eps."""
    x_disc: jnp.ndarray          # (n_disc, 1) reconstruction grid
    y_obs: jnp.ndarray           # (M, 1) observations
    noise: float                 # observation std; 1.0 recovers a plain MSE scale
    F: Optional[jnp.ndarray] = None   # (M, n_disc) forward operator; None = identity

    def apply(self, net):
        """Forward operator applied to a discretised field u, shape (n_disc, 1)."""
        u = jax.vmap(net)(self.x_disc)
        return u if self.F is None else self.F @ u


def build_F(obs_pts, grid_pts, kernel_width):
    """Row-normalised Gaussian blur, obs_pts x grid_pts."""
    Fq = jnp.exp(-(jnp.reshape(obs_pts, (-1, 1))
                   - jnp.reshape(grid_pts, (1, -1))) ** 2 / (2 * kernel_width ** 2))
    return Fq / jnp.sum(Fq, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# base model
# --------------------------------------------------------------------------
class BNN:
    """Base class. Subclasses override `penalty` (and optionally `psi_update`)."""

    def __init__(
            self, width, activation,
            n_epochs_pre_training, n_epochs_lbfgs,
            optimizer_pretraining, n_inner=1, # n_inner=1 is normal training, when no GSBL-BNN is used
            seed=42,
        ):
        self.width = width
        self.activation = activation
        # n_epochs_* are GRADIENT STEPS; the loop runs them n_inner at a time.
        # Integer division silently drops the remainder, so a request of 500
        # with n_inner=7 would run 497. Keep the requested count and round UP
        # to whole outer loops so the comparison across models is exact.
        self.n_steps_pre_training_requested = n_epochs_pre_training
        self.n_steps_lbfgs_requested = n_epochs_lbfgs
        if n_epochs_pre_training % n_inner or n_epochs_lbfgs % n_inner:
            import warnings
            warnings.warn(
                f"n_inner={n_inner} does not divide n_epochs_pre_training={n_epochs_pre_training} "
                f"/ n_epochs_lbfgs={n_epochs_lbfgs}; rounding UP to whole outer "
                f"loops (extra steps are recorded in loss_history).", stacklevel=2)
        self.n_epochs_pre_training = -(-n_epochs_pre_training // n_inner)   # ceil
        self.n_epochs_lbfgs = -(-n_epochs_lbfgs // n_inner)
        self.n_inner = n_inner
        self.seed = seed
        # One entry per GRADIENT STEP, index-aligned with loss_history.
        self.wall_clock_time = []
        self._t0 = None                 # set at the top of train()
        self._save_every = None         # gradient steps between checkpoints
        self._save_name = "NN"

        # ---- one seed, split ONCE into named keys; never rebind ----
        self.key = jr.PRNGKey(seed)
        self.init_key, self.data_key, self.state_key = jr.split(self.key, 3)

        self.net = self._nn_init()
        self.params, self.static = eqx.partition(self.net, eqx.is_array)
        self.theta0, self.unravel = ravel_pytree(self.params)
        self.K = self.theta0.size

        self.loss_history = []
        self.optimizer_pretraining = optimizer_pretraining

    # -- architecture --------------------------------------------------
    def _nn_init(self):
        keys = jr.split(self.init_key, len(self.width))
        layers = []
        for i in range(len(self.width) - 2):
            layers.append(eqx.nn.Linear(self.width[i], self.width[i + 1], key=keys[i]))
            layers.append(eqx.nn.Lambda(self.activation))   # uses the ARGUMENT
        layers.append(eqx.nn.Linear(self.width[-2], self.width[-1], key=keys[-1]))
        return eqx.nn.Sequential(layers)

    # -- loss: a PURE function of (params, state) ----------------------
    def penalty(self, theta, state, net):
        """Negative log prior, up to a constant. Override in a subclass."""
        return 0.0

    def data_misfit(self, params, prob):
        net = eqx.combine(params, self.static)
        y_pred = prob.apply(net)
        return jnp.sum((prob.y_obs - y_pred) ** 2) / (2 * prob.noise ** 2)

    def loss_fn(self, params, state, prob):
        net = eqx.combine(params, self.static)
        theta, _ = ravel_pytree(params)
        return self.data_misfit(params, prob) + self.penalty(theta, state, net)

    def init_state(self):
        """Extra optimisation variables (GSBL: psi). None for the simple priors."""
        return None, self.params

    def psi_update(self, theta, state, params):
        return state

    @eqx.filter_jit
    def pre_training_block(self, params, opt_state, state, optimizer, loss, update_state=True):
        def inner(carry, _):
            params, opt_state = carry
            v, g = jax.value_and_grad(loss)(params, state)
            upd, opt_state = optimizer.update(g, opt_state, params)
            return (eqx.apply_updates(params, upd), opt_state), v
        (params, opt_state), vs = jax.lax.scan(
            inner, (params, opt_state), None, length=self.n_inner)
        theta, _ = ravel_pytree(params)
        # `update_state=False` holds `state` fixed -- that is what a burn-in is.
        # It is a static bool, so each setting gets its own compiled trace.
        new_state = self.psi_update(theta, state, params) if update_state else state
        return params, opt_state, vs, new_state

    @eqx.filter_jit
    def lbfgs_block(self, params, opt_state, state, solver, loss):
        def inner(carry, _):
            params, opt_state = carry
            v, g = jax.value_and_grad(loss)(params, state)
            upd, opt_state = solver.update(
                g, opt_state, params,
                value=v, grad=g, value_fn=loss, s=state)
            return (eqx.apply_updates(params, upd), opt_state), v
        (params, opt_state), vs = jax.lax.scan(
            inner, (params, opt_state), None, length=self.n_inner)
        theta, _ = ravel_pytree(params)
        return params, opt_state, vs, self.psi_update(theta, state, params)

    # -- bookkeeping ---------------------------------------------------
    def _record(self, vs, params, state):
        """Log one loss + one timestamp per GRADIENT STEP, and checkpoint.

        The n_inner inner steps run inside a jitted `lax.scan`, so they cannot
        be timed individually; the block's elapsed time is spread evenly over
        its steps. That keeps wall_clock_time index-aligned with loss_history
        (both are per gradient step, burn-in included), so "epoch" means the
        same thing for every model regardless of n_inner.
        """
        # .tolist() forces JAX to finish the block, so the stamp below is not
        # taken while the computation is still queued asynchronously.
        losses = vs.flatten().tolist()
        if self._t0 is None:            # init_state() called outside train()
            self._t0 = time.time()
        t = time.time() - self._t0
        t_prev = self.wall_clock_time[-1] if self.wall_clock_time else 0.0
        n = len(losses)
        self.wall_clock_time.extend( # Staircasing the inner epochs to have a continuous wall-clock time series, even though the inner steps are not timed individually.
            t_prev + (t - t_prev) * (i + 1) / n for i in range(n))

        step0 = len(self.loss_history)
        self.loss_history.extend(losses)
        step1 = len(self.loss_history)

        # Save when this block crossed a multiple of save_every. A block is
        # atomic, so with n_inner > save_every this fires once per block --
        # the intermediate params inside the scan are never materialised.
        if self._save_every and step0 // self._save_every != step1 // self._save_every:
            self._checkpoint(params, state, step1)

    def _checkpoint(self, params, state, step):
        self.params, self.state = params, state
        self.net = eqx.combine(params, self.static)   # save() reads self.net
        self.save(f"{self._save_name}_epoch{step}")

    # -- training ------------------------------------------------------
    def train(
        self,
        prob,
        save_every=None,  # save every N gradient steps (inner+outer); None = no saving
        save_name="NN"  # filename prefix for saving; None = default):
    ):
        self._save_every, self._save_name = save_every, save_name
        # Resuming after load() continues the clock instead of restarting it.
        self._t0 = time.time() - (self.wall_clock_time[-1] if self.wall_clock_time else 0.0)

        state, params = self.init_state()   # Use initialization distributions
        self.params, self.state = params, state # update
        # params = self.params

        # jitted closures: `prob` and `self` are captured by the CLOSURE, not
        # passed as jit arguments, so nothing stale is baked into the trace.
        def loss(p, s):
            return self.loss_fn(p, s, prob)

        if self.n_epochs_pre_training > 0:
            optimizer = self.optimizer_pretraining
            opt_state = optimizer.init(params)

            for _ in tqdm(range(self.n_epochs_pre_training), desc=f"{type(self).__name__} Pre-Training"):
                params, opt_state, vs, state = self.pre_training_block(params, opt_state, state, optimizer, loss)
                self._record(vs, params, state)

        if self.n_epochs_lbfgs > 0:
            solver = optax.lbfgs()
            opt_state = solver.init(params)

            for _ in tqdm(range(self.n_epochs_lbfgs), desc=f"{type(self).__name__} L-BFGS"):
                params, opt_state, vs, state = self.lbfgs_block(params, opt_state, state, solver, loss)
                self._record(vs, params, state)

        # commit the result exactly once, at the end
        self.params, self.state = params, state
        self.net = eqx.combine(params, self.static)
        if save_every:
            self.save(f"{save_name}_epoch{len(self.loss_history)}")   # final state
        return self

    # -- evaluation ----------------------------------------------------
    def evaluate(self, x_eval):
        return jax.vmap(self.net)(x_eval)

    def rel_l1(self, x_eval, y_true):
        y = self.evaluate(x_eval)
        return float(jnp.sum(jnp.abs(y - y_true)) / jnp.sum(jnp.abs(y_true)))

    # -- Storing ------------------------------------------------------
    def save(self, filename):
        """Save the trained model to a plain-text .csv file."""
        params, _ = eqx.partition(self.net, eqx.is_array)
        theta, _ = ravel_pytree(params)
        columns = (theta.tolist(), self.loss_history, self.wall_clock_time)
        formats = ('%.9g', '%.6g', '%.6g')
        with open(os.path.splitext(filename)[0] + '.csv', 'w') as f:
            f.write('theta,loss_history,wall_clock_time\n')
            for i in range(max(len(c) for c in columns)):
                f.write(','.join(fmt % c[i] if i < len(c) else '' for c, fmt in zip(columns, formats)) + '\n')

    def load(self, filename):
        """Load the trained model from a .csv file written by `save`."""
        stem = os.path.splitext(filename)[0]
        with open(stem + '.csv') as f:
            rows = [line.rstrip('\n').split(',') for line in f][1:]  # skip header
        theta, loss_history, wall_clock_time = (
            [float(r[j]) for r in rows if r[j]] for j in range(3))
        params = self.unravel(jnp.asarray(theta, dtype=self.theta0.dtype))
        self.params, self.loss_history, self.wall_clock_time = params, loss_history, wall_clock_time
        self.net = eqx.combine(params, self.static)
