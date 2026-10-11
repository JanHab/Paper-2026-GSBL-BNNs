
import equinox as eqx
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
import time
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

from src.bnn import BNN

# Extension of BNN with hierarchical GSBL prior

class GSBLBNN(BNN):
    """
    Hierarchical sparsity-promoting prior.

    shared_bias=True  -> one psi per NEURON BLOCK (incoming weights + its bias), the variant that couples a neuron's kink location to its amplitude.
    shared_bias=False -> one psi per PARAMETER.
    """

    def __init__(self, *args, vartheta1=100.0, kappa=1.0, c_bias=2.0,
                 r=-1.0, beta=0.5, shared_bias=True, n_inner=10, initializer='GG', **kwargs):
        kwargs["n_inner"] = n_inner
        super().__init__(*args, **kwargs)
        self.vartheta1, self.kappa, self.c_bias = vartheta1, kappa, c_bias
        self.r, self.beta = r, beta
        self.alpha = 2 * abs(r) * beta
        self.shared_bias = shared_bias
        self.psi_norm_history = []   # only used by GSBLBNN
        self._build_blocks()
        self.initializer = initializer

    # -- bookkeeping ---------------------------------------------------
    def _record(self, vs, params, state):
        """Log one loss + one timestamp per GRADIENT STEP, and checkpoint.

        The n_inner inner steps run inside a jitted `lax.scan`, so they cannot
        be timed individually; the block's elapsed time is spread evenly over
        its steps. That keeps wall_clock_time index-aligned with loss_history
        (both are per gradient step), so "epoch" means the
        same thing for every model regardless of n_inner.
        """
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
        self.psi_norm_history.extend(jnp.linalg.norm(state).tolist() for _ in range(n))  # psi is the state for GSBLBNN
        step1 = len(self.loss_history)

        # Save when this block crossed a multiple of save_every.
        if self._save_every and step0 // self._save_every != step1 // self._save_every:
            self._checkpoint(params, state, step1)

    def _build_blocks(self):
        block_id, c2, s_g, offset = [], [], [], 0
        for lyr in self.net.layers:
            if isinstance(lyr, eqx.nn.Linear):
                n_out, n_in = lyr.weight.shape
                if self.shared_bias:
                    ids = jnp.arange(n_out) + offset
                    block_id += [jnp.repeat(ids, n_in), ids]
                    c2 += [jnp.ones(n_out * n_in), jnp.full((n_out,), self.c_bias ** 2)]
                    s_g += [jnp.full((n_out,), n_in + 1)]
                    offset += n_out
                else:
                    n = n_out * n_in + n_out
                    block_id += [jnp.arange(n) + offset]
                    c2 += [jnp.ones(n)]
                    s_g += [jnp.ones(n)]
                    offset += n
        self.block_id = jnp.concatenate(block_id)
        self.c2 = jnp.concatenate(c2)
        self.s_g = jnp.concatenate(s_g)
        self.G = offset
        self.vartheta = self._vartheta_vector()
        assert self.block_id.size == self.K
        assert self.vartheta.size == self.G == self.s_g.size

    def _vartheta_vector(self):
        blocks = []
        for li, lyr in enumerate(self.net.layers):
            if isinstance(lyr, eqx.nn.Linear):
                n_out, n_in = lyr.weight.shape
                if li == 0:
                    v = self.vartheta1
                elif self.alpha >= 2:
                    v = self.kappa * n_in ** (-1.0)
                else:
                    v = self.kappa * n_in ** (-2.0 / self.alpha)
                n = n_out if self.shared_bias else n_out * n_in + n_out
                blocks.append(jnp.full((n,), v))
        return jnp.concatenate(blocks)

    def init_state(self):
        """Draw psi from the prior AND re-draw theta from the conditional prior."""
        psi_key, theta_key = jr.split(self.state_key)
        params, _ = eqx.partition(self.net, eqx.is_array)
        _, unravel = ravel_pytree(params)

        # Generalized Gamma initialization
        if self.initializer == 'GG':
            psi_tilde = self.vartheta * jr.gamma(psi_key, self.beta, shape=(self.G,)) ** (1 / self.r)
            params  = unravel(jnp.sqrt(self.c2 * psi_tilde[self.block_id]) * jr.normal(theta_key, (self.K,)))
        elif self.initializer == 'Normal':
            params = unravel(jr.normal(theta_key, (self.K,)))
        elif self.initializer == 'Glorot':
            # ToDo: Take care of different layers
            glorot_std = jnp.sqrt(2.0 / (self.net.layers[0].weight.shape[0] + self.net.layers[0].weight.shape[1]))
            params = unravel(glorot_std * jr.normal(theta_key, (self.K,)))
        elif self.initializer == 'He':
            # ToDo: Take care of different layers
            he_std = jnp.sqrt(2.0 / self.net.layers[1].weight.shape[0])
            params = unravel(he_std * jr.normal(theta_key, (self.K,)))
        elif self.initializer == 'Xavier':
            xavier_std = jnp.sqrt(1.0 / self.net.layers[0].weight.shape[1])
            params = unravel(xavier_std * jr.normal(theta_key, (self.K,)))
        elif self.initializer == 'LeCun':
            lecun_std = jnp.sqrt(1.0 / self.net.layers[0].weight.shape[1])
            params = unravel(lecun_std * jr.normal(theta_key, (self.K,)))
        elif self.initializer == 'Default':
            # Use the default initialization of the Equinox Linear layers
            params = unravel(jr.normal(theta_key, (self.K,)))
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")
        theta, _ = ravel_pytree(params)
        psi = self.psi_update(theta, jnp.copy(self.vartheta), params)  # initial psi update

        self.state, self.params = psi, params

        # commit the result to the net
        self.net = eqx.combine(params, self.static)

        return psi, params

    def penalty(self, theta, psi, net):
        return 0.5 * jnp.sum(theta ** 2 / (self.c2 * psi[self.block_id]))

    def psi_update(self, theta, psi, params):
        """Exact block minimiser for r = -1."""
        assert self.r == -1.0 or self.r == 1.0, "psi_update is only implemented for r = -1 or r = 1"
        norm_c2 = jax.ops.segment_sum(
            theta ** 2 / self.c2, self.block_id,
            num_segments=self.G
        )
        if self.r == -1.0:
            return (self.vartheta + norm_c2 / 2) / (self.beta + 1 + self.s_g / 2)
        elif self.r == 1.0:
            eta_l = self.r * self.beta - 1 - self.s_g / 2
            return self.vartheta / 2 * (
                eta_l + jnp.sqrt(eta_l ** 2 + 2 * norm_c2 / self.vartheta)
            )



class GSBLBNN_CombinedOptimization(GSBLBNN):
    """
    GSBLBNN with theta and psi optimised JOINTLY.

    One optimiser (Adam/Muon, then L-BFGS) steps in (theta, psi) on the full
    negative log posterior. There is no inner/outer split and no closed-form
    psi update: `train` is overridden with a plain step loop, so nothing in
    BNN or GSBLBNN changes. psi is optimised in log-space, phi = log psi, so it
    stays positive and every block lives on a comparable scale; the objective
    is unchanged, only the coordinates are, so the minimiser is the same.
    """

    def __init__(self, *args, n_inner=1, **kwargs):
        assert n_inner == 1, "combined optimisation has no inner/outer split; use n_inner=1"
        super().__init__(*args, n_inner=n_inner, **kwargs)
        # Coefficient of -log(psi) in the joint objective: hyperprior exponent
        # plus the log-normaliser of the conditional Gaussian prior on theta.
        self.eta = self.r * self.beta - 1 - self.s_g / 2
        assert self.r < 0 or bool(jnp.all(self.eta > 0)), (
            "for r > 0 the joint objective is unbounded below (theta, psi -> 0) "
            "unless r * beta > 1 + s_g / 2; use r < 0 or raise beta")

    # -- loss ----------------------------------------------------------
    def psi_terms(self, psi):
        """Part of the negative log posterior that depends on psi only."""
        return jnp.sum((psi / self.vartheta) ** self.r - self.eta * jnp.log(psi))

    def loss_fn(self, params, psi, prob):
        return super().loss_fn(params, psi, prob) + self.psi_terms(psi)

    # -- joint optimisation --------------------------------------------
    # The optimiser sees ONE pytree c = (params, log psi).
    @staticmethod
    def _pack(params, psi):
        return params, jnp.log(psi)

    @staticmethod
    def _unpack(c):
        params, phi = c
        return params, jnp.exp(phi)

    @eqx.filter_jit
    def _step(self, c, opt_state, optimizer, loss):
        v, g = jax.value_and_grad(loss)(c)
        # value / grad / value_fn feed the L-BFGS line search; Adam and Muon ignore them.
        upd, opt_state = optimizer.update(g, opt_state, c, value=v, grad=g, value_fn=loss)
        return eqx.apply_updates(c, upd), opt_state, v

    def _run(self, c, optimizer, loss, n_steps, desc):
        optimizer = optax.with_extra_args_support(optimizer)
        opt_state = optimizer.init(c)
        for _ in tqdm(range(n_steps), desc=f"{type(self).__name__} {desc}"):
            c, opt_state, v = self._step(c, opt_state, optimizer, loss)
            self._record(v, *self._unpack(c))
        return c

    def train(self, prob, save_every=None, save_name="NN"):
        self._save_every, self._save_name = save_every, save_name
        # Resuming after load() continues the clock instead of restarting it.
        self._t0 = time.time() - (self.wall_clock_time[-1] if self.wall_clock_time else 0.0)

        psi, params = self.init_state()
        c = self._pack(params, psi)

        def loss(c):
            return self.loss_fn(*self._unpack(c), prob)

        if self.n_epochs_pre_training > 0:
            c = self._run(c, self.optimizer_pretraining, loss, self.n_epochs_pre_training, "Pre-Training")
        if self.n_epochs_lbfgs > 0:
            c = self._run(c, optax.lbfgs(), loss, self.n_epochs_lbfgs, "L-BFGS")

        self.params, self.state = self._unpack(c)
        self.net = eqx.combine(self.params, self.static)
        if save_every:
            self.save(f"{save_name}_epoch{len(self.loss_history)}")
        return self
