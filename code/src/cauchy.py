
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree

from src.bnn import BNN

# Extension of BNN with Cauchy prior

class CauchyBNN(BNN):
    """theta ~ Cauchy(0, s):  penalty = sum log(1 + (theta/s)^2)."""

    def __init__(self, *args, cauchy_scale=0.1, initializer='Default', **kwargs):
        super().__init__(*args, **kwargs)
        self.cauchy_scale = cauchy_scale
        self.initializer = initializer

    def penalty(self, theta, state, net):
        # Cauchy prior on weights
        regularization = 0.0
        for layer in net.layers:
            if isinstance(layer, eqx.nn.Linear):
                regularization += jnp.sum(jnp.log(1 + (layer.weight / self.cauchy_scale) ** 2)) # Cauchy prior regularization
                regularization += jnp.sum(jnp.log(1 + (layer.bias / self.cauchy_scale) ** 2)) # Cauchy prior regularization
        return regularization

    def init_state(self):
        """Draw theta from the Cauchy prior."""
        _, theta_key = jr.split(self.state_key)
        params, _ = eqx.partition(self.net, eqx.is_array)
        _, unravel = ravel_pytree(params)

        if self.initializer == 'Cauchy':  
            # Initialize parameters from Cauchy prior
            params = unravel(jr.cauchy(theta_key, (self.K,)) * self.cauchy_scale)

        self.params = params

        # commit the result to the net
        self.net = eqx.combine(params, self.static)

        return None, params

class CauchyBNNExtended(CauchyBNN):
    """Extension of CauchyBNN where the prior distribution differs for the first layer to match the implementation of the GSBL BNNs."""

    def __init__(self, *args, vartheta1=100, kappa=1.0, r=-1.0, beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.vartheta1 = vartheta1
        self.kappa = kappa
        assert r==-1.0 and beta==0.5, "r=-1.0 and beta=0.5 required for extended Cauchy prior"
        self.r = r
        self.beta = beta
        self.alpha = 2 * abs(r) * beta

    def penalty(self, theta, state, net):
        # Cauchy prior on weights
        regularization = 0.0
        for li, layer in enumerate(net.layers):
            if isinstance(layer, eqx.nn.Linear):
                n_out, n_in = layer.weight.shape
                if li == 0:
                    v = self.vartheta1
                elif self.alpha >= 2:
                    v = self.kappa * n_in ** (-1.0)
                else:
                    v = self.kappa * n_in ** (-2.0 / self.alpha)
                cauchy_scale = jnp.sqrt(2 * v)
                regularization += jnp.sum(jnp.log(1 + (layer.weight / cauchy_scale) ** 2)) # Cauchy prior regularization for first layer
                regularization += jnp.sum(jnp.log(1 + (layer.bias / cauchy_scale) ** 2)) # Cauchy prior regularization for first layer
        return regularization