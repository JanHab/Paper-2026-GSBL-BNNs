
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree

from src.bnn import BNN

# Extension of BNN with Gaussian prior

class GaussianBNN(BNN):
    """theta ~ N(0, s^2):  penalty = ||theta||^2 / (2 s^2)."""

    def __init__(self, *args, gauss_std=1.0, initializer='Default', **kwargs):
        super().__init__(*args, **kwargs)
        self.gauss_std = gauss_std
        self.initializer = initializer

    def penalty(self, theta, state, net):
        # Gaussian prior on weights
        regularization = 0.0
        for layer in net.layers:
            if isinstance(layer, eqx.nn.Linear):
                regularization += jnp.sum(layer.weight**2)
                regularization += jnp.sum(layer.bias**2)
        return regularization * 1 / (2 * self.gauss_std ** 2)

    def init_state(self):
        """Draw theta from the Gaussian prior."""
        _, theta_key = jr.split(self.state_key)
        params, _ = eqx.partition(self.net, eqx.is_array)
        _, unravel = ravel_pytree(params)

        if self.initializer == 'Gaussian':  
            # Initialize parameters from Gaussian prior
            params = unravel(jr.normal(theta_key, (self.K,)) * self.gauss_std)

        self.params = params

        # commit the result to the net
        self.net = eqx.combine(params, self.static)

        return None, params