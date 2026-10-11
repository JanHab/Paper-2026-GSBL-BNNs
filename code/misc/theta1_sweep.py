# Generates theta1_sweep.pdf (Figure "fig:theta1_sweep").
#
# Illustrates the effect of the first-layer scale vartheta^(1) on prior draws of
# u_theta for the shared-psi GSBL-BNN prior with Cauchy marginals (r = -1, beta = 1/2,
# alpha = 2|r|beta = 1) and bias scale c = 2.
#
# Scales (see the calibration subsubsection and eq:fan_in in the paper):
#   first layer      vartheta^(1) in {10, 100, 1000}   (panels; jump-sharpness knob)
#   deeper layers    vartheta^(l) = n^(-2/alpha) = 64^(-2), n = 64 fan-in (incl. output layer)
#
# Architecture: 1-64-64-64-64-1, phi = tanh, Omega = [-1, 1].
# The SAME underlying random draws (standard normals Z and gammas G) are reused across
# the three panels, so each network sharpens in place as vartheta^(1) grows: with the
# shared psi, a first-layer neuron's transition location xi = -c Z_b / Z_omega does not
# depend on psi, while the transition width scales like (vartheta^(1))^(-1/2).
# Each draw is shifted and scaled to span [-1, 1] (min -> -1, max -> +1), so that the jump
# structure is comparable across draws regardless of constant offsets.
#
# Usage: python theta1_sweep.py

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../out/figures/theta1_sweep.pdf')

R, BETA = -1.0, 0.5
ALPHA = 2.0 * abs(R) * BETA
C = 2.0
ARCH = [1, 64, 64, 64, 64, 1]
TH1S = [10.0, 100.0, 1000.0]
SEEDS = [2, 10, 12]
X = np.linspace(-1.0, 1.0, 801)


def base_draw(seed):
    """Draw the vartheta-independent randomness once: gammas G and normals Z per layer."""
    rng = np.random.default_rng(seed)
    layers = []
    for l in range(len(ARCH) - 1):
        n_in, n_out = ARCH[l], ARCH[l + 1]
        G = rng.gamma(BETA, 1.0, (n_out, 1))         # psi = vartheta / G  (r = -1)
        Z = rng.standard_normal((n_out, n_in))       # weights
        Zb = rng.standard_normal(n_out)              # bias
        layers.append((G, Z, Zb))
    return layers


def u_of(layers, th1):
    """Forward pass with first-layer scale th1; deeper layers fan-in calibrated."""
    h = X[None, :]
    for l, (G, Z, Zb) in enumerate(layers):
        th = th1 if l == 0 else float(ARCH[l]) ** (-2.0 / ALPHA)
        psi = th / G
        W = np.sqrt(psi) * Z
        b = C * np.sqrt(psi[:, 0]) * Zb
        a = W @ h + b[:, None]
        h = np.tanh(a) if l < len(layers) - 1 else a
    return h[0]


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9})
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0), sharey=True)
bases = [base_draw(s) for s in SEEDS]

for ax, th1 in zip(axes, TH1S):
    for lay in bases:
        u = u_of(lay, th1)
        u = 2.0 * (u - u.min()) / max(u.max() - u.min(), 1e-12) - 1.0   # rescale to [-1, 1]
        ax.plot(X, u, lw=1.0, alpha=0.9)
    ax.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax.set_xlim(-1, 1); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r'$x$')
    ax.set_title(r'$\vartheta^{(1)} = %g$' % th1)
axes[0].set_ylabel(r'$u_{\theta}(x)$ (rescaled)', fontsize=7)

fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight', pad_inches=0.02)
print('wrote', OUT)
