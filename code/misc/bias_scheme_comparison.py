# Generates bias_scheme_comparison.pdf (Figure "fig:bias_calibration").
#
# Compares three bias distributions for the GSBL-BNN prior with Cauchy marginals
# (r = -1, beta = 1/2, tail index alpha = 2|r|beta = 1):
#   (A, D) heavy-tailed bias: every weight AND bias carries its own psi ~ GG(r, beta, vartheta^(l))
#   (B, E) uniform per-weight bias: bias uniform over the neuron's attained pre-activation range
#   (C, F) shared-psi bias (ours): a neuron's incoming weights are N(0, psi), its bias N(0, c^2 psi),
#          with one shared psi ~ GG(r, beta, vartheta^(l)) per neuron; bias scale c = 2.
#
# Scales (see the calibration subsubsection and eq:fan_in in the paper):
#   first layer      vartheta^(1) = 100                (jump-sharpness knob)
#   deeper layers    vartheta^(l) = n^(-2/alpha) = 64^(-2), n = 64 fan-in (incl. output layer)
#
# Architecture: 1-64-64-64-64-1, phi = tanh, Omega = [-1, 1].
# Row 1: histograms of transition locations (pre-activation sign changes) per layer,
#        aggregated over N_NETS = 400 independent networks; annotations give the fraction
#        of neurons transitioning within Omega.
# Row 2: three representative prior draws of u_theta per scheme, each shifted and scaled
#        to span [-1, 1] (min -> -1, max -> +1).
#
# Usage: python bias_scheme_comparison.py

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../out/figures/bias_scheme_comparison.pdf')

# ---------------------------------------------------------------- model setup
R, BETA = -1.0, 0.5                  # Cauchy hyperprior
ALPHA = 2.0 * abs(R) * BETA          # tail index alpha = 1
TH1 = 100.0                          # first-layer scale vartheta^(1)
C = 2.0                              # bias scale c (shared-psi scheme)
ARCH = [1, 64, 64, 64, 64, 1]        # four hidden layers of width 64
N_NETS = 400                         # networks per transition histogram
N_DRAWS = 3                          # prior draws per panel
XPLOT = np.linspace(-1.3, 1.3, 521)  # grid for transition detection (beyond Omega)
XOM = np.linspace(-1.0, 1.0, 401)    # grid for prior draws (Omega only)
SEED_HIST, SEED_DRAW = 0, 12


def vartheta(layer):
    """Layer scales: vartheta^(1) = TH1, deeper layers fan-in calibrated (eq:fan_in)."""
    if layer == 0:
        return TH1
    n = ARCH[layer]                  # fan-in of layer `layer` (previous width)
    return float(n) ** (-2.0 / ALPHA)


def psi_gg(rng, th, size):
    """psi ~ GG(r, beta, th) for r = -1: psi = th / G with G ~ Gamma(beta, 1)."""
    return th / rng.gamma(BETA, 1.0, size)


def sample_net(rng, scheme):
    """Draw (weights, biases) for one network under the given bias scheme."""
    Ws, bs = [], []
    for l in range(len(ARCH) - 1):
        n_in, n_out = ARCH[l], ARCH[l + 1]
        th = vartheta(l)
        if scheme == 'shared':
            psi = psi_gg(rng, th, (n_out, 1))                    # one psi per neuron
            W = np.sqrt(psi) * rng.standard_normal((n_out, n_in))
            b = C * np.sqrt(psi[:, 0]) * rng.standard_normal(n_out)
        else:
            psi = psi_gg(rng, th, (n_out, n_in))                 # one psi per weight
            W = np.sqrt(psi) * rng.standard_normal((n_out, n_in))
            if scheme == 'heavy':
                psb = psi_gg(rng, th, n_out)                     # own psi per bias
                b = np.sqrt(psb) * rng.standard_normal(n_out)
            else:                                                # 'uniform'
                b = None                                         # set from pre-activations
        Ws.append(W); bs.append(b)
    return Ws, bs


def forward_transitions(Ws, bs, rng, x):
    """Forward pass on grid x; returns per-layer transition locations and neuron counts."""
    om = np.abs(x) <= 1.0                                        # sub-grid inside Omega
    h = x[None, :]
    trans = [[] for _ in range(len(ARCH) - 1)]
    hit = np.zeros(len(ARCH) - 1)                                # neurons transitioning in Omega
    for l, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ h
        if b is None:                                            # uniform per-weight bias
            lo, hi = z[:, om].min(axis=1), z[:, om].max(axis=1)  # range attained ON Omega
            b = -rng.uniform(lo, hi)
        a = z + b[:, None]
        s = np.signbit(a)
        for i in range(a.shape[0]):
            k = np.flatnonzero(s[i, :-1] != s[i, 1:])
            if k.size:
                xi = 0.5 * (x[k] + x[k + 1])
                trans[l].extend(xi.tolist())
                if np.any(np.abs(xi) <= 1.0):
                    hit[l] += 1
        h = np.tanh(a)
    return trans, hit


def forward_u(Ws, bs, rng, x):
    """Forward pass returning the scalar output u_theta on grid x."""
    h = x[None, :]
    for l, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ h
        if b is None:
            lo, hi = z.min(axis=1), z.max(axis=1)
            b = -rng.uniform(lo, hi)
        a = z + b[:, None]
        h = np.tanh(a) if l < len(Ws) - 1 else a                 # linear output layer
    return h[0]


# ---------------------------------------------------------------- statistics
SCHEMES = ['heavy', 'uniform', 'shared']
LABELS = ['heavy-tailed bias', 'uniform per-weight bias', r'shared $\psi$ ($c{=}2$)']
trans_all, frac_all = {}, {}
for scheme in SCHEMES:
    rng = np.random.default_rng(SEED_HIST)
    trans = [[] for _ in range(len(ARCH) - 1)]
    hits = np.zeros(len(ARCH) - 1)
    for _ in range(N_NETS):
        Ws, bs = sample_net(rng, scheme)
        t, h = forward_transitions(Ws, bs, rng, XPLOT)
        for l in range(len(t)):
            trans[l].extend(t[l])
        hits += h
    trans_all[scheme] = trans
    frac_all[scheme] = hits / (N_NETS * np.array(ARCH[1:-1] + [ARCH[-2]], dtype=float))

# ---------------------------------------------------------------- plotting
RED = '#c0504d'
GRAY = '0.5'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
fig, axes = plt.subplots(2, 3, figsize=(7.75, 4.0))
BINS = np.linspace(-1.15, 1.15, 27)                              # 26 bins

for col, (scheme, lab) in enumerate(zip(SCHEMES, LABELS)):
    ax = axes[0, col]
    ax.axhspan(-1.0, 1.0, color=GRAY, alpha=0.12, lw=0)
    for l in range(4):                                           # hidden layers 1..4
        cnt, _ = np.histogram(trans_all[scheme][l], bins=BINS)
        if cnt.max() > 0:
            w = 0.38 * cnt / cnt.max()
        else:
            w = np.zeros_like(cnt, dtype=float)
        yc = 0.5 * (BINS[:-1] + BINS[1:])
        pos = l + 1
        ax.fill_betweenx(yc, pos - w, pos + w, step='mid', color=RED, alpha=0.55, lw=0.8, edgecolor=RED)
        ax.annotate(f'{100 * frac_all[scheme][l]:.0f}%', (pos, 1.24), color=RED,
                    ha='center', fontsize=7)
    ax.set_xlim(0.4, 4.6); ax.set_ylim(-1.3, 1.45)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel(r'layer $\ell$')
    if col == 0:
        ax.set_ylabel(r'transition location in $\Omega$')
    ax.set_title(f'({chr(65 + col)}) transitions, {lab}')

    ax = axes[1, col]
    rng = np.random.default_rng(SEED_DRAW)
    for j in range(N_DRAWS):
        Ws, bs = sample_net(rng, scheme)
        u = forward_u(Ws, bs, rng, XOM)
        u = 2.0 * (u - u.min()) / max(u.max() - u.min(), 1e-12) - 1.0   # rescale to [-1, 1]
        ax.plot(XOM, u, lw=1.1, alpha=0.9)
    ax.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax.set_xlim(-1, 1); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r'$x$')
    if col == 0:
        ax.set_ylabel(r'$u_{\theta}(x)$ (rescaled)')
    ax.set_title(f'({chr(68 + col)}) prior draws $u_{{\\theta}}$, {lab}')

fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight', pad_inches=0.02)
print('wrote', OUT)
