"""2D nodal discontinuous-Galerkin solver for scalar conservation laws, in pure JAX.

    u_t + f(u)_x + g(u)_y = 0    on [x_lo, x_hi] x [y_lo, y_hi]

The flux pair `(f, g)` is data, not code: pass a `Flux2D`. `KPP` gives the
Kurganov-Petrova-Popov rotating-wave problem `f = sin u`, `g = cos u`;
`burgers2d(u_ref)` gives `f = g = u^2/2`; `advection2d(cx, cy)` gives the linear
case. Anything else with a scalar `f`, `g` and their derivatives works too.

Scheme
------
Q1 tensor-product nodal DG (the 2D analogue of the P1 element in
`burgers1d_dg.py`), lumped (nodal) mass matrix, local Lax-Friedrichs (Rusanov)
interface flux, dimension-by-dimension minmod slope limiter, SSP-RK3 in time
with a fixed dt set from `cfl`, the mesh spacings and the flux's wave-speed
bounds. The whole rollout compiles into one `lax.scan`, so `jax.grad`
differentiates it natively.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

# SSP-RK3 as three applications of  x <- Pi( a*(x + dt*L(x)) + b*d ),
# with Pi the limiter and d the state at the start of the step.
STAGES = ((1.0, 0.0), (0.25, 0.75), (2.0 / 3.0, 1.0 / 3.0))

DEGREE = 1


def minmod3(a, b, c):
    """Zero unless all three arguments share a sign, else the smallest in modulus."""
    s = jnp.sign(a)
    same = (s == jnp.sign(b)) & (s == jnp.sign(c))
    return jnp.where(
        same,
        s * jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c))),
        0.0)


def _elementwise_deriv(fn):
    """d/du of a scalar function, applied elementwise to an array of any shape."""
    d = jax.grad(fn)

    def deriv(u):
        return jnp.reshape(jax.vmap(d)(jnp.reshape(u, (-1,))), jnp.shape(u))

    return deriv


# ---------------------------------------------------------------------------
# flux
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Flux2D:
    """A scalar flux pair plus the wave speeds and bounds the scheme needs.

    Frozen and hashable (its fields are strings, floats and function objects) so
    a solver holding one stays usable as a static `jit` argument.

    f, g   : x- and y-flux, elementwise on arrays.
    a, b   : df/du and dg/du, used for the Rusanov dissipation.
    ax_max : bound on |df/du| over the states that occur, sets dt.
    ay_max : bound on |dg/du|, likewise.
    """

    name: str
    f: Callable
    g: Callable
    a: Callable
    b: Callable
    ax_max: float
    ay_max: float

    @classmethod
    def autodiff(cls, name, f, g, ax_max, ay_max):
        """Same, with `a` and `b` obtained by autodiff instead of by hand."""
        return cls(
            name, f, g, _elementwise_deriv(f), _elementwise_deriv(g),
            ax_max, ay_max
        )


def _neg_sin(u):
    return -jnp.sin(u)


#: Kurganov-Petrova-Popov rotating wave: u_t + (sin u)_x + (cos u)_y = 0.
#: Non-convex in both directions; |f'| = |cos u| <= 1, |g'| = |sin u| <= 1.
KPP = Flux2D("kpp", jnp.sin, jnp.cos, jnp.cos, _neg_sin, 1.0, 1.0)


def _half_square(u):
    return 0.5 * u * u


def _identity(u):
    return u


def burgers2d(u_ref=1.5):
    """f = g = u^2/2. `u_ref` is headroom over max|u|, as in the 1D solver."""
    return Flux2D(
        f"burgers2d(u_ref={u_ref})", _half_square, _half_square,
        _identity, _identity, u_ref, u_ref
    )


def burgers_x(u_ref=1.5):
    """f = u^2/2, g = 0: the 1D problem embedded in 2D. Used for cross-checks."""
    return Flux2D(
        f"burgers_x(u_ref={u_ref})", _half_square, jnp.zeros_like,
        _identity, jnp.zeros_like, u_ref, 0.0
    )


def advection2d(cx=1.0, cy=1.0):
    """f = cx u, g = cy u."""
    return Flux2D(
        f"advection2d({cx}, {cy})",
        lambda u: cx * u, lambda u: cy * u,
        lambda u: jnp.full_like(u, cx), lambda u: jnp.full_like(u, cy),
        abs(cx), abs(cy)
    )


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalarDG2D:
    """Solver configuration. Frozen and hashable so it can be a static jit argument.

    Every array-valued quantity is a property recomputed from the scalars, which
    keeps the dataclass itself hashable.

    Defaults are the standard KPP test case: [-2, 2] x [-2.5, 1.5] to t = 1.
    """

    flux: Flux2D = KPP
    nx: int = 200
    ny: int = 200
    x_lo: float = -2.0
    x_hi: float = 2.0
    y_lo: float = -2.5
    y_hi: float = 1.5
    t_end: float = 1.0
    cfl: float = 0.9
    limiter: bool = True        # off to expose the smooth operator in gradient checks
    bc: str = "transmissive"    # "transmissive" (zero-gradient) or "periodic"
    n_quad: int = 3             # Gauss points per direction, volume and faces
    n_speed: int = 5            # samples of |f'| across an interface jump
    remat: bool = True             # checkpoint the RK3 step to save memory

    # -- derived scalars ------------------------------------------------
    @property
    def dx(self):
        return (self.x_hi - self.x_lo) / self.nx

    @property
    def dy(self):
        return (self.y_hi - self.y_lo) / self.ny

    @property
    def n_steps(self):
        speed = self.flux.ax_max / self.dx + self.flux.ay_max / self.dy
        dt_cfl = self.cfl / ((2 * DEGREE + 1) * speed)
        return max(1, int(np.ceil(self.t_end / dt_cfl)))

    @property
    def dt(self):
        return self.t_end / self.n_steps

    @property
    def n_dofs(self):
        return 4 * self.nx * self.ny

    # -- reference-element quadrature (numpy, evaluated at trace time) ---
    @property
    def _quad(self):
        """(points, weights) of `n_quad`-point Gauss-Legendre on [0, 1]."""
        s, w = np.polynomial.legendre.leggauss(self.n_quad)
        return 0.5 * (s + 1.0), 0.5 * w

    @property
    def _basis(self):
        """(n_quad, 2) values of the two 1D Lagrange nodal functions at the points."""
        s, _ = self._quad
        return np.stack([1.0 - s, s], axis=1)

    # -- derived grids (numpy; plotting and network input) ---------------
    @property
    def cell_left(self):
        return self.x_lo + np.arange(self.nx) * self.dx

    @property
    def cell_bottom(self):
        return self.y_lo + np.arange(self.ny) * self.dy

    @property
    def cell_x(self):
        """(nx,) cell-centre abscissae."""
        return self.cell_left + 0.5 * self.dx

    @property
    def cell_y(self):
        """(ny,) cell-centre ordinates."""
        return self.cell_bottom + 0.5 * self.dy

    @property
    def x_nodes(self):
        """(ny, nx, 2, 2) x-coordinate of every nodal dof, laid out like `u`."""
        xs = self.cell_left[None, :, None, None] + \
            np.array([0.0, self.dx])[None, None, None, :]
        return np.broadcast_to(xs, (self.ny, self.nx, 2, 2)).astype(np.float32)

    @property
    def y_nodes(self):
        """(ny, nx, 2, 2) y-coordinate of every nodal dof, laid out like `u`."""
        ys = self.cell_bottom[:, None, None, None] + \
            np.array([0.0, self.dy])[None, None, :, None]
        return np.broadcast_to(ys, (self.ny, self.nx, 2, 2)).astype(np.float32)

    @property
    def xy_flat(self):
        """(4*nx*ny, 2) node coordinates in `u.ravel()` order: the network input."""
        return np.stack([self.x_nodes.ravel(), self.y_nodes.ravel()],
                        axis=1).astype(np.float32)

    @property
    def x_sample(self):
        """(2*nx,) plotting abscissae, at the cell edges."""
        left = self.cell_left
        return np.stack([left,
                         left + self.dx], axis=1).ravel().astype(np.float32)

    @property
    def y_sample(self):
        """(2*ny,) plotting ordinates, at the cell edges."""
        bot = self.cell_bottom
        return np.stack([bot,
                         bot + self.dy], axis=1).ravel().astype(np.float32)

    # -- spatial operator -----------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def L(self, u):
        """du/dt = L(u), mass-inverted. `u` and the result are (ny, nx, 2, 2).

        Weak form on a cell K, per nodal test function phi_i:

            m_i du_i/dt = int_K (f d_x phi_i + g d_y phi_i)
                          - sum_faces int_face phi_i (F.n)

        with the lumped mass m_i = dx dy / 4. All six integrals are Gauss-
        Legendre; interface fluxes are Rusanov.
        """
        B = jnp.asarray(self._basis)                 # (nq, 2)
        w = jnp.asarray(self._quad[1])               # (nq,)
        D = jnp.asarray([-1.0, 1.0])                 # d/ds of the nodal functions
        dx, dy = self.dx, self.dy
        f, g = self.flux.f, self.flux.g

        # --- volume term ------------------------------------------------
        # u at the nq x nq tensor quadrature points; r indexes eta, q indexes xi
        uq = jnp.einsum('jiyx,ry,qx->jirq', u, B, B)
        # int f(u) d_x phi  ->  dy * D[kx] * sum_qr w w f(u) B[r, ky]
        Wx = jnp.einsum('jirq,r,q,ry->jiy', f(uq), w, w, B)
        Wy = jnp.einsum('jirq,r,q,qx->jix', g(uq), w, w, B)
        vol = dy * Wx[..., :, None] * D[None, None, None, :] \
            + dx * Wy[..., None, :] * D[None, None, :, None]

        # --- x-faces ----------------------------------------------------
        # traces on the low/high x-edge of each cell, linear in eta
        tr_lo = jnp.einsum('jiy,ry->jir', u[:, :, :, 0], B)
        tr_hi = jnp.einsum('jiy,ry->jir', u[:, :, :, 1], B)
        um, up = self._interfaces(tr_lo, tr_hi, axis=1)          # (ny, nx+1, nq)
        Sx = dy * jnp.einsum('jir,r,ry->jiy',
                             self._rusanov(f, self.flux.a, um, up), w, B)
        # the cell's low-x face carries +, its high-x face -
        fx = jnp.stack([Sx[:, :-1], -Sx[:, 1:]], axis=-1)

        # --- y-faces ----------------------------------------------------
        tr_lo = jnp.einsum('jix,qx->jiq', u[:, :, 0, :], B)
        tr_hi = jnp.einsum('jix,qx->jiq', u[:, :, 1, :], B)
        um, up = self._interfaces(tr_lo, tr_hi, axis=0)          # (ny+1, nx, nq)
        Sy = dx * jnp.einsum('jiq,q,qx->jix',
                             self._rusanov(g, self.flux.b, um, up), w, B)
        fy = jnp.stack([Sy[:-1], -Sy[1:]], axis=-2)

        return (vol + fx + fy) / (0.25 * dx * dy)

    def _interfaces(self, tr_lo, tr_hi, axis):
        """Pair up cell edge traces into the `n+1` interface states along `axis`.

        Interface `k` sits between cell `k-1` and cell `k`; `um` is the state to
        its low side, `up` the state to its high side. `transmissive` closes the
        two outermost interfaces with a zero jump, so no flux is created there.
        """
        sl = lambda x, i: jax.lax.slice_in_dim(x, i, i + 1, axis=axis)
        n = tr_lo.shape[axis]
        if self.bc == "periodic":
            left, right = sl(tr_hi, n - 1), sl(tr_lo, 0)
        elif self.bc == "transmissive":
            left, right = sl(tr_lo, 0), sl(tr_hi, n - 1)
        else:
            raise ValueError(f"bc must be 'periodic' or 'transmissive', got {self.bc!r}")
        return (jnp.concatenate([left, tr_hi], axis=axis),
                jnp.concatenate([tr_lo, right], axis=axis))

    def _rusanov(self, flux_fn, speed_fn, um, up):
        """Local Lax-Friedrichs flux with the wave speed maximised across the jump.

        `n_speed` points on the segment [um, up]; endpoints only (`n_speed = 2`)
        is not enough for a non-convex flux like KPP's.
        """
        s = jnp.linspace(0.0, 1.0, self.n_speed)
        us = um[..., None] + (up - um)[..., None] * s
        alpha = jnp.max(jnp.abs(speed_fn(us)), axis=-1)
        return 0.5 * (flux_fn(um) + flux_fn(up)) - 0.5 * alpha * (up - um)

    # -- limiter ---------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def limit(self, u):
        """Dimension-by-dimension minmod limiter against neighbouring cell averages.

        A hard switch, so `jax.grad` returns a subgradient: where the limiter
        clips, the derivative w.r.t. the original slope is zero -- a valid
        descent direction rather than the derivative of a smooth map.

        The reconstruction keeps only the mean and the two slopes, so the Q1
        cross term is discarded whenever the limiter is active.
        """
        if not self.limiter:
            return u

        mean = jnp.mean(u, axis=(-2, -1))
        slope_x = 0.5 * ((u[:, :, 0, 1] - u[:, :, 0, 0])
                         + (u[:, :, 1, 1] - u[:, :, 1, 0]))
        slope_y = 0.5 * ((u[:, :, 1, 0] - u[:, :, 0, 0])
                         + (u[:, :, 1, 1] - u[:, :, 0, 1]))

        dxp, dxm = self._neighbour_diff(mean, axis=1)
        dyp, dym = self._neighbour_diff(mean, axis=0)
        lim_x = minmod3(slope_x, dxp, dxm)
        lim_y = minmod3(slope_y, dyp, dym)

        off = jnp.asarray([-0.5, 0.5])
        return (mean[..., None, None]
                + lim_y[..., None, None] * off[:, None]
                + lim_x[..., None, None] * off[None, :])

    def _neighbour_diff(self, m, axis):
        """Forward/backward differences of the cell averages `m` along `axis`.

        `transmissive` replicates the edge cell, which zeroes the outward
        difference and hence flattens the outermost cells' slope.
        """
        if self.bc == "periodic":
            return jnp.roll(m, -1, axis) - m, m - jnp.roll(m, 1, axis)
        sl = lambda i, k: jax.lax.slice_in_dim(m, i, k, axis=axis)
        n = m.shape[axis]
        hi = jnp.concatenate([sl(1, n), sl(n - 1, n)], axis=axis)
        lo = jnp.concatenate([sl(0, 1), sl(0, n - 1)], axis=axis)
        return hi - m, m - lo

    # -- time stepping ----------------------------------------------------
    def _step(self, d, _):
        x = d
        for a, b in STAGES:
            x = self.limit(a * (x + self.dt * self.L(x)) + b * d)
        return x, None

    @partial(jax.jit, static_argnums=(0, 2))
    def solve(self, ic, n_steps=None):
        d = self.limit(ic)
        step = jax.checkpoint(self._step) if self.remat else self._step
        d, _ = jax.lax.scan(step, d, None,
                            length=self.n_steps if n_steps is None else n_steps)
        return d


    def frame_steps(self, n_frames):
        """Steps per frame of `rollout`; `n_frames * frame_steps >= n_steps`."""
        return int(np.ceil(self.n_steps / n_frames))

    def frame_times(self, n_frames):
        """(n_frames,) times the `rollout` snapshots are taken at."""
        return self.dt * self.frame_steps(n_frames) * np.arange(1, n_frames + 1)

    @partial(jax.jit, static_argnums=(0, 2))
    def rollout(self, ic, n_frames=5):
        """`n_frames` snapshots, shape (n_frames, ny, nx, 2, 2).

        Runs `n_frames * frame_steps(n_frames)` steps, so the last snapshot can
        sit slightly past `t_end`; `frame_times` gives the exact times.
        """
        per = self.frame_steps(n_frames)

        def frame(d, _):
            d, _ = jax.lax.scan(self._step, d, None, length=per)
            return d, d

        _, frames = jax.lax.scan(frame, self.limit(ic), None, length=n_frames)
        return frames

    # -- sampling / observation -------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def image(self, u):
        """(ny, nx, 2, 2) -> (2*ny, 2*nx) nodal values on the `y_sample` x `x_sample` grid.

        Discontinuities survive: each cell contributes a 2x2 block, so
        `pcolormesh(x_sample, y_sample, image(u))` shows the jumps.
        """
        E = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])   # nodal functions at 0.0, 1.0
        v = jnp.einsum('jiyx,py,sx->jips', u, E, E)     # (ny, nx, sy, sx)
        return jnp.transpose(v, (0, 2, 1, 3)).reshape(2 * self.ny, 2 * self.nx)

    @partial(jax.jit, static_argnums=0)
    def cell_mean(self, u):
        """(ny, nx, 2, 2) -> (ny, nx) cell averages, for contour plots."""
        return jnp.mean(u, axis=(-2, -1))

    def observation_indices(self, n_obs, nx_ref=None, ny_ref=None):
        """Return the indices and coordinates of `n_obs` observations."""
        nx_ref = nx_ref or self.nx
        ny_ref = ny_ref or self.ny
        assert self.nx % nx_ref == 0 and self.ny % ny_ref == 0, "grid must be an integer refinement of the reference grid"
        rx, ry = self.nx // nx_ref, self.ny // ny_ref
        m = int(np.ceil(np.sqrt(n_obs)))
        rows = np.round(np.linspace(0, 2 * ny_ref - 1, m)).astype(int)
        cols = np.round(np.linspace(0, 2 * nx_ref - 1, m)).astype(int)
        rows = ry * rows + (ry - 1) * (rows % 2)      # same cell, same side
        cols = rx * cols + (rx - 1) * (cols % 2)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        keep = np.linspace(0, m * m - 1, n_obs).astype(int)
        rr, cc = rr.ravel()[keep], cc.ravel()[keep]
        return (rr * (2 * self.nx) + cc, self.x_sample[cc], self.y_sample[rr])

    # -- initial conditions -------------------------------------------------
    def project(self, fn):
        """Nodal interpolant of `fn(x, y)` (numpy in, numpy out), as (ny, nx, 2, 2)."""
        return jnp.asarray(np.asarray(fn(self.x_nodes, self.y_nodes),
                                      dtype=np.float32))

    def disk(self, radius=1.0, inside=3.5 * np.pi, outside=0.25 * np.pi,
             centre=(0.0, 0.0)):
        """The KPP initial condition: `inside` on a disk, `outside` around it.

        Defaults are the standard test: u = 14 pi / 4 for |x| < 1, else pi / 4.
        Strict `<` so nodes exactly on the circle take the outer value.
        """
        cx, cy = centre
        return self.project(
            lambda x, y: np.where((x - cx) ** 2 + (y - cy) ** 2 < radius ** 2,
                                  inside, outside))

    def square_wave(self, lo=0.25, hi=0.75):
        """The 1D solver's square wave, extruded in y. For cross-checks."""
        return self.project(lambda x, y: np.where((x > lo) & (x < hi), 1.0, 0.0))

