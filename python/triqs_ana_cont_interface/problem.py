"""Decomposition of TRIQS containers into independent scalar continuation tasks."""

import warnings
from dataclasses import dataclass

import numpy as np
from triqs.gfs import BlockGf, Gf, is_gf_hermitian
from triqs.mesh import MeshImFreq, MeshImTime

from ._h5 import register_dataclass
from ._util import as_blocks, hermitian_part, matrixify, moment_matrix, norb, pick
from .grids import RealGrid

MODES = ("diagonal", "poorman", "eigenbasis")

# How far the Matsubara matrix may depart from symmetry, in error bars, before
# mode='poorman' refuses to run. See _check_symmetry.
ASYMMETRY_TOLERANCE = 5.0

# How far two blocks declared degenerate may differ, in error bars (rms), before
# the copy is called into question. Independent noise on two genuinely degenerate
# blocks already gives sqrt(2), so the tolerance sits above that.
DEGENERACY_TOLERANCE = 3.0


@register_dataclass
@dataclass
class ContinuationTask:
    """One scalar continuation: exactly one ana_cont problem's worth of data."""

    block: str
    i: int
    j: int
    im_axis: np.ndarray
    im_data: np.ndarray
    error: np.ndarray
    offdiag: bool = False
    shift: complex = 0.0
    model: np.ndarray = None
    model_norm: float = 1.0

    @property
    def key(self):
        return (self.block, self.i, self.j)


@register_dataclass
@dataclass
class GfLayout:
    """How to rebuild the TRIQS container from the scalar results."""

    block_names: tuple
    target_shapes: dict
    mesh_in: object
    beta: float
    is_block: bool

    def __post_init__(self):
        # HDF5 turns tuples into lists; keep shapes comparable to ()
        self.block_names = tuple(self.block_names)
        self.target_shapes = {str(k): tuple(v) for k, v in self.target_shapes.items()}


@register_dataclass
@dataclass
class Recipe:
    """What procedure to apply."""

    kernel_mode: str
    mode: str
    quantity: str  # 'gf' | 'sigma'
    rotation: dict = None
    sigma_inf: dict = None
    first_moment: dict = None
    degenerate_blocks: tuple = None

    def __post_init__(self):
        if self.degenerate_blocks is not None:
            self.degenerate_blocks = tuple(
                tuple(str(b) for b in group) for group in self.degenerate_blocks
            )

    @property
    def is_freq(self):
        return self.kernel_mode.startswith("freq")

    @property
    def representative(self):
        """Copied block -> the block continued in its place.

        Blocks that are continued themselves are absent, so use .get(name, name).
        """
        return {
            name: group[0]
            for group in self.degenerate_blocks or ()
            for name in group[1:]
        }


@register_dataclass
@dataclass
class ContinuationProblem:
    gf: object
    grid: RealGrid
    tasks: tuple
    layout: GfLayout
    recipe: Recipe

    def __post_init__(self):
        self.tasks = tuple(self.tasks)

    def task(self, block, i=0, j=0):
        for t in self.tasks:
            if t.key == (block, i, j):
                return t
        raise KeyError((block, i, j))


def _kernel_mode(mesh):
    if isinstance(mesh, MeshImFreq):
        domain = "freq"
    elif isinstance(mesh, MeshImTime):
        domain = "time"
    else:
        raise TypeError(
            "Unsupported mesh {}. Only MeshImFreq and MeshImTime are supported; "
            "convert a DLR mesh first (make_gf_imfreq / make_gf_imtime).".format(
                type(mesh).__name__
            )
        )
    if mesh.statistic != "Fermion":
        raise NotImplementedError(
            "Only fermionic statistics are supported; bosonic continuation is not implemented."
        )
    return "{}_fermionic".format(domain)


def _shared_mesh(blocks):
    """Every block must live on the same mesh: one selection is applied to all."""
    mesh = blocks[0][1].mesh
    for name, g in blocks[1:]:
        if g.mesh != mesh:
            raise ValueError(
                "block '{}' has mesh {} but block '{}' has {}; all blocks must share "
                "one mesh, since the same frequency selection is applied to each".format(
                    name, g.mesh, blocks[0][0], mesh
                )
            )
    return mesh


def _tau_stride(n_avail, n_tau):
    """Stride that divides the tau interval evenly, so the reduced grid is
    itself a MeshImTime and the backtransform needs no interpolation."""
    target = max(1, int(round((n_avail - 1) / max(1, int(n_tau) - 1))))
    divisors = [d for d in range(1, n_avail) if (n_avail - 1) % d == 0]
    return min(divisors, key=lambda d: (abs(d - target), d))


def _selection(mesh, n_iw, n_tau):
    """Points of the input mesh actually used, as a slice, plus the axis values."""
    if isinstance(mesh, MeshImFreq):
        n_avail = mesh.n_iw
        n_use = n_avail if n_iw is None else min(int(n_iw), n_avail)
        if n_iw is None and n_avail > 300:
            warnings.warn(
                "using all {0} positive Matsubara frequencies; the high-frequency region "
                "is trivial and encoded in the kernel, so consider n_iw=100 or so".format(
                    n_avail
                )
            )
        sel = slice(n_avail, n_avail + n_use)
        return sel, np.array([complex(p.value) for p in mesh]).imag[sel]
    n_avail = len(mesh)
    if n_tau is None and n_avail > 2000:
        warnings.warn(
            "using all {0} imaginary-time points; consider n_tau=200 or so".format(n_avail)
        )
    stride = 1 if n_tau is None else _tau_stride(n_avail, n_tau)
    sel = slice(None, None, stride)
    return sel, np.array([float(p.value) for p in mesh])[sel]


def _resolve_error(error, block, i, j, sel, npts):
    if isinstance(error, (int, float, np.floating)):
        out = np.full(npts, float(error))
    elif isinstance(error, dict):
        return _resolve_error(error[block], block, i, j, sel, npts)
    elif isinstance(error, (Gf, BlockGf)):
        g = error[block] if isinstance(error, BlockGf) else error
        # ana_cont takes stdev as the deviation of the real part and reuses it
        # for the imaginary part, so only the real part is read here
        out = np.abs(matrixify(g.data)[sel, i, j].real)
    else:
        arr = np.asarray(error, dtype=float)
        if arr.ndim != 1:
            raise ValueError("error array must be one-dimensional, got shape {}".format(arr.shape))
        out = arr if arr.shape[0] == npts else arr[sel]
    out = np.asarray(out, dtype=float)
    if out.shape[0] != npts:
        raise ValueError(
            "error has {} points but {} are used for {}".format(out.shape[0], npts, (block, i, j))
        )
    if not np.all(out > 0.0):
        raise ValueError("error must be strictly positive for {}".format((block, i, j)))
    if np.max(out) < 1e-8:
        warnings.warn(
            "error bars below 1e-8 for {}: maxent needs some noise to work against".format(
                (block, i, j)
            )
        )
    return out


def _rotate(data, u):
    """U^dag M U on the target indices, for every mesh point."""
    return np.einsum("ai,wab,bj->wij", u.conj(), data, u, optimize=True)


def _indices(n, mode):
    diagonal = [(i, i, False) for i in range(n)]
    if mode == "poorman":
        return diagonal + [(i, j, True) for i in range(n) for j in range(i + 1, n)]
    return diagonal


def _warn_discarded_offdiag(name, data, mode):
    """Off-diagonal weight that `mode` throws away, relative to the diagonal."""
    n = data.shape[-1]
    if n < 2:
        return
    diagonal = max(np.max(np.abs(data[:, i, i])) for i in range(n))
    offdiag = max(np.max(np.abs(data[:, i, j])) for i in range(n) for j in range(n) if i != j)
    if diagonal > 0.0 and offdiag / diagonal > 1e-6:
        warnings.warn(
            "block '{}': mode='{}' discards off-diagonal weight "
            "max|g_ij|/max|g_ii| = {:.3g}".format(name, mode, offdiag / diagonal)
            + (
                "; the rotation does not diagonalize this block"
                if mode == "eigenbasis"
                else "; use mode='poorman' or 'eigenbasis' to continue it"
            )
        )


def _block_name(names, item):
    """The block a degeneracy group refers to.

    Groups are given as block indices, which is what TRIQS' own degeneracy
    helpers produce and what BlockGf itself accepts; block names also work.
    """
    if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
        return str(item)
    if not -len(names) <= item < len(names):
        raise ValueError(
            "degenerate_blocks names block index {}, but the input has {} blocks "
            "({})".format(item, len(names), names)
        )
    return names[item]


def _resolve_degenerate(blocks, shapes, groups):
    """Normalize the degeneracy groups to block names and check they
    describe this container."""
    if groups is None:
        return None
    names = [name for name, _ in blocks]
    seen = set()
    out = []
    for group in groups:
        group = tuple(_block_name(names, b) for b in group)
        for name in group:
            if name not in names:
                raise ValueError(
                    "degenerate_blocks names block '{}', which is not in the input "
                    "(the blocks are {})".format(name, names)
                )
            if name in seen:
                raise ValueError(
                    "block '{}' appears in more than one degeneracy group; a block can "
                    "be copied from only one other".format(name)
                )
            seen.add(name)
            if shapes[name] != shapes[group[0]]:
                raise ValueError(
                    "blocks '{}' and '{}' are declared degenerate but have target shapes "
                    "{} and {}; the spectra are copied element by element".format(
                        name, group[0], shapes[name], shapes[group[0]]
                    )
                )
        if len(group) > 1:
            out.append(group)
    return tuple(out) or None


def _check_degeneracy(groups, tasks):
    """A copied spectrum is only as right as the two blocks are equal, so
    measure it, in error bars like the other input checks. validate() measures it
    again on the result, since each copy keeps its own data."""
    if not groups:
        return
    by_key = {t.key: t for t in tasks}
    for group in groups:
        for name in group[1:]:
            worst = max(
                np.sqrt(np.mean(
                    np.abs(t.im_data - by_key[(group[0], t.i, t.j)].im_data) ** 2 / t.error ** 2
                ))
                for key, t in by_key.items()
                if key[0] == name
            )
            if worst > DEGENERACY_TOLERANCE:
                warnings.warn(
                    "blocks '{}' and '{}' are declared degenerate but differ by {:.1f} error "
                    "bars (rms); '{}' becomes a copy of '{}' and that difference is lost. "
                    "Drop the group to continue them separately.".format(
                        name, group[0], worst, name, group[0]
                    )
                )


def _check_symmetry(name, data, errors, sel, shift):
    """poorman needs a symmetric matrix, which is stronger than hermitian.

    ana_cont returns a real spectral function per element, and a real A_ij
    forces G_ij(iw) = G_ji(iw). A hermitian block with an imaginary A_ij is
    antisymmetric instead, and continuing g_ij alone then fits the data with a
    large oscillating real spectrum that no residual check can detect.

    The requirement applies to what is actually continued, i.e. after the
    constant is subtracted: a hermitian Sigma_inf with a complex off-diagonal
    makes the raw matrix asymmetric while the spectral part stays symmetric.
    """
    n = data.shape[-1]
    const = np.zeros((n, n), dtype=complex) if shift is None else shift
    for i in range(n):
        for j in range(i + 1, n):
            upper = data[sel, i, j] - const[i, j]
            lower = data[sel, j, i] - const[j, i]
            deviation = np.max(np.abs(upper - lower))
            tolerance = ASYMMETRY_TOLERANCE * np.max(errors[(i, j)])
            if deviation > tolerance:
                raise ValueError(
                    "block '{}' element ({}, {}) is not symmetric: max|g_ij - g_ji| = {:.3g}, "
                    "which is {:.0f} error bars. mode='poorman' continues one real spectrum "
                    "per element, which requires g_ij == g_ji. Use mode='eigenbasis' with a "
                    "rotation that diagonalizes the block.".format(
                        name, i, j, deviation, deviation / np.max(errors[(i, j)])
                    )
                )


def _check_moment_signs(name, moments):
    """A_Sigma is positive, so its integral -- the diagonal first moment -- is
    too. A negative one means the estimate is wrong, and abs() would turn it
    into a plausible-looking model norm."""
    diagonal = np.real(np.diag(moments))
    if np.any(diagonal <= 0.0):
        warnings.warn(
            "block '{}' has a non-positive first moment on the diagonal ({}); the "
            "spectral function of a self-energy integrates to a positive number, so "
            "this estimate is wrong. Pass model_norm= explicitly.".format(
                name, np.array2string(diagonal, precision=4)
            )
        )


def _high_frequency_moments(g, u):
    """Sigma_inf and the first moment from the high-frequency behaviour.

    Re Sigma(i w_n) -> Sigma_inf and i (Sigma - Sigma_inf) w_n -> the first
    moment, both with an O(1/w_n^2) correction, averaged over the top
    frequencies to suppress noise and hermitized. Used in preference to
    fit_hermitian_tail, which is dominated by noise in exactly that region: on
    a self-energy with 1e-3 noise it returns a negative first moment, which is
    not a usable norm.
    """
    data = matrixify(g.data)
    if u is not None:
        data = _rotate(data, u)
    n_iw = g.mesh.n_iw
    w_n = np.array([complex(p.value) for p in g.mesh]).imag[n_iw:]
    n_avg = max(2, n_iw // 10)
    top, w_top = data[n_iw:][-n_avg:], w_n[-n_avg:]

    constant = hermitian_part(top).mean(axis=0)
    residual = top - constant[None]
    moment = hermitian_part(1j * residual * w_top[:, None, None]).mean(axis=0)
    _warn_unstable_moments(g, data, n_iw, w_n, constant)
    return constant, moment


def _warn_unstable_moments(g, data, n_iw, w_n, constant):
    """The 1/w_n^2 bias is invisible in the estimate itself, so compare the top
    decile against the one below it; a large drift means w_max is too small."""
    n_avg = max(2, n_iw // 10)
    if n_iw < 4 * n_avg:
        return
    lower = hermitian_part(data[n_iw:][-2 * n_avg : -n_avg]).mean(axis=0)
    scale = np.max(np.abs(constant))
    drift = np.max(np.abs(constant - lower))
    if scale > 0.0 and drift > 0.05 * scale:
        warnings.warn(
            "the high-frequency moments are still drifting ({:.1%} between the top two "
            "deciles of w_n, w_max = {:.3g}): they are biased by O(1/w_n^2). Pass "
            "sigma_inf= and model_norm= explicitly, or use a mesh with more "
            "frequencies.".format(drift / scale, w_n[-1])
        )


def _build(gf, grid, error, quantity, n_iw, n_tau, mode, rotation, model, moments,
           degenerate):
    if mode not in MODES:
        raise ValueError("mode must be one of {}, got {!r}".format(MODES, mode))
    blocks = as_blocks(gf)
    is_block = isinstance(gf, BlockGf)
    mesh = _shared_mesh(blocks)
    kernel_mode = _kernel_mode(mesh)
    sel, im_axis = _selection(mesh, n_iw, n_tau)
    is_tau = kernel_mode.startswith("time")
    npts = len(im_axis)

    if not is_tau:
        for name, g in blocks:
            if not is_gf_hermitian(g):
                warnings.warn(
                    "block '{}' is not hermitian; consider make_hermitian() first".format(name)
                )

    tasks = []
    shapes = {}
    for name, g in blocks:
        shape = tuple(g.target_shape)
        shapes[name] = shape
        n = norb(shape)
        data = matrixify(g.data)
        if mode == "eigenbasis":
            data = _rotate(data, rotation[name])
        if mode in ("diagonal", "eigenbasis"):
            _warn_discarded_offdiag(name, data, mode)
        if is_tau:
            # TRIQS G(tau) is negative; ana_cont's time kernel is positive and
            # satisfies int dw K(tau, w) A(w) = -G(tau)
            data = -data.real

        errors = {
            (i, j): _resolve_error(error, name, i, j, sel, npts)
            for i, j, _ in _indices(n, mode)
        }
        shift = None if moments is None else moment_matrix(pick(moments[0], name), n)
        norms = None if moments is None else moment_matrix(pick(moments[1], name), n)
        if norms is not None:
            _check_moment_signs(name, norms)
        if mode == "poorman":
            _check_symmetry(name, data, errors, sel, shift)
        for i, j, offdiag in _indices(n, mode):
            tasks.append(
                ContinuationTask(
                    block=name,
                    i=i,
                    j=j,
                    im_axis=im_axis,
                    im_data=np.ascontiguousarray(data[sel, i, j]),
                    error=errors[(i, j)],
                    offdiag=offdiag,
                    shift=0.0 if shift is None else complex(shift[i, j]),
                    model=None if model is None else np.asarray(model, dtype=float),
                    model_norm=1.0 if norms is None else float(np.abs(norms[i, j])),
                )
            )

    degenerate = _resolve_degenerate(blocks, shapes, degenerate)
    _check_degeneracy(degenerate, tasks)

    layout = GfLayout(
        block_names=tuple(n for n, _ in blocks),
        target_shapes=shapes,
        mesh_in=mesh,
        beta=float(mesh.beta),
        is_block=is_block,
    )
    recipe = Recipe(
        kernel_mode=kernel_mode,
        mode=mode,
        quantity=quantity,
        rotation=rotation,
        sigma_inf=None if moments is None else moments[0],
        first_moment=None if moments is None else moments[1],
        degenerate_blocks=degenerate,
    )
    return ContinuationProblem(
        gf=gf, grid=grid, tasks=tuple(tasks), layout=layout, recipe=recipe
    )


def gf_problem(
    gf,
    grid,
    error,
    n_iw=None,
    n_tau=None,
    mode="diagonal",
    rotation=None,
    model=None,
    degenerate_blocks=None,
):
    """Build a continuation problem for a Green's function.

    error is required: a scalar, a 1-D array, a dict keyed by block, or a
    Gf/BlockGf of the same structure (only its real part is read).

    degenerate_blocks groups blocks known to be equal, as block indices:
    [[0, 1]] declares the first two blocks degenerate. Only the first block of
    each group is continued; the others take a copy of its spectra, while
    keeping their own constant and their own data. Block names work too.
    """
    rotation = _resolve_rotation(gf, mode, rotation)
    return _build(
        gf, grid, error, "gf", n_iw, n_tau, mode, rotation, model, None, degenerate_blocks
    )


def sigma_problem(
    sigma_iw,
    grid,
    error,
    n_iw=None,
    mode="diagonal",
    rotation=None,
    model=None,
    sigma_inf=None,
    model_norm=None,
    degenerate_blocks=None,
):
    """Build a continuation problem for a self-energy.

    Sigma_inf and the first moment come from the high-frequency behaviour
    unless given: Sigma_inf is subtracted from the data and re-added after the
    Kramers-Kronig transform, and the first moment sets the norm of the default
    model, because A_Sigma integrates to the first moment, not to 1.

    degenerate_blocks groups blocks known to be equal, e.g. [[0, 1]]; see
    gf_problem. Sigma_inf and the first moment are still taken per block.
    """
    rotation = _resolve_rotation(sigma_iw, mode, rotation)
    blocks = as_blocks(sigma_iw)
    if not isinstance(blocks[0][1].mesh, MeshImFreq):
        raise TypeError("sigma_problem needs a Matsubara-frequency self-energy")
    shifts, norms = {}, {}
    for name, g in blocks:
        u = rotation[name] if rotation else None
        constant, moment = _high_frequency_moments(g, u)
        n = norb(g.target_shape)
        # supplied moments are physical, i.e. in the basis of the input, so
        # they are rotated into the working basis here and rotated back for
        # reporting: what the user passes in is what comes back out
        if sigma_inf is not None:
            constant = _into_basis(moment_matrix(pick(sigma_inf, name), n), u)
        if model_norm is not None:
            moment = _into_basis(moment_matrix(pick(model_norm, name), n), u)
        shifts[name], norms[name] = constant, moment
    return _build(
        sigma_iw, grid, error, "sigma", n_iw, None, mode, rotation, model,
        (shifts, norms), degenerate_blocks,
    )


def _into_basis(matrix, u):
    return matrix if u is None else _rotate(matrix[None], u)[0]


def _resolve_rotation(gf, mode, rotation):
    if mode != "eigenbasis":
        return rotation
    if rotation is None:
        raise ValueError(
            "mode='eigenbasis' needs an explicit rotation: the unitary that diagonalizes "
            "the object (e.g. the eigenvectors of H(k)). It cannot be derived reliably "
            "from the input, because the tail coefficients it would come from are "
            "destroyed by noise."
        )
    if not isinstance(rotation, dict):
        rotation = {name: rotation for name, _ in as_blocks(gf)}
    return {k: np.atleast_2d(np.asarray(v)) for k, v in rotation.items()}
