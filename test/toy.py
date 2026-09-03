"""Toy TRIQS objects with a known spectral function."""

import numpy as np
from triqs.gfs import BlockGf, Gf
from triqs.mesh import MeshImFreq, MeshImTime

BETA = 20.0
N_IW = 40
WMAX = 8.0
NW = 301
NOISE = 1e-4

_trapz = getattr(np, "trapezoid", np.trapz)


def real_axis(wmax=WMAX, nw=NW):
    return np.linspace(-wmax, wmax, nw)


def two_peaks(w):
    a = 0.4 * np.exp(-0.5 * (w - 1.8) ** 2) + 0.6 * np.exp(-0.5 * (w + 1.8) ** 2)
    return a / _trapz(a, w)


def one_peak(w, center=0.5, width=1.0):
    a = np.exp(-0.5 * ((w - center) / width) ** 2)
    return a / _trapz(a, w)


def _hermitian_noise(shape, n_half, amplitude, seed):
    """Noise that respects G(-iw) = conj(G(iw)), as real data does."""
    rng = np.random.RandomState(seed)
    half = amplitude * (rng.randn(n_half, *shape[1:]) + 1j * rng.randn(n_half, *shape[1:]))
    if len(shape) == 3:
        mirrored = np.conj(np.swapaxes(half, 1, 2))[::-1]
    else:
        mirrored = np.conj(half)[::-1]
    return np.concatenate((mirrored, half))


def gf_imfreq(spectra, beta=BETA, n_iw=N_IW, noise=NOISE, seed=1, offdiag=None, shift=None):
    """A Gf whose spectral function is `spectra`.

    spectra: one array (scalar-valued Gf) or a list of arrays (matrix-valued,
    one per diagonal element). `offdiag` adds a (0, 1) element whose spectrum is
    that fraction of sqrt(A_00 A_11). `shift` adds a constant, as a self-energy has.
    """
    w = real_axis()
    mesh = MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw)
    iw = np.array([complex(p.value) for p in mesh])
    kernel = 1.0 / (iw[:, None] - w[None, :])
    scalar = not isinstance(spectra, (list, tuple))
    if scalar:
        g = Gf(mesh=mesh, target_shape=[])
        g.data[:] = _trapz(kernel * spectra[None, :], w, axis=1)
    else:
        n = len(spectra)
        g = Gf(mesh=mesh, target_shape=[n, n])
        for i, a in enumerate(spectra):
            g.data[:, i, i] = _trapz(kernel * a[None, :], w, axis=1)
        if offdiag:
            a_od = offdiag * np.sqrt(spectra[0] * spectra[1])
            g.data[:, 0, 1] = _trapz(kernel * a_od[None, :], w, axis=1)
            g.data[:, 1, 0] = g.data[:, 0, 1]
    if shift is not None:
        g.data[:] += np.asarray(shift)
    if noise:
        g.data[:] += _hermitian_noise(g.data.shape, n_iw, noise, seed)
    return g


def block_gf_imfreq(names=("up", "dn"), **kwargs):
    blocks = [gf_imfreq(seed=1 + k, **kwargs) for k in range(len(names))]
    return BlockGf(name_list=list(names), block_list=blocks)


def tau_kernel(tau, w, beta):
    """exp(-tau w) / (1 + exp(-beta w)), in a form that does not overflow."""
    x, bw = np.outer(tau, w), beta * w
    pos = np.exp(-np.clip(x, -700, 700)) / (1.0 + np.exp(-np.clip(bw, -700, 700)))[None, :]
    neg = np.exp(np.clip((beta - tau)[:, None] * w[None, :], -700, 700)) / (
        1.0 + np.exp(np.clip(bw, -700, 700))
    )[None, :]
    return np.where(w[None, :] > 0.0, pos, neg)


def gf_imtime(spectra, beta=BETA, n_tau=1001, noise=NOISE, seed=2):
    """G(tau) = -int dw A(w) K(tau, w), built directly from the spectrum.

    Direct integration rather than a Fourier transform of G(iw): the transform
    needs a clean high-frequency tail, which noisy Matsubara data does not have.
    """
    w = real_axis()
    mesh = MeshImTime(beta=beta, statistic="Fermion", n_tau=n_tau)
    tau = np.array([float(p.value) for p in mesh])
    kernel = tau_kernel(tau, w, beta)
    scalar = not isinstance(spectra, (list, tuple))
    if scalar:
        g = Gf(mesh=mesh, target_shape=[])
        g.data[:] = -_trapz(kernel * spectra[None, :], w, axis=1)
    else:
        n = len(spectra)
        g = Gf(mesh=mesh, target_shape=[n, n])
        for i, a in enumerate(spectra):
            g.data[:, i, i] = -_trapz(kernel * a[None, :], w, axis=1)
    if noise:
        g.data[:] += noise * np.random.RandomState(seed).randn(*g.data.shape)
    return g
