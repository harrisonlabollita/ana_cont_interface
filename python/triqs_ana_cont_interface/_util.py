"""Helpers shared across the package."""

import numpy as np
from triqs.gfs import BlockGf, Gf

trapz = getattr(np, "trapezoid", np.trapz)


def norb(target_shape):
    """Number of orbitals; a scalar-valued Gf counts as one."""
    return 1 if tuple(target_shape) == () else target_shape[0]


def matrixify(data):
    """View Gf data as (nw, n, n), so index logic need not special-case scalars."""
    return data.reshape(-1, 1, 1) if data.ndim == 1 else data


def as_blocks(gf):
    """(name, Gf) pairs. A bare Gf becomes the single block '0'."""
    if isinstance(gf, BlockGf):
        return [(str(n), g) for n, g in gf]
    return [("0", gf)]


def pick(value, block):
    return value[block] if isinstance(value, dict) else value


def hermitian_part(data):
    """(M + M^dag) / 2 over the target indices of a (nw, n, n) array."""
    return 0.5 * (data + np.conj(np.swapaxes(data, 1, 2)))


def moment_matrix(value, n):
    """Broadcast a moment to an (n, n) array indexable by [i, j].

    A scalar or a length-n vector goes on the diagonal, as TRIQS does for
    square targets: a constant self-energy shift is diagonal, not a matrix of
    equal entries. An (n, n) array is taken as given.
    """
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr * np.eye(n, dtype=arr.dtype)
    if arr.ndim == 1 and arr.shape[0] == n:
        return np.diag(arr)
    if arr.shape == (n, n):
        return arr
    raise ValueError(
        "cannot use a moment of shape {} for a {}x{} target; give a scalar, "
        "a length-{} vector, or an ({}, {}) matrix".format(arr.shape, n, n, n, n, n)
    )


def build_container(mesh, layout, per_block):
    """Assemble (nw, n, n) arrays into a Gf or BlockGf matching `layout`."""
    blocks = []
    for name in layout.block_names:
        g = Gf(mesh=mesh, target_shape=list(layout.target_shapes[name]))
        arr = per_block[name]
        g.data[:] = arr if g.data.ndim == 3 else arr[:, 0, 0]
        blocks.append(g)
    if not layout.is_block:
        return blocks[0]
    return BlockGf(name_list=list(layout.block_names), block_list=blocks)
