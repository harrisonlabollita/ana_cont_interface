"""Real-frequency grids and their TRIQS meshes."""

from dataclasses import dataclass

import numpy as np
from triqs.mesh import MeshReFreq, MeshReFreqPts

from ._h5 import register_dataclass


@register_dataclass
@dataclass(frozen=True)
class RealGrid:
    """A real-frequency grid together with the TRIQS mesh that represents it.

    Uniform grids map onto MeshReFreq, non-uniform ones onto MeshReFreqPts,
    so a non-equispaced ana_cont grid needs no interpolation on output.
    """

    values: np.ndarray
    mesh: object

    def __len__(self):
        return len(self.values)


def linear_grid(w_min, w_max, n_w):
    values = np.linspace(w_min, w_max, num=n_w, endpoint=True)
    return RealGrid(values=values, mesh=MeshReFreq(w_min, w_max, n_w))


def tangent_grid(wmax, n_w, sharpness=2.1):
    """Grid that is dense around zero and sparse at the edges.

    Same construction as ana_cont's 'centered symmetric' grid
    (gui/gui_backend.py). `sharpness` > 2 controls the concentration;
    2.1 is the usual choice, 2.5 is milder.
    """
    u = np.linspace(-np.pi / sharpness, np.pi / sharpness, num=n_w, endpoint=True)
    values = wmax * np.tan(u) / np.tan(np.pi / sharpness)
    return RealGrid(values=values, mesh=MeshReFreqPts(list(values)))
