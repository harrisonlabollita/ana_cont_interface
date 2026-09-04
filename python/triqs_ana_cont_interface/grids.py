# Copyright (c) 2026 Harrison LaBollita
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https://www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Harrison LaBollita

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
