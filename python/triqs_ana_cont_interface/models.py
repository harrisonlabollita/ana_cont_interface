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

"""Default models. ana_cont never normalizes the model, so we do it here."""

import numpy as np

from ._util import trapz


def _normalized(model, w, norm):
    return model * (norm / trapz(model, w))


def flat_model(grid, norm=1.0):
    return _normalized(np.ones_like(grid.values), grid.values, norm)


def gaussian_model(grid, center=0.0, width=1.0, norm=1.0):
    w = grid.values
    return _normalized(np.exp(-0.5 * ((w - center) / width) ** 2), w, norm)


def super_gaussian_model(grid, width, exponent=6, norm=1.0):
    """exp(-(w/width)**exponent): flat inside the bandwidth, smooth cutoff outside."""
    w = grid.values
    return _normalized(np.exp(-((w / width) ** exponent)), w, norm)


def poorman_model(a_i, a_j, floor=1e-6):
    """Off-diagonal model sqrt(A_ii * A_jj) (Kraberger et al.).

    The model is the scale of the allowed fluctuation, not an expected sign,
    and ana_cont requires it strictly positive -- hence the floor.
    """
    return np.sqrt(np.abs(a_i * a_j)) + floor
