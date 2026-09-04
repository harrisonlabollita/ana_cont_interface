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

"""TRIQS front-end for the ana_cont analytic continuation library."""

from ._util import moment_matrix
from .grids import RealGrid, linear_grid, tangent_grid
from .models import flat_model, gaussian_model, poorman_model, super_gaussian_model
from .problem import ContinuationProblem, ContinuationTask, gf_problem, sigma_problem
from .result import ContinuationResult, SigmaResult, TaskDiagnostics
from .solve import solve
from .validate import ValidationReport, backtransform, check_moments, validate

__all__ = [
    "RealGrid",
    "linear_grid",
    "tangent_grid",
    "flat_model",
    "gaussian_model",
    "super_gaussian_model",
    "poorman_model",
    "moment_matrix",
    "ContinuationProblem",
    "ContinuationTask",
    "gf_problem",
    "sigma_problem",
    "solve",
    "ContinuationResult",
    "SigmaResult",
    "TaskDiagnostics",
    "backtransform",
    "check_moments",
    "validate",
    "ValidationReport",
]
