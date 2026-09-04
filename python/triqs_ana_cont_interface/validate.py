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

"""Validation of a continuation: does it fit the data, and does it carry the right weight."""

from dataclasses import dataclass

import numpy as np
from triqs.gfs import fit_hermitian_tail
from triqs.mesh import MeshImFreq, MeshImTime

from ._util import as_blocks, build_container, matrixify, moment_matrix, norb, pick, trapz
from .result import SigmaResult

_MISSING_PROBLEM = "{} needs result.problem; this result was built without it"


def _continuation(result):
    """The plain continuation inside a result, and which quantity it holds."""
    if isinstance(result, SigmaResult):
        return result.aux, "sigma"
    return result, "gf"


def backtransform(result):
    """K.A as a TRIQS container, on the input mesh restricted to the fitted points.

    This is the fit to the data that the continuation actually produced; plotting
    it against the input is the acceptance criterion for a continuation.
    """
    inner, _ = _continuation(result)
    problem = inner.problem
    if problem is None:
        raise ValueError(_MISSING_PROBLEM.format("backtransform"))
    layout = problem.layout
    is_freq = problem.recipe.is_freq
    sign = 1.0 if is_freq else -1.0
    n_used = len(inner.diagnostics[0].im_axis)
    mesh = (
        MeshImFreq(beta=layout.beta, statistic="Fermion", n_iw=n_used)
        if is_freq
        else MeshImTime(beta=layout.beta, statistic="Fermion", n_tau=n_used)
    )

    per_block = {}
    for name in layout.block_names:
        n = norb(layout.target_shapes[name])
        half = np.zeros((n_used, n, n), dtype=complex)
        for d in inner.diagnostics:
            if d.block != name:
                continue
            half[:, d.i, d.j] = sign * (d.backtransform + d.shift)
            if d.i != d.j:
                # the spectrum is symmetric, the constant hermitian
                half[:, d.j, d.i] = sign * (d.backtransform + np.conj(d.shift))
        if is_freq:
            mirrored = np.conj(np.swapaxes(half, 1, 2))[::-1]
            per_block[name] = np.concatenate((mirrored, half))
        else:
            per_block[name] = half.real.astype(complex)
    return build_container(mesh, layout, per_block)


def _reference_m0(result, quantity, g_in, name, n, reference, is_freq):
    """The expected zeroth moment, and where it came from.

    Taken without a tail fit wherever an exact reference exists, because
    fit_hermitian_tail is badly degraded by noise.
    """
    if reference is not None:
        return moment_matrix(pick(reference, name), n), "user"
    if quantity == "sigma":
        return moment_matrix(pick(result.first_moment, name), n), "first-moment"
    if is_freq:
        return np.eye(n), "sum-rule"
    edge = -(matrixify(g_in.data)[0] + matrixify(g_in.data)[-1])
    return moment_matrix(edge, n), "edge"


def check_moments(result, reference=None, n_max=0):
    """Integrate the continued spectrum and compare with the expected weight.

    m_n = int dw w^n A(w) should equal tail[n+1] of the input, uniformly for a
    Green's function and a self-energy. The reference for m_0 comes from an
    exact sum rule where one exists (see _reference_m0):

      - Green's function on a Matsubara mesh: m_0 = 1, the anticommutator sum rule.
      - Green's function in imaginary time:   m_0 = -(G(0) + G(beta)).
      - Self-energy:                          m_0 = the first moment that set
        the model norm, which closes the loop over the whole pipeline.

    Higher moments (n_max > 0) have no such sum rule and do come from
    fit_hermitian_tail; its fit error is then reported as 'tail_error'.
    Pass `reference` to supply the expected m_0 yourself.
    """
    inner, quantity = _continuation(result)
    problem = inner.problem
    if problem is None:
        raise ValueError(_MISSING_PROBLEM.format("check_moments"))
    is_freq = problem.recipe.is_freq
    if n_max > 0 and not is_freq:
        raise ValueError(
            "n_max > 0 needs a Matsubara input: higher moments come from "
            "fit_hermitian_tail, which does not accept an imaginary-time mesh"
        )
    grid = problem.grid.values
    spectra = dict(as_blocks(inner.a_w))
    inputs = dict(as_blocks(problem.gf))

    out = {}
    for name, g_in in inputs.items():
        shape = problem.layout.target_shapes[name]
        n = norb(shape)
        first, source = _reference_m0(result, quantity, g_in, name, n, reference, is_freq)
        expected, sources, tail_error = [first], [source], None
        if n_max > 0:
            tail, tail_error = fit_hermitian_tail(g_in)
            expected += [moment_matrix(tail[k + 1], n) for k in range(1, n_max + 1)]
            sources += ["tail"] * n_max

        spectrum = matrixify(spectra[name].data)
        for i in range(n):
            diagonal = spectrum[:, i, i].real
            computed = np.array(
                [trapz(grid ** k * diagonal, grid) for k in range(len(expected))]
            )
            target = np.array([np.real(m[i, i]) for m in expected])
            scale = np.where(np.abs(target) > 1e-12, np.abs(target), 1.0)
            out[(name, i, i)] = {
                "computed": computed,
                "expected": target,
                "rel_error": np.abs(computed - target) / scale,
                "source": sources,
                "tail_error": tail_error,
            }
    return out


_COLUMNS = ("element", "alpha_opt", "chi2", "min chi2", "rms/err", "max/err", "m0",
            "expected", "rel", "source")
_ROW = "{:>14s} {:>10s} {:>10s} {:>10s} {:>8s} {:>8s} {:>10s} {:>10s} {:>9s}  {:<12s}"


@dataclass
class ValidationReport:
    ok: bool
    max_residual_over_error: float
    max_residual_rms: float
    residuals: dict
    moments: dict
    backtransform: object
    residual_threshold: float
    moment_threshold: float

    def __str__(self):
        head = _ROW.format(*_COLUMNS)
        lines = [head, "-" * len(head)]
        for key in sorted(self.residuals, key=lambda k: (str(k[0]), k[1], k[2])):
            d = self.residuals[key]
            m = self.moments.get(key)
            lines.append(
                _ROW.format(
                    "{}[{},{}]".format(*key),
                    _fmt(d["alpha_opt"]),
                    _fmt(d["chi2"]),
                    _fmt(d["chi2_min"]),
                    "{:.2f}".format(d["residual_rms"]),
                    "{:.2f}".format(d["residual_over_error"]),
                    _fmt(m["computed"][0]) if m else "-",
                    _fmt(m["expected"][0]) if m else "-",
                    _fmt(m["rel_error"][0]) if m else "-",
                    m["source"][0] if m else "-",
                )
            )
        lines.append(
            "{} (rms residual/error < {:g}, moment rel. error < {:g})".format(
                "PASS" if self.ok else "FAIL", self.residual_threshold, self.moment_threshold
            )
        )
        return "\n".join(lines)


def _fmt(x):
    if x is None:
        return "-"
    x = float(np.real(x))
    return "{:.4g}".format(x) if np.isfinite(x) else "-"


def validate(result, residual_threshold=2.0, moment_threshold=0.1, reference=None):
    """Run both checks and report. Never raises on a bad continuation."""
    inner, _ = _continuation(result)
    report = backtransform(result)
    moments = check_moments(result, reference=reference)

    residuals = {}
    for d in inner.diagnostics:
        # complex Matsubara data carries two degrees of freedom per point, so a
        # perfect fit sits near sqrt(2) here rather than 1
        scaled = np.abs((d.im_data - d.shift) - d.backtransform) / d.error
        residuals[d.key] = {
            "residual_over_error": float(np.max(scaled)),
            "residual_rms": float(np.sqrt(np.mean(scaled ** 2))),
            "alpha_opt": d.alpha_opt,
            "chi2": d.chi2,
            "chi2_min": d.chi2_min,
        }

    worst = max(v["residual_over_error"] for v in residuals.values())
    worst_rms = max(v["residual_rms"] for v in residuals.values())
    moment_ok = all(np.all(m["rel_error"] < moment_threshold) for m in moments.values())
    return ValidationReport(
        ok=bool(worst_rms < residual_threshold and moment_ok),
        max_residual_over_error=worst,
        max_residual_rms=worst_rms,
        residuals=residuals,
        moments=moments,
        backtransform=report,
        residual_threshold=residual_threshold,
        moment_threshold=moment_threshold,
    )
