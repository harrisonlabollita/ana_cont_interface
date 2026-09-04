"""Map a ContinuationProblem onto ana_cont and reassemble TRIQS containers."""

import warnings
from dataclasses import replace

import numpy as np

import ana_cont.continuation as cont

from ._util import build_container, moment_matrix, norb, pick
from .models import flat_model, poorman_model
from .result import ContinuationResult, SigmaResult, TaskDiagnostics, _lookup

ALPHA_DETERMINATION = ("chi2kink", "classic", "historic", "bryan")
OPTIMIZERS = ("newton", "scipy_lm")


def solve(problem, alpha_determination="chi2kink", optimizer="newton", preblur=0.0, **kwargs):
    """Solve every task of `problem` and reassemble the result.

    Extra keyword arguments are forwarded to ana_cont (alpha_start, alpha_end,
    alpha_div, fit_position, ...).
    """
    if alpha_determination not in ALPHA_DETERMINATION:
        raise ValueError(
            "alpha_determination must be one of {}, got {!r}".format(
                ALPHA_DETERMINATION, alpha_determination
            )
        )
    if optimizer not in OPTIMIZERS:
        raise ValueError("optimizer must be one of {}, got {!r}".format(OPTIMIZERS, optimizer))
    if preblur > 0.0 and not problem.recipe.is_freq:
        raise ValueError(
            "preblur is only implemented for frequency kernels (ana_cont's "
            "kernels.py convolves freq_fermionic and freq_bosonic only)"
        )
    if alpha_determination == "chi2kink":
        _check_alpha_range(kwargs)

    options = dict(
        alpha_determination=alpha_determination,
        optimizer=optimizer,
        preblur=preblur > 0.0,
        blur_width=preblur,
        **kwargs
    )

    # a stable sort puts every diagonal task before every off-diagonal one, so
    # the poorman models can be built from the diagonal spectra as we go, and
    # every continued block before the blocks that copy from it
    source = problem.recipe.representative
    spectra, diagnostics = {}, []
    for task in sorted(problem.tasks, key=lambda t: (t.offdiag, t.block in source)):
        if task.block in source:
            spectra[task.key] = spectra[(source[task.block], task.i, task.j)]
            diagnostics.append(_copied_record(task, diagnostics, source[task.block]))
            continue
        model = task.model
        if model is None:
            model = (
                poorman_model(
                    spectra[(task.block, task.i, task.i)],
                    spectra[(task.block, task.j, task.j)],
                )
                if task.offdiag
                else flat_model(problem.grid, norm=task.model_norm)
            )
        spectrum, record = _solve_task(task, problem, model, options)
        spectra[task.key] = spectrum
        diagnostics.append(record)
    return _assemble(problem, spectra, tuple(diagnostics))


def _copied_record(task, diagnostics, source):
    """Diagnostics for a block copied from a degenerate one.

    The solution is the source's, but the data, error bars and constant are the
    copy's own, so validate() measures the fit and the degeneracy claim together.
    """
    record = _lookup(diagnostics, source, task.i, task.j)
    return replace(
        record,
        block=task.block,
        im_data=task.im_data,
        error=task.error,
        shift=task.shift,
        copied_from=source,
    )


def _check_alpha_range(kwargs):
    """chi2kink fits a four-parameter function to log(chi2) vs log(alpha).

    The defaults mirror ana_cont's solve_chi2kink signature.
    """
    start = kwargs.get("alpha_start", 1e9)
    end = kwargs.get("alpha_end", 1e-3)
    div = kwargs.get("alpha_div", 10.0)
    if not (start > end > 0.0 and div > 1.0):
        raise ValueError(
            "need alpha_start > alpha_end > 0 and alpha_div > 1, got "
            "alpha_start={}, alpha_end={}, alpha_div={}".format(start, end, div)
        )
    n_alpha = int(np.floor(np.log(start / end) / np.log(div))) + 1
    if n_alpha < 4:
        raise ValueError(
            "alpha_start={}, alpha_end={}, alpha_div={} gives only {} alpha points; "
            "chi2kink needs at least 4 to fit its four-parameter kink. Widen the "
            "range or lower alpha_div.".format(start, end, div, n_alpha)
        )


def model_for(model, problem):
    """The default model as ana_cont will integrate it.

    For time kernels ana_cont rescales re_axis by beta and integrates the model
    against the rescaled axis, so a model normalized on the physical grid would
    assert beta times the intended spectral weight.
    """
    return model if problem.recipe.is_freq else model / problem.layout.beta


def _solve_task(task, problem, model, options):
    beta = problem.layout.beta
    model = model_for(model, problem)
    scalar_problem = cont.AnalyticContinuationProblem(
        im_axis=task.im_axis,
        re_axis=problem.grid.values,
        im_data=task.im_data - task.shift,
        kernel_mode=problem.recipe.kernel_mode,
        beta=beta,
    )
    sol, scan = scalar_problem.solve(
        method="maxent_svd",
        model=model,
        stdev=task.error,
        offdiag=task.offdiag,
        interactive=False,
        verbose=False,
        **options
    )
    if sol.chi2 is not None and sol.chi2 < 1e-3 * len(task.im_axis):
        warnings.warn(
            "{}: chi2 = {:.3g} for {} data points, far below the error bars. The input "
            "may be noiseless or the error underestimated; maxent needs some noise to "
            "work against and will overfit here.".format(task.key, sol.chi2, len(task.im_axis))
        )
    finite = [o for o in scan if np.isfinite(o.chi2)]
    record = TaskDiagnostics(
        block=task.block,
        i=task.i,
        j=task.j,
        alpha_opt=sol.alpha,
        chi2=sol.chi2,
        chi2_min=min((o.chi2 for o in finite), default=None),
        entropy=sol.entropy,
        norm=sol.norm,
        blur_width=sol.blur_width,
        n_sv=int(scalar_problem.solver.n_sv),
        shift=task.shift,
        alpha_scan=np.array([[np.log10(o.alpha), np.log10(o.chi2)] for o in finite]),
        im_axis=task.im_axis,
        im_data=task.im_data,
        error=task.error,
        backtransform=sol.backtransform,
    )
    return sol.A_opt, record


def _kkt(spectrum, w):
    """Kramers-Kronig: the retarded function whose spectrum is `spectrum`."""
    return cont.GreensFunction(spectrum=spectrum, wgrid=w, kind="fermionic").kkt()


def _rotate_back(data, u):
    """U M U^dag on the target indices, the inverse of problem._rotate."""
    return np.einsum("ia,wab,jb->wij", u, data, u.conj(), optimize=True)


def _spectral_matrix(data):
    """A = (i/2pi)(G - G^dag), the hermitian part of the retarded function.

    Elementwise -Im G / pi is only equal to this when the off-diagonal spectra
    are real, which fails after a complex eigenbasis rotation.
    """
    return (1j / (2.0 * np.pi)) * (data - np.conj(np.swapaxes(data, 1, 2)))


def _block_matrices(problem, spectra):
    """Per block: the continued object, and the constant to add to it."""
    grid = problem.grid
    layout = problem.layout
    shifts = {t.key: t.shift for t in problem.tasks}
    continued, constant = {}, {}
    for name in layout.block_names:
        n = norb(layout.target_shapes[name])
        raw = np.zeros((len(grid), n, n), dtype=complex)
        const = np.zeros((n, n), dtype=complex)
        for key, spectrum in spectra.items():
            if key[0] != name:
                continue
            _, i, j = key
            raw[:, i, j] = _kkt(spectrum, grid.values)
            const[i, j] = shifts[key]
            if i != j:
                # a real spectrum gives G_ji = G_ij; the constant is hermitian
                raw[:, j, i] = raw[:, i, j]
                const[j, i] = np.conj(const[i, j])
        if problem.recipe.mode == "eigenbasis":
            u = problem.recipe.rotation[name]
            raw = _rotate_back(raw, u)
            const = _rotate_back(const[None], u)[0]
        continued[name], constant[name] = raw, const
    return continued, constant


def _report_shape(matrix, target_shape):
    """User-facing moments follow the container: scalar target, scalar moment."""
    return matrix if tuple(target_shape) else matrix[0, 0]


def _first_moments(problem, constant):
    moments = {}
    for name in problem.layout.block_names:
        shape = problem.layout.target_shapes[name]
        moment = moment_matrix(pick(problem.recipe.first_moment, name), norb(shape))
        if problem.recipe.mode == "eigenbasis":
            moment = _rotate_back(moment[None], problem.recipe.rotation[name])[0]
        moments[name] = _report_shape(moment, shape)
    return moments


def _assemble(problem, spectra, diagnostics):
    continued, constant = _block_matrices(problem, spectra)
    mesh = problem.grid.mesh
    layout = problem.layout
    a_w = build_container(
        mesh, layout, {k: _spectral_matrix(v) for k, v in continued.items()}
    )
    full = {k: continued[k] + constant[k][None] for k in continued}

    if problem.recipe.quantity == "gf":
        return ContinuationResult(
            g_w=build_container(mesh, layout, full),
            a_w=a_w,
            diagnostics=diagnostics,
            problem=problem,
        )
    aux = ContinuationResult(
        g_w=build_container(mesh, layout, continued),
        a_w=a_w,
        diagnostics=diagnostics,
        problem=problem,
    )
    return SigmaResult(
        sigma_w=build_container(mesh, layout, full),
        sigma_inf={
            k: _report_shape(v, layout.target_shapes[k]) for k, v in constant.items()
        },
        first_moment=_first_moments(problem, constant),
        aux=aux,
    )
