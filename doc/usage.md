# Usage

See the README for installation and the shortest example. Build a problem record, then solve
it; the record is data, so you can inspect it, edit a single element's model or error bar, and
solve it repeatedly with different settings.

```python
from triqs_ana_cont_interface import gf_problem, solve, linear_grid, validate

prob = gf_problem(g_iw, grid=linear_grid(-10, 10, 501), error=1e-4, n_iw=60)
res  = solve(prob, alpha_determination='chi2kink', preblur=0.5)

res.g_w                  # Gf / BlockGf on MeshReFreq, complex retarded
res.a_w                  # the spectral matrix A = (i/2pi)(g_w - g_w^dag)
res.diag('up', 0, 0)     # per-element diagnostics
print(validate(res))     # per-element table of fit quality and spectral weight
```

`error` is required — a scalar, a 1-D array, a dict keyed by block, or a `Gf`/`BlockGf` of the
same structure. `ana_cont` has no error estimator, and a defaulted error bar produces a
plausible-looking but meaningless spectrum.

## The preblur scan

One problem, many solves. Selection criterion: the largest blur width that does not degrade
`min(chi2)`.

```python
for b in (0.0, 0.3, 0.5, 1.0, 1.5):
    res = solve(prob, preblur=b)
    print(b, res.diag('up', 0, 0).chi2_min, validate(res).max_residual_rms)
```

`preblur` is frequency-only: ana_cont convolves the `freq_*` kernels alone, so it is
rejected for an imaginary-time problem rather than failing inside the alpha loop.

## Imaginary time

```python
prob = gf_problem(g_tau, grid=linear_grid(-8, 8, 301), error=1e-4, n_tau=201)
```

`n_tau` subsamples with a stride that divides the interval evenly, so the reduced grid is itself
a `MeshImTime`. The `-G(tau)` sign convention is applied internally, and the default model is
divided by `beta`: ana_cont rescales `re_axis` by `beta` for time kernels and integrates the model
against the rescaled axis, so a model normalized on the physical grid would otherwise assert
`beta` times the intended spectral weight.

## Self-energy

```python
from triqs_ana_cont_interface import sigma_problem, tangent_grid

prob = sigma_problem(sigma_iw, grid=tangent_grid(wmax=20, n_w=501), error=sigma_err, n_iw=250)
res  = solve(prob, preblur=0.3)

res.sigma_w        # BlockGf on MeshReFreqPts -- Sigma(w)
res.sigma_inf      # the constant that was subtracted and re-added
res.first_moment   # the model norm that was used
res.aux            # the underlying continuation of Sigma - Sigma_inf
```

`sigma_inf=` and `model_norm=` accept a scalar, a length-N vector, or an N x N matrix per block
(a dict keyed by block also works). A scalar or vector goes on the **diagonal**, as TRIQS does
for square targets — a constant self-energy shift is diagonal, not a matrix of equal entries.
They are physical quantities in the basis of the *input*: under `mode='eigenbasis'` they are
rotated into the working basis on the way in and back out again, so `res.sigma_inf` and
`res.first_moment` always match `res.sigma_w`.

`Sigma_inf` is hermitian, so its off-diagonal elements are complex and are kept complex. A
non-positive diagonal first moment is warned about rather than passed through `abs()`, since
`A_Sigma` integrates to a positive number and a negative estimate means the estimate is wrong.

**Choosing `n_iw` for a self-energy.** The spectral weight of `A_Sigma` is fixed by the `1/(i w_n)`
tail, so truncating `n_iw` too aggressively loosens the constraint on it — the fit gets better
while the weight drifts. Measured on real SrVO3 CTHYB data (beta=40, 1025 frequencies):

| `n_iw` | `w_n` max | rms residual/error | moment error | Z |
|---|---|---|---|---|
| 120  |  18.8 | 0.79 | 18.3% | 0.682 |
| 300  |  47.0 | 1.72 | 12.9% | 0.679 |
| 600  |  94.2 | 2.06 |  9.5% | 0.678 |
| 1025 | 160.9 | 1.92 |  7.4% | 0.677 |

`check_moments` is what makes that trade visible; the quasiparticle weight is barely affected
either way. For a Green's function the sum rule is enforced by the kernel itself and the same
pressure does not apply.

`Sigma_inf` and the first moment are read off the high-frequency behaviour unless you pass
`sigma_inf=` / `model_norm=`:

    Re Sigma(i w_n) -> Sigma_inf        -Im Sigma(i w_n) * w_n -> first moment

each averaged over the top decile of Matsubara frequencies. `fit_hermitian_tail` is *not* used
here: it is dominated by noise in exactly that region. On this package's toy self-energy it
returns `Sigma_inf` = 1.19 (true 2.5) and a **negative** first moment at 1e-3 noise, which is not
a usable norm; the averages above stay within 2e-4 and 2% respectively across 1e-5 to 1e-3 noise.
Subtracting a wrong constant is fatal — a constant is not in the span of the kernel, so chi2
explodes rather than degrading gracefully.

The first moment sets the *norm of the default model*, because `A_Sigma` integrates to the first
moment, not to 1.

## Matrix-valued

```python
solve(gf_problem(g_iw, ..., mode='diagonal'))    # default; warns with the discarded weight
solve(gf_problem(g_iw, ..., mode='poorman'))     # diagonals, then off-diagonals
solve(gf_problem(g_iw, ..., mode='eigenbasis', rotation=states))
```

- `diagonal` continues `g[i, i]` only and warns, naming `max|g_ij| / max|g_ii|`.
- `poorman` continues the upper triangle with `offdiag=True` and the geometric-mean default
  model `sqrt(A_ii A_jj)` (Kraberger et al.), filling the lower triangle by symmetry. It
  **requires a symmetric input**, which is stronger than hermitian: ana_cont returns one real
  spectrum per element, and a real `A_ij` forces `g_ij == g_ji`. A hermitian block with an
  imaginary `A_ij` is antisymmetric instead, and continuing `g_ij` alone would fit the data with
  a large oscillating spectrum that no residual check can detect — so the asymmetry is measured
  against the error bars and rejected. The requirement applies after the constant is subtracted,
  so a complex off-diagonal `Sigma_inf` is fine.
- `eigenbasis` rotates with a supplied unitary, continues the N diagonals, and rotates back.
  The rotation must be given: the tail coefficients it would otherwise come from are destroyed
  by noise. If the rotation does not actually diagonalize the block, the discarded off-diagonal
  weight is reported, as in `diagonal` mode.

`a_w` is the hermitian spectral matrix `A = (i/2pi)(g_w - g_w^dag)`. Elementwise `-Im g_w/pi`
is only equal to it when the off-diagonal spectra are real, which fails after a complex
`eigenbasis` rotation; for a scalar or real-symmetric target the two agree exactly.

All blocks of a `BlockGf` must share one mesh, since a single frequency selection is applied to
every block.

## Solver options

`alpha_determination` accepts `'chi2kink'` (default), `'classic'`, `'historic'` and `'bryan'`;
`optimizer` accepts `'newton'` (default) and `'scipy_lm'`. Both are validated up front, because
an invalid value otherwise fails inside ana_cont's per-alpha `except` and surfaces as an
`IndexError` on an empty result list. `'bryan'` averages over alpha, so `alpha_opt` and `chi2` are
`None` in the diagnostics and render as `-`. `'scipy_lm'` is roughly 40x slower than `'newton'`.

`chi2kink` fits a four-parameter function to `log(chi2)` vs `log(alpha)`, so the alpha range must
yield at least 4 points; a narrower range is rejected with that explanation rather than a
`curve_fit` type error.
