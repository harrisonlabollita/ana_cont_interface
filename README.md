# triqs_ana_cont_interface

A TRIQS front-end for [`ana_cont`](https://github.com/josefkaufmann/ana_cont). Give it a `Gf` or
`BlockGf` on an imaginary mesh, get the continued object back on a real-frequency mesh, with the
physics conventions applied once and the diagnostics needed to judge the result attached to the
output.

## Install

Needs `triqs` and `ana_cont` importable; neither is pip-installable, so neither is declared as a
dependency.

```bash
pip install -e .
```

## Usage

Build a problem record, then solve it. The record is data — inspect it, edit one element's model
or error bar, solve it repeatedly with different settings.

```python
from triqs_ana_cont_interface import gf_problem, solve, linear_grid, validate

prob = gf_problem(g_iw, grid=linear_grid(-10, 10, 501), error=1e-4, n_iw=60)
res  = solve(prob, preblur=0.5)

res.g_w                  # Gf / BlockGf on MeshReFreq, complex retarded
res.a_w                  # the spectral matrix A = (i/2pi)(g_w - g_w^dag)
print(validate(res))     # per-element table of fit quality and spectral weight
```

Self-energies go through `sigma_problem`, which handles `Sigma_inf` and the model norm:

```python
from triqs_ana_cont_interface import sigma_problem, tangent_grid

res = solve(sigma_problem(sigma_iw, grid=tangent_grid(20, 501), error=err, n_iw=400))
res.sigma_w
```

`error` is required. ana_cont has no error estimator, and a defaulted error bar produces a
plausible-looking but meaningless spectrum.

## Documentation

- [`doc/usage.md`](doc/usage.md) — imaginary time, self-energies, matrix-valued input,
  degenerate blocks, the preblur scan, solver options
- [`doc/validation.md`](doc/validation.md) — `backtransform`, `check_moments`, `validate`
- [`doc/conventions.md`](doc/conventions.md) — what is applied internally, warnings, HDF5,
  limitations
- [`doc/testing.md`](doc/testing.md) — running the tests

## Status

Fermionic `MeshImFreq` and `MeshImTime`, Green's functions and self-energies, diagonal and
matrix-valued. No bosonic kernels and no Pade yet. Verified against real SrVO3 CTHYB data: the
quasiparticle weight from the continued `Sigma(w)` agrees with the Matsubara estimate to under 1%.

## License

MIT, the same as [`ana_cont`](https://github.com/josefkaufmann/ana_cont).
