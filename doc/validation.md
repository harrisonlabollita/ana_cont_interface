# Validation

Two checks, each usable on its own, plus `validate` which runs both.

```python
bt = backtransform(res)   # K.A as a Gf/BlockGf on the input mesh, restricted to fitted points
oplot(g_iw['up'][0, 0]); oplot(bt['up'][0, 0])

check_moments(res)        # int dw w^n A(w) against the expected weight
validate(res).ok          # bool; .max_residual_rms, .moments, .backtransform
```

`validate` never raises on a bad continuation — it reports. `.ok` uses the rms residual over the
error bars (that is `sqrt(chi2/N)`) rather than the max, which is a max-statistic over many
points.

The m0 reference avoids a tail fit wherever an exact one exists, because `fit_hermitian_tail`
degrades badly with noise:

| input | m0 reference | `source` |
|---|---|---|
| Green's function, Matsubara | 1, the anticommutator sum rule | `sum-rule` |
| Green's function, imaginary time | `-(G(0) + G(beta))` | `edge` |
| self-energy | the first moment that set the model norm | `first-moment` |
| `reference=` given | your value | `user` |

Higher moments (`n_max > 0`) have no such sum rule and do come from the tail fit; its error is
then reported as `tail_error`. They need a Matsubara input, since `fit_hermitian_tail` does not
accept an imaginary-time mesh.
