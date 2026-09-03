# Conventions applied internally

| | |
|---|---|
| Matsubara | Positive frequencies only; `im_axis` is the real-valued `omega_n`. |
| Imaginary time | Continues `-G(tau)`; TRIQS `G(tau) < 0` while the ana_cont kernel is positive. |
| Sigma constant | Subtracted from every element (diagonal and off-diagonal) and re-added after Kramers-Kronig. |
| Sigma model norm | The first moment, not 1. |
| Sigma moments | From the high-frequency average of `Re Sigma` and `-Im Sigma * w_n`, not from a tail fit. |
| Real-frequency mesh | `linear_grid` -> `MeshReFreq`; `tangent_grid` -> `MeshReFreqPts`, so a non-uniform grid needs no interpolation. |
| Batch | `interactive=False, verbose=False` always; no plot windows mid-scan. |

Warnings are raised for a non-hermitian input, error bars below 1e-8, an unset `n_iw` with more
than 300 positive frequencies available, discarded off-diagonal weight in `diagonal` mode, and a
chi2 far below the error bars (noiseless input or an underestimated error, where maxent overfits).

# HDF5

Results are registered with the TRIQS h5 protocol, so they store directly — no `.data` hop — and
reload complete enough that `validate` still works.

```python
with HDFArchive('cont.h5', 'w') as ar:
    ar['res'] = res
```

# Limitations

- Fermionic only. A bosonic mesh raises; the susceptibility convention differs enough to need its
  own tested pass.
- No Pade. `ana_cont/pade.pyx` needs Cython and a build step.
- Off-diagonal spectra are real, because `ana_cont` returns a real spectral function per element.
  A genuinely complex-hermitian off-diagonal element is not representable this way; use
  `mode='eigenbasis'`.
- DLR meshes are rejected with a message telling you to convert first.
- The continuation loop over elements is serial. It is embarrassingly parallel.
