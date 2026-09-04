# Changelog

## Version 4.0.0

triqs_ana_cont_interface version 4.0.0 is the initial release, compatible with
TRIQS version 4.0.0. It provides

* `gf_problem` / `sigma_problem` continuation records for `Gf` and `BlockGf`
  input on `MeshImFreq` and `MeshImTime`, fermionic, scalar- and matrix-valued
* `solve`, mapping each record onto ana_cont's MaxEnt solver and reassembling
  TRIQS containers on `MeshReFreq` / `MeshReFreqPts`
* diagonal, `poorman` and `eigenbasis` treatments of matrix-valued input
* `degenerate_blocks`, continuing one block per degeneracy group
* `backtransform`, `check_moments` and `validate` for judging a continuation
* HDF5 serialization of the problem, result and diagnostics records

We thank all contributors: Harrison LaBollita
