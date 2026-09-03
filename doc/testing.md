# Tests

```bash
python -m unittest discover -s test -v
python -m unittest test.test_sigma -v

python -m coverage run --source=triqs_ana_cont_interface -m unittest discover -s test
python -m coverage report -m
```

89 tests, ~11 s, 99% line coverage. Toy TRIQS objects with a known spectral function; assertions
are on physics, so there are no stored reference arrays to regenerate.

Line coverage is not the target: the suite is checked by mutation testing (flip a sign, transpose
an index, drop the `/pi`, remove the stride divisor search), and catches 7 of 8 such mutations.
The survivor -- reading `data[sel, j, i]` instead of `data[sel, i, j]` during task extraction --
is an equivalent mutant by construction: `poorman` is the only mode that creates off-diagonal
tasks and it enforces `g_ij == g_ji`, while `eigenbasis` creates diagonal-only tasks.

The six uncovered lines are the `except ImportError` fallbacks for a missing `h5` package and two
`continue`/property lines that no current field reaches.
