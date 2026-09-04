# Tests

The tests run under `ctest` from the build directory:

```bash
cd ana_cont_interface.build
ctest --output-on-failure
ctest -R test_sigma --output-on-failure
```

Each file is also a standalone `unittest` module, so a single test can be run directly
as long as the package is on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/ana_cont_interface.build/python:$PYTHONPATH
python ../ana_cont_interface.src/test/python/test_sigma.py -v
python -m unittest discover -s ../ana_cont_interface.src/test/python -v
```

For coverage:

```bash
python -m coverage run --source=triqs_ana_cont_interface \
    -m unittest discover -s ../ana_cont_interface.src/test/python
python -m coverage report -m
```
