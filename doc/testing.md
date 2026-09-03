# Tests

```bash
python -m unittest discover -s test -v
python -m unittest test.test_sigma -v

python -m coverage run --source=triqs_ana_cont_interface -m unittest discover -s test
python -m coverage report -m
```
