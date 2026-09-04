import os
import tempfile
import unittest

import numpy as np
from h5 import HDFArchive
from triqs.gfs import BlockGf

import toy
from triqs_ana_cont_interface import (
    backtransform,
    gf_problem,
    linear_grid,
    sigma_problem,
    solve,
    tangent_grid,
    validate,
)


def _flat(container):
    if isinstance(container, BlockGf):
        return np.concatenate([g.data.ravel() for _, g in container])
    return container.data.ravel()


def _roundtrip(result):
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "c.h5")
        with HDFArchive(path, "w") as archive:
            archive["result"] = result
        with HDFArchive(path, "r") as archive:
            return archive["result"]


class TestArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w = toy.real_axis()
        cls.results = {
            "scalar": solve(
                gf_problem(toy.gf_imfreq(toy.two_peaks(w)), grid=linear_grid(-8, 8, 201),
                           error=toy.NOISE, n_iw=20)
            ),
            "block": solve(
                gf_problem(toy.block_gf_imfreq(spectra=[toy.two_peaks(w), toy.one_peak(w)]),
                           grid=tangent_grid(20.0, 201), error=toy.NOISE, n_iw=20)
            ),
            "imtime": solve(
                gf_problem(toy.gf_imtime(toy.two_peaks(w)), grid=linear_grid(-8, 8, 201),
                           error=toy.NOISE, n_tau=201)
            ),
            "sigma": solve(
                sigma_problem(toy.gf_imfreq(1.7 * toy.one_peak(w, 0.0, 1.5), shift=2.5),
                              grid=linear_grid(-20, 20, 201), error=toy.NOISE,
                              n_iw=20, sigma_inf=2.5, model_norm=1.7)
            ),
        }

    def test_main_result_survives(self):
        for name, result in self.results.items():
            with self.subTest(name):
                back = _roundtrip(result)
                field = "sigma_w" if name == "sigma" else "g_w"
                np.testing.assert_allclose(
                    _flat(getattr(back, field)), _flat(getattr(result, field))
                )
                self.assertEqual(type(back), type(result))

    def test_diagnostics_survive_with_their_keys(self):
        back = _roundtrip(self.results["block"])
        self.assertIsInstance(back.diagnostics, tuple)
        self.assertEqual(
            {d.key for d in back.diagnostics},
            {d.key for d in self.results["block"].diagnostics},
        )
        original = self.results["block"].diag("up", 0, 0)
        restored = back.diag("up", 0, 0)
        self.assertAlmostEqual(restored.alpha_opt, original.alpha_opt)
        np.testing.assert_allclose(restored.alpha_scan, original.alpha_scan)
        np.testing.assert_allclose(restored.error, original.error)

    def test_validation_still_works_after_reload(self):
        for name, result in self.results.items():
            with self.subTest(name):
                back = _roundtrip(result)
                self.assertEqual(
                    validate(back).ok, validate(result).ok
                )
                np.testing.assert_allclose(
                    _flat(backtransform(back)), _flat(backtransform(result))
                )

    def test_problem_shapes_are_tuples_after_reload(self):
        back = _roundtrip(self.results["scalar"])
        self.assertEqual(back.problem.layout.target_shapes["0"], ())
        self.assertIsInstance(back.problem.tasks, tuple)


if __name__ == "__main__":
    unittest.main()
