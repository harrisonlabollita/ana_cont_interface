import dataclasses
import unittest

import numpy as np

import toy
from triqs_ana_cont_interface import (
    flat_model,
    gaussian_model,
    gf_problem,
    linear_grid,
    poorman_model,
    sigma_problem,
    solve,
    super_gaussian_model,
    validate,
)
from triqs_ana_cont_interface.problem import moment_matrix

_trapz = getattr(np, "trapezoid", np.trapz)


class TestAlphaDetermination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, 201)
        cls.prob = gf_problem(
            toy.gf_imfreq(toy.two_peaks(toy.real_axis())),
            grid=cls.grid, error=toy.NOISE, n_iw=30,
        )

    def test_every_mode_runs_and_normalizes(self):
        for mode in ("classic", "historic"):
            with self.subTest(mode):
                res = solve(self.prob, alpha_determination=mode)
                self.assertAlmostEqual(_trapz(res.a_w.data, self.grid.values), 1.0, places=2)

    def test_bryan_reports_no_single_alpha_without_crashing(self):
        # bryan averages over alpha, so alpha_opt and chi2 are undefined
        res = solve(self.prob, alpha_determination="bryan")
        diagnostics = res.diag("0")
        self.assertIsNone(diagnostics.alpha_opt)
        self.assertIsNone(diagnostics.chi2)
        report = validate(res)
        self.assertIn("-", str(report))
        self.assertTrue(report.ok)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            solve(self.prob, alpha_determination="magic")

    def test_unknown_optimizer_is_rejected(self):
        with self.assertRaises(ValueError):
            solve(self.prob, optimizer="magic")

    def test_bad_alpha_range_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            solve(self.prob, alpha_start=1e3, alpha_end=1e2)
        self.assertIn("at least 4", str(caught.exception))
        with self.assertRaises(ValueError):
            solve(self.prob, alpha_start=1e-3, alpha_end=1e9)

class TestModels(unittest.TestCase):
    def setUp(self):
        self.grid = linear_grid(-10, 10, 201)

    def test_normalization(self):
        for model in (
            flat_model(self.grid, norm=2.5),
            gaussian_model(self.grid, center=1.0, width=2.0, norm=2.5),
            super_gaussian_model(self.grid, width=6.0, norm=2.5),
        ):
            with self.subTest(model=model[:1]):
                self.assertAlmostEqual(_trapz(model, self.grid.values), 2.5)
                self.assertTrue(np.all(model > 0.0))

    def test_poorman_model_is_strictly_positive(self):
        w = self.grid.values
        model = poorman_model(toy.two_peaks(w), np.zeros_like(w))
        self.assertTrue(np.all(model > 0.0))

    def test_per_task_model_override(self):
        prob = gf_problem(
            toy.gf_imfreq(toy.two_peaks(toy.real_axis())),
            grid=self.grid, error=toy.NOISE, n_iw=20,
        )
        model = gaussian_model(self.grid, width=3.0)
        prob.tasks = tuple(dataclasses.replace(t, model=model) for t in prob.tasks)
        self.assertIsNotNone(prob.task("0").model)
        res = solve(prob)
        self.assertAlmostEqual(_trapz(res.a_w.data, self.grid.values), 1.0, places=1)


class TestErrorForms(unittest.TestCase):
    def setUp(self):
        self.grid = linear_grid(-8, 8, 201)
        self.gb = toy.block_gf_imfreq(spectra=[toy.two_peaks(toy.real_axis())])

    def test_dict_keyed_by_block(self):
        prob = gf_problem(self.gb, grid=self.grid, error={"up": 1e-4, "dn": 2e-4}, n_iw=20)
        self.assertAlmostEqual(prob.task("up", 0, 0).error[0], 1e-4)
        self.assertAlmostEqual(prob.task("dn", 0, 0).error[0], 2e-4)

    def test_block_gf(self):
        error = self.gb.copy()
        error["up"].data[:] = 1e-4
        error["dn"].data[:] = 3e-4
        prob = gf_problem(self.gb, grid=self.grid, error=error, n_iw=20)
        self.assertAlmostEqual(prob.task("up", 0, 0).error[0], 1e-4)
        self.assertAlmostEqual(prob.task("dn", 0, 0).error[0], 3e-4)

    def test_full_mesh_array_is_sliced(self):
        g = toy.gf_imfreq(toy.two_peaks(toy.real_axis()))
        prob = gf_problem(g, grid=self.grid, error=np.full(2 * toy.N_IW, 5e-4), n_iw=20)
        self.assertEqual(len(prob.task("0").error), 20)
        self.assertAlmostEqual(prob.task("0").error[0], 5e-4)

    def test_two_dimensional_array_rejected(self):
        g = toy.gf_imfreq(toy.two_peaks(toy.real_axis()))
        with self.assertRaises(ValueError):
            gf_problem(g, grid=self.grid, error=np.ones((5, 5)))

    def test_wrong_length_array_rejected(self):
        g = toy.gf_imfreq(toy.two_peaks(toy.real_axis()))
        with self.assertRaises(ValueError):
            gf_problem(g, grid=self.grid, error=np.ones(7), n_iw=20)


class TestMomentBroadcast(unittest.TestCase):
    def test_scalar_goes_on_the_diagonal(self):
        np.testing.assert_allclose(moment_matrix(2.5, 2), [[2.5, 0.0], [0.0, 2.5]])

    def test_vector_goes_on_the_diagonal(self):
        np.testing.assert_allclose(moment_matrix(np.array([2.5, 1.0]), 2), [[2.5, 0.0], [0.0, 1.0]])

    def test_bad_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            moment_matrix(np.ones(5), 2)

    def test_scalar_sigma_inf_on_a_matrix_block(self):
        w = toy.real_axis()
        gb = toy.block_gf_imfreq(spectra=[1.7 * toy.one_peak(w), 1.2 * toy.one_peak(w)], shift=2.5)
        prob = sigma_problem(
            gb, grid=linear_grid(-20, 20, 201), error=toy.NOISE, n_iw=20, mode="poorman",
            sigma_inf={"up": 2.5, "dn": 1.0}, model_norm={"up": 1.7, "dn": 1.2},
        )
        self.assertAlmostEqual(prob.task("up", 0, 0).shift.real, 2.5)
        self.assertAlmostEqual(prob.task("up", 1, 1).shift.real, 2.5)
        self.assertAlmostEqual(prob.task("up", 0, 1).shift.real, 0.0)
        self.assertAlmostEqual(prob.task("dn", 0, 0).shift.real, 1.0)
        self.assertAlmostEqual(prob.task("up", 0, 0).model_norm, 1.7)

if __name__ == "__main__":
    unittest.main()
