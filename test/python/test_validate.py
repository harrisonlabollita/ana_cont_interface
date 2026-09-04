import unittest

import numpy as np
from triqs.mesh import MeshImFreq

import toy
from triqs_ana_cont_interface import (
    backtransform,
    check_moments,
    gf_problem,
    linear_grid,
    sigma_problem,
    solve,
    validate,
)

N_USED = 30


class TestBacktransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = toy.gf_imfreq(toy.two_peaks(toy.real_axis()))
        cls.res = solve(
            gf_problem(cls.g, grid=linear_grid(-8, 8, 201), error=toy.NOISE, n_iw=N_USED)
        )
        cls.bt = backtransform(cls.res)

    def test_mesh_covers_exactly_the_fitted_points(self):
        self.assertIsInstance(self.bt.mesh, MeshImFreq)
        self.assertEqual(self.bt.mesh.n_iw, N_USED)
        self.assertAlmostEqual(self.bt.mesh.beta, toy.BETA)

    def test_negative_half_is_mirrored(self):
        np.testing.assert_allclose(
            self.bt.data[:N_USED], np.conj(self.bt.data[N_USED:])[::-1], atol=1e-12
        )

    def test_it_fits_the_input(self):
        used = self.g.data[toy.N_IW : toy.N_IW + N_USED]
        deviation = np.max(np.abs(used - self.bt.data[N_USED:])) / toy.NOISE
        self.assertLess(deviation, 5.0, msg="max residual/error = {:.2f}".format(deviation))

    def test_sigma_backtransform_includes_the_constant(self):
        g = toy.gf_imfreq(1.7 * toy.one_peak(toy.real_axis(), 0.0, 1.5), shift=2.5)
        res = solve(
            sigma_problem(g, grid=linear_grid(-20, 20, 201), error=toy.NOISE,
                          n_iw=N_USED, sigma_inf=2.5, model_norm=1.7)
        )
        bt = backtransform(res)
        used = g.data[toy.N_IW : toy.N_IW + N_USED]
        deviation = np.max(np.abs(used - bt.data[N_USED:])) / toy.NOISE
        self.assertLess(deviation, 5.0, msg="max residual/error = {:.2f}".format(deviation))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = toy.gf_imfreq(toy.two_peaks(toy.real_axis()))
        cls.res = solve(
            gf_problem(g, grid=linear_grid(-8, 8, 201), error=toy.NOISE, n_iw=N_USED),
            preblur=0.4,
        )

    def test_tight_residual_threshold_fails(self):
        self.assertFalse(validate(self.res, residual_threshold=1e-3).ok)

    def test_wrong_reference_fails_the_moment_check(self):
        self.assertFalse(validate(self.res, reference=0.5).ok)

    def test_report_prints_a_table(self):
        text = str(validate(self.res))
        self.assertIn("0[0,0]", text)
        self.assertIn("PASS", text)
        self.assertIn("sum-rule", text)

    def test_higher_moments_come_from_the_tail(self):
        entry = check_moments(self.res, n_max=1)[("0", 0, 0)]
        self.assertEqual(entry["source"], ["sum-rule", "tail"])
        self.assertIsNotNone(entry["tail_error"])


class TestEdgeCases(unittest.TestCase):
    def test_a_result_without_its_problem_says_so(self):
        from triqs_ana_cont_interface import ContinuationResult

        res = ContinuationResult(g_w=None, a_w=None, diagnostics={}, problem=None)
        for fn in (backtransform, check_moments):
            with self.subTest(fn.__name__), self.assertRaises(ValueError) as caught:
                fn(res)
            self.assertIn("result.problem", str(caught.exception))

    def test_eigenbasis_sigma_on_a_scalar_target(self):
        w = toy.real_axis()
        g = toy.gf_imfreq(1.7 * toy.one_peak(w, 0.0, 1.5), shift=2.5)
        prob = sigma_problem(
            g, grid=linear_grid(-20, 20, 201), error=toy.NOISE, n_iw=20,
            mode="eigenbasis", rotation=np.ones((1, 1)),
        )
        self.assertAlmostEqual(prob.task("0").shift.real, 2.5, places=3)
        self.assertTrue(validate(solve(prob)).ok)


if __name__ == "__main__":
    unittest.main()
