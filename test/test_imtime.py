import unittest

import numpy as np
from triqs.mesh import MeshImTime

import toy
from triqs_ana_cont_interface import (
    backtransform,
    check_moments,
    gf_problem,
    linear_grid,
    solve,
    validate,
)
from triqs_ana_cont_interface.models import flat_model
from triqs_ana_cont_interface.solve import model_for

_trapz = getattr(np, "trapezoid", np.trapz)


class TestImTime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = toy.real_axis()
        cls.spec = toy.two_peaks(cls.w)
        cls.g_tau = toy.gf_imtime(cls.spec, n_tau=1001)
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, 201)
        cls.res = solve(gf_problem(cls.g_tau, grid=cls.grid, error=toy.NOISE, n_tau=201))

    def test_sign_convention_gives_a_positive_normalized_spectrum(self):
        a = self.res.a_w.data
        self.assertGreater(a.min(), -1e-6)
        self.assertAlmostEqual(_trapz(a, self.grid.values), 1.0, places=2)

    def test_subsampling_lands_on_a_real_mesh(self):
        bt = backtransform(self.res)
        self.assertIsInstance(bt.mesh, MeshImTime)
        self.assertEqual(len(bt.mesh), 201)
        self.assertAlmostEqual(float(list(bt.mesh)[-1].value), toy.BETA, places=10)

    def test_moment_reference_is_the_edge_sum_rule(self):
        entry = check_moments(self.res)[("0", 0, 0)]
        self.assertEqual(entry["source"][0], "edge")
        self.assertAlmostEqual(entry["expected"][0], 1.0, places=3)
        self.assertLess(entry["rel_error"][0], 0.02)

    def test_validates(self):
        report = validate(self.res)
        self.assertTrue(report.ok, msg="\n" + str(report))

    def test_agrees_with_the_frequency_representation(self):
        g_iw = toy.gf_imfreq(self.spec, n_iw=100)
        res_iw = solve(gf_problem(g_iw, grid=self.grid, error=toy.NOISE, n_iw=40))
        difference = _trapz(np.abs(res_iw.a_w.data - self.res.a_w.data), self.grid.values)
        self.assertLess(difference, 0.15, msg="int|A_iw - A_tau| = {:.4f}".format(difference))


class TestImTimeContract(unittest.TestCase):
    """The tau path's own conventions, which the frequency path cannot check."""

    @classmethod
    def setUpClass(cls):
        cls.w = toy.real_axis()
        cls.g_tau = toy.gf_imtime(toy.two_peaks(cls.w), n_tau=1001)
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, 201)
        cls.res = solve(gf_problem(cls.g_tau, grid=cls.grid, error=toy.NOISE, n_tau=201))

    def test_backtransform_reproduces_the_input(self):
        # the assembled container must carry G(tau) itself, not -G(tau): the
        # residual check reads the raw diagnostics and cannot see this sign
        bt = backtransform(self.res)
        used = self.g_tau.data[:: (len(self.g_tau.mesh) - 1) // (len(bt.mesh) - 1)].real
        deviation = np.max(np.abs(used - bt.data.real))
        self.assertLess(deviation, 5.0 * toy.NOISE, msg="max|G - bt| = {:.3g}".format(deviation))
        self.assertLess(deviation, np.max(np.abs(used + bt.data.real)))

    def test_reduced_tau_grid_is_itself_a_mesh(self):
        # the stride must divide the interval evenly, or backtransform lands on
        # a mesh whose points are not the ones that were fitted
        from triqs.mesh import MeshImTime

        for n_tau in (101, 201, 251, 300, 400):
            with self.subTest(n_tau=n_tau):
                prob = gf_problem(self.g_tau, grid=self.grid, error=toy.NOISE, n_tau=n_tau)
                axis = prob.task("0").im_axis
                mesh = MeshImTime(beta=toy.BETA, statistic="Fermion", n_tau=len(axis))
                np.testing.assert_allclose(axis, [float(p.value) for p in mesh], atol=1e-12)
                self.assertAlmostEqual(axis[-1], toy.BETA, places=12)

    def test_model_asserts_the_intended_weight(self):
        # ana_cont rescales re_axis by beta for time kernels, so the model has
        # to be divided by beta to represent the same spectral weight
        prob = gf_problem(self.g_tau, grid=self.grid, error=toy.NOISE, n_tau=201)
        model = flat_model(self.grid, norm=1.0)
        rescaled_axis = self.grid.values * prob.layout.beta
        self.assertAlmostEqual(_trapz(model_for(model, prob), rescaled_axis), 1.0, places=6)

    def test_frequency_only_features_are_rejected(self):
        prob = gf_problem(self.g_tau, grid=self.grid, error=toy.NOISE, n_tau=201)
        with self.assertRaises(ValueError):
            solve(prob, preblur=0.5)
        with self.assertRaises(ValueError):
            check_moments(self.res, n_max=1)


class TestMatrixImTime(unittest.TestCase):
    def test_matrix_valued_tau_continuation(self):
        w = toy.real_axis()
        spectra = [toy.two_peaks(w), toy.one_peak(w)]
        g_tau = toy.gf_imtime(spectra, n_tau=1001)
        grid = linear_grid(-toy.WMAX, toy.WMAX, 201)
        res = solve(gf_problem(g_tau, grid=grid, error=toy.NOISE, n_tau=201))
        self.assertEqual(res.a_w.target_shape, (2, 2))
        for i in range(2):
            self.assertAlmostEqual(_trapz(res.a_w.data[:, i, i], grid.values), 1.0, places=2)
        self.assertTrue(validate(res).ok)


if __name__ == "__main__":
    unittest.main()
