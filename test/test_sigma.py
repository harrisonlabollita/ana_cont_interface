import unittest

import numpy as np

import toy
from triqs_ana_cont_interface import check_moments, linear_grid, sigma_problem, solve, validate

_trapz = getattr(np, "trapezoid", np.trapz)

SIGMA_INF = 2.5
FIRST_MOMENT = 1.7


def sigma_gf(noise=toy.NOISE, n_iw=toy.N_IW, shift=SIGMA_INF, moment=FIRST_MOMENT, **kwargs):
    w = toy.real_axis()
    return toy.gf_imfreq(moment * toy.one_peak(w, center=0.0, width=1.5),
                         n_iw=n_iw, noise=noise, shift=shift, **kwargs)


class TestSigmaMoments(unittest.TestCase):
    def test_constant_is_discovered_exactly(self):
        prob = sigma_problem(sigma_gf(noise=0.0), grid=linear_grid(-20, 20, 201), error=1e-4)
        self.assertAlmostEqual(prob.task("0").shift.real, SIGMA_INF, places=4)

    def test_first_moment_bias_shrinks_with_the_frequency_range(self):
        # -Im Sigma * w_n -> first moment with an O(1/w_n^2) correction
        errors = []
        for n_iw in (40, 200):
            prob = sigma_problem(
                sigma_gf(noise=0.0, n_iw=n_iw), grid=linear_grid(-20, 20, 201), error=1e-4
            )
            errors.append(abs(prob.task("0").model_norm - FIRST_MOMENT))
        self.assertLess(errors[0], 0.02 * FIRST_MOMENT)
        self.assertLess(errors[1], errors[0])

    def test_discovered_moments_survive_noise(self):
        # the high-frequency estimate must hold where a tail fit does not:
        # fit_hermitian_tail returns a negative first moment at 1e-3 noise
        for noise in (1e-4, 1e-3):
            with self.subTest(noise=noise):
                prob = sigma_problem(
                    sigma_gf(noise=noise), grid=linear_grid(-20, 20, 201), error=noise
                )
                task = prob.task("0")
                self.assertAlmostEqual(task.shift.real, SIGMA_INF, places=2)
                self.assertGreater(task.model_norm, 0.0)
                self.assertLess(abs(task.model_norm - FIRST_MOMENT), 0.05 * FIRST_MOMENT)

    def test_default_path_produces_a_valid_continuation(self):
        for noise in (1e-4, 1e-3):
            with self.subTest(noise=noise):
                res = solve(
                    sigma_problem(sigma_gf(noise=noise), grid=linear_grid(-20, 20, 401),
                                  error=noise, n_iw=30)
                )
                report = validate(res)
                self.assertTrue(report.ok, msg="\n" + str(report))

    def test_first_moment_sets_the_model_norm_and_is_recovered(self):
        res = solve(
            sigma_problem(sigma_gf(), grid=linear_grid(-20, 20, 401), error=toy.NOISE,
                          sigma_inf=SIGMA_INF, model_norm=FIRST_MOMENT)
        )
        entry = check_moments(res)[("0", 0, 0)]
        self.assertEqual(entry["source"][0], "first-moment")
        self.assertAlmostEqual(entry["expected"][0], FIRST_MOMENT, places=6)
        self.assertLess(entry["rel_error"][0], 0.05)


class TestSigmaAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = linear_grid(-20, 20, 401)
        cls.res = solve(
            sigma_problem(sigma_gf(), grid=cls.grid, error=toy.NOISE,
                          sigma_inf=SIGMA_INF, model_norm=FIRST_MOMENT)
        )

    def test_constant_is_added_back_exactly(self):
        difference = self.res.sigma_w.data - self.res.aux.g_w.data
        np.testing.assert_allclose(difference, SIGMA_INF, atol=1e-12)
        self.assertIsNotNone(self.res.diag("0").alpha_opt)

    def test_real_part_approaches_sigma_inf_at_the_edges(self):
        edges = 0.5 * (self.res.sigma_w.data[0].real + self.res.sigma_w.data[-1].real)
        self.assertAlmostEqual(edges, SIGMA_INF, places=1)

    def test_validates(self):
        report = validate(self.res)
        self.assertTrue(report.ok, msg="\n" + str(report))


class TestSigmaMatrix(unittest.TestCase):
    def test_offdiagonal_constant_is_subtracted_too(self):
        w = toy.real_axis()
        shift = np.array([[2.5, 0.4], [0.4, 1.5]])
        g = toy.gf_imfreq(
            [1.7 * toy.one_peak(w, 0.0, 1.5), 1.2 * toy.one_peak(w, 0.5, 1.0)],
            noise=0.0, shift=shift, offdiag=0.3,
        )
        prob = sigma_problem(g, grid=linear_grid(-20, 20, 201), error=1e-4, mode="poorman")
        np.testing.assert_allclose(prob.task("0", 0, 1).shift.real, 0.4, atol=1e-3)
        np.testing.assert_allclose(prob.task("0", 0, 0).shift.real, 2.5, atol=1e-3)


class TestSigmaInterface(unittest.TestCase):
    def test_imaginary_time_input_is_rejected(self):
        g_tau = toy.gf_imtime(toy.two_peaks(toy.real_axis()))
        with self.assertRaises(TypeError):
            sigma_problem(g_tau, grid=linear_grid(-20, 20, 201), error=1e-4)

class TestSigmaMatrixMoments(unittest.TestCase):
    """Sigma_inf is a hermitian matrix, and the moments must match sigma_w."""

    def setUp(self):
        self.w = toy.real_axis()
        self.grid = linear_grid(-20, 20, 201)
        self.spectra = [1.7 * toy.one_peak(self.w, 0.0, 1.5),
                        1.2 * toy.one_peak(self.w, 0.5, 1.0)]

    def test_complex_offdiagonal_constant_is_preserved(self):
        # Sigma_inf is hermitian, so its off-diagonal is complex; taking the
        # real part elementwise would leave i Im(Sigma_inf) in the data, which
        # no real spectral function can reproduce
        constant = np.array([[2.5, 0.4 + 0.7j], [0.4 - 0.7j, 1.5]])
        g = toy.gf_imfreq(self.spectra, noise=toy.NOISE, shift=constant, offdiag=0.3)
        prob = sigma_problem(g, grid=self.grid, error=1e-4, n_iw=40, mode="poorman")
        # the imaginary part is exact; the real part carries the O(1/w^2) bias
        self.assertAlmostEqual(prob.task("0", 0, 1).shift.imag, 0.7, places=2)
        self.assertAlmostEqual(prob.task("0", 0, 1).shift.real, 0.4, places=2)
        self.assertAlmostEqual(prob.task("0", 0, 0).shift.real, 2.5, places=3)
        self.assertAlmostEqual(prob.task("0", 0, 0).shift.imag, 0.0, places=6)
        res = solve(prob)
        # the hermitian partner is the conjugate
        np.testing.assert_allclose(
            res.sigma_inf["0"], np.conj(res.sigma_inf["0"]).T, atol=1e-12
        )
        np.testing.assert_allclose(res.sigma_inf["0"][0, 1], 0.4 + 0.7j, atol=1e-2)

    def test_moments_are_reported_in_the_basis_of_sigma_w(self):
        # in eigenbasis mode the moments are estimated in the rotated basis;
        # reporting them unrotated makes check_moments compare across bases
        theta = 0.4
        u = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        constant = u @ np.diag([2.5, 1.0]) @ u.conj().T
        g_diag = toy.gf_imfreq(self.spectra, noise=toy.NOISE, shift=np.diag([2.5, 1.0]))
        g = g_diag.copy()
        g.data[:] = np.einsum("ia,wab,jb->wij", u, g_diag.data, u.conj())
        moments = np.diag([1.7, 1.2])
        res = solve(
            sigma_problem(g, grid=self.grid, error=toy.NOISE, n_iw=40,
                          mode="eigenbasis", rotation=u,
                          sigma_inf=constant, model_norm=u @ moments @ u.conj().T)
        )
        # reported in the input's basis, matching sigma_w, not the rotated one
        np.testing.assert_allclose(res.sigma_inf["0"], constant, atol=1e-10)
        np.testing.assert_allclose(res.first_moment["0"], u @ moments @ u.conj().T, atol=1e-10)
        edge = 0.5 * (res.sigma_w.data[0].real + res.sigma_w.data[-1].real)
        np.testing.assert_allclose(edge, constant, atol=0.2)

    def test_non_positive_first_moment_warns(self):
        import warnings

        g = toy.gf_imfreq(self.spectra, noise=0.0, shift=2.5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sigma_problem(g, grid=self.grid, error=1e-4, n_iw=20, model_norm=-1.0)
        self.assertTrue(
            any("non-positive first moment" in str(c.message) for c in caught),
            msg=str([str(c.message) for c in caught]),
        )


if __name__ == "__main__":
    unittest.main()
