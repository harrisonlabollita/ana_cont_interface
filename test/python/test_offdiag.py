# Copyright (c) 2026 Harrison LaBollita
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https://www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Harrison LaBollita

import unittest
import warnings

import numpy as np

import toy
from triqs_ana_cont_interface import gf_problem, linear_grid, poorman_model, solve, validate

_trapz = getattr(np, "trapezoid", np.trapz)
FRACTION = 0.3


def rotation(theta=0.4):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def complex_rotation(theta=0.6, phi=1.1):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s * np.exp(-1j * phi)], [s * np.exp(1j * phi), c]])


def rotated_gf(u, spectra, **kwargs):
    """A Gf that is diagonal in the basis `u`, expressed in the lab basis."""
    g_diag = toy.gf_imfreq(spectra, **kwargs)
    g_lab = g_diag.copy()
    g_lab.data[:] = np.einsum("ia,wab,jb->wij", u, g_diag.data, u.conj())
    return g_lab


def rotated_spectrum(u, spectra):
    diagonal = np.stack([np.diag(v) for v in np.stack(spectra, axis=1)]).astype(complex)
    return np.einsum("ia,wab,jb->wij", u, diagonal, u.conj())


class TestPoorman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = toy.real_axis()
        cls.a0, cls.a1 = toy.two_peaks(cls.w), toy.one_peak(cls.w)
        cls.true_offdiag = FRACTION * np.sqrt(cls.a0 * cls.a1)
        g = toy.gf_imfreq([cls.a0, cls.a1], offdiag=FRACTION, noise=toy.NOISE)
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, toy.NW)
        cls.res = solve(gf_problem(g, grid=cls.grid, error=toy.NOISE, n_iw=30, mode="poorman"))

    def test_one_task_per_upper_triangle_element(self):
        keys = sorted(t.key for t in self.res.problem.tasks)
        self.assertEqual(keys, [("0", 0, 0), ("0", 0, 1), ("0", 1, 1)])
        self.assertTrue(self.res.problem.task("0", 0, 1).offdiag)
        self.assertFalse(self.res.problem.task("0", 0, 0).offdiag)

    def test_lower_triangle_is_filled_by_symmetry(self):
        # the continued spectrum is real, so A_ji = A_ij and G_ji = G_ij
        np.testing.assert_array_equal(self.res.g_w.data[:, 1, 0], self.res.g_w.data[:, 0, 1])

    def test_offdiagonal_spectrum_is_recovered(self):
        a01 = self.res.a_w.data[:, 0, 1]
        deviation = _trapz(np.abs(a01 - self.true_offdiag), self.w)
        weight = _trapz(np.abs(self.true_offdiag), self.w)
        self.assertLess(deviation, 0.2 * weight)
        self.assertAlmostEqual(_trapz(a01, self.w), weight, places=2)

    def test_spectral_matrix_stays_positive_semidefinite(self):
        self.assertGreater(np.min(np.linalg.det(self.res.a_w.data)), -1e-8)

    def test_validates(self):
        report = validate(self.res)
        self.assertTrue(report.ok, msg="\n" + str(report))


class TestEigenbasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = toy.real_axis()
        cls.u = rotation()
        a0, a1 = toy.two_peaks(cls.w), toy.one_peak(cls.w)
        diagonal = np.stack([np.diag(v) for v in np.stack([a0, a1], axis=1)])
        cls.true = np.einsum("ia,wab,jb->wij", cls.u, diagonal, cls.u.conj()).real
        g_diag = toy.gf_imfreq([a0, a1], noise=toy.NOISE)
        g_lab = g_diag.copy()
        g_lab.data[:] = np.einsum("ia,wab,jb->wij", cls.u, g_diag.data, cls.u.conj())
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, toy.NW)
        cls.res = solve(
            gf_problem(g_lab, grid=cls.grid, error=toy.NOISE, n_iw=30,
                       mode="eigenbasis", rotation=cls.u)
        )

    def test_all_lab_basis_elements_are_recovered(self):
        for i in range(2):
            for j in range(2):
                deviation = _trapz(np.abs(self.res.a_w.data[:, i, j] - self.true[:, i, j]), self.w)
                self.assertLess(deviation, 0.15, msg="element ({}, {})".format(i, j))

    def test_rotation_is_required(self):
        g = toy.gf_imfreq([toy.two_peaks(self.w), toy.one_peak(self.w)], noise=toy.NOISE)
        with self.assertRaises(ValueError):
            gf_problem(g, grid=self.grid, error=toy.NOISE, mode="eigenbasis")


class TestModeCoverage(unittest.TestCase):
    def setUp(self):
        self.w = toy.real_axis()
        self.grid = linear_grid(-toy.WMAX, toy.WMAX, 201)

    def test_poorman_over_a_block_gf(self):
        gb = toy.block_gf_imfreq(
            spectra=[toy.two_peaks(self.w), toy.one_peak(self.w)], offdiag=FRACTION
        )
        res = solve(
            gf_problem(gb, grid=self.grid, error=toy.NOISE, n_iw=20, mode="poorman")
        )
        self.assertEqual(len(res.problem.tasks), 6)  # 3 per block
        for name in ("up", "dn"):
            np.testing.assert_array_equal(
                res.g_w[name].data[:, 1, 0], res.g_w[name].data[:, 0, 1]
            )
        self.assertTrue(validate(res).ok)

    def test_eigenbasis_on_a_scalar_gf_is_a_no_op(self):
        g = toy.gf_imfreq(toy.two_peaks(self.w))
        rotated = solve(
            gf_problem(g, grid=self.grid, error=toy.NOISE, n_iw=20,
                       mode="eigenbasis", rotation=np.ones((1, 1)))
        )
        plain = solve(gf_problem(g, grid=self.grid, error=toy.NOISE, n_iw=20))
        np.testing.assert_allclose(rotated.a_w.data, plain.a_w.data)

    def test_eigenbasis_in_imaginary_time(self):
        a0, a1 = toy.two_peaks(self.w), toy.one_peak(self.w)
        grid = linear_grid(-toy.WMAX, toy.WMAX, toy.NW)  # must match toy.real_axis()
        u = rotation()
        g_diag = toy.gf_imtime([a0, a1], n_tau=1001)
        g_lab = g_diag.copy()
        g_lab.data[:] = np.einsum("ia,wab,jb->wij", u, g_diag.data, u.conj())
        res = solve(
            gf_problem(g_lab, grid=grid, error=toy.NOISE, n_tau=201,
                       mode="eigenbasis", rotation=u)
        )
        diagonal = np.stack([np.diag(v) for v in np.stack([a0, a1], axis=1)])
        true = np.einsum("ia,wab,jb->wij", u, diagonal, u.conj()).real
        for i in range(2):
            for j in range(2):
                deviation = _trapz(np.abs(res.a_w.data[:, i, j] - true[:, i, j]), self.w)
                self.assertLess(deviation, 0.15, msg="element ({}, {})".format(i, j))


class TestComplexTarget(unittest.TestCase):
    """A complex-hermitian target: the case where symmetry and hermiticity differ."""

    @classmethod
    def setUpClass(cls):
        cls.w = toy.real_axis()
        cls.spectra = [toy.two_peaks(cls.w), toy.one_peak(cls.w)]
        cls.u = complex_rotation()
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, toy.NW)
        cls.g = rotated_gf(cls.u, cls.spectra, noise=toy.NOISE)

    def test_poorman_rejects_an_asymmetric_target(self):
        # poorman continues one real spectrum per element, which needs
        # g_ij == g_ji; hermiticity alone is not enough
        from triqs.gfs import is_gf_hermitian

        self.assertTrue(is_gf_hermitian(self.g))
        self.assertFalse(np.allclose(self.g.data[:, 0, 1], self.g.data[:, 1, 0]))
        with self.assertRaises(ValueError) as caught:
            gf_problem(self.g, grid=self.grid, error=toy.NOISE, n_iw=20, mode="poorman")
        self.assertIn("symmetric", str(caught.exception))

    def test_spectral_matrix_is_the_hermitian_part(self):
        # A = (i/2pi)(G - G^dag). Elementwise -Im G/pi mixes in Re G once the
        # rotation is complex, and is not even hermitian.
        res = solve(
            gf_problem(self.g, grid=self.grid, error=toy.NOISE, n_iw=30,
                       mode="eigenbasis", rotation=self.u)
        )
        true = rotated_spectrum(self.u, self.spectra)
        a = res.a_w.data
        np.testing.assert_allclose(a, np.conj(np.swapaxes(a, 1, 2)), atol=1e-12)
        deviation = _trapz(np.abs(a[:, 0, 1] - true[:, 0, 1]), self.w)
        self.assertLess(deviation, 0.15, msg="int|A01 - true| = {:.4f}".format(deviation))
        naive = -res.g_w.data.imag / np.pi
        self.assertGreater(_trapz(np.abs(naive[:, 0, 1] - true[:, 0, 1]), self.w), 0.5)

    def test_eigenbasis_warns_when_the_rotation_does_not_diagonalize(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gf_problem(self.g, grid=self.grid, error=toy.NOISE, n_iw=20,
                       mode="eigenbasis", rotation=np.eye(2))
        messages = [str(c.message) for c in caught]
        self.assertTrue(
            any("does not diagonalize" in m for m in messages), msg=str(messages)
        )


class TestPoormanModel(unittest.TestCase):
    def test_it_uses_both_diagonals(self):
        # sqrt(A_ii A_jj) must vanish where the two diagonals do not overlap;
        # using A_ii alone would put weight there
        w = toy.real_axis()
        left = toy.one_peak(w, center=-4.0, width=0.5)
        right = toy.one_peak(w, center=4.0, width=0.5)
        both = poorman_model(left, right)
        one = poorman_model(left, left)
        self.assertLess(both.max(), 0.05 * one.max())


if __name__ == "__main__":
    unittest.main()
