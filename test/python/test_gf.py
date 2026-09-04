import unittest
import warnings

import numpy as np
from triqs.gfs import BlockGf
from triqs.mesh import MeshReFreq, MeshReFreqPts

import toy
from triqs_ana_cont_interface import (
    check_moments,
    gf_problem,
    linear_grid,
    solve,
    tangent_grid,
    validate,
)

_trapz = getattr(np, "trapezoid", np.trapz)


class TestScalarGf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = toy.real_axis()
        cls.spec = toy.two_peaks(cls.w)
        cls.g = toy.gf_imfreq(cls.spec)
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, toy.NW)
        cls.res = solve(gf_problem(cls.g, grid=cls.grid, error=toy.NOISE, n_iw=30), preblur=0.4)

    def test_containers(self):
        self.assertIsInstance(self.res.g_w.mesh, MeshReFreq)
        self.assertEqual(self.res.g_w.data.shape, (toy.NW,))
        self.assertEqual(self.res.g_w.data.dtype, np.complex128)
        # a_w is the hermitian spectral matrix, so it is complex in general;
        # for a scalar target its imaginary part is identically zero
        self.assertEqual(self.res.a_w.data.dtype, np.complex128)
        np.testing.assert_allclose(self.res.a_w.data.imag, 0.0, atol=1e-15)

    def test_spectrum_is_positive_and_normalized(self):
        a = self.res.a_w.data.real
        self.assertGreater(a.min(), -1e-8)
        self.assertAlmostEqual(_trapz(a, self.w), 1.0, places=2)

    def test_peaks_recovered(self):
        a = self.res.a_w.data.real
        for centre in (-1.8, 1.8):
            near = np.abs(self.w - centre) < 1.0
            self.assertGreater(a[near].max(), 0.5 * a.max())

    def test_validates(self):
        report = validate(self.res)
        self.assertTrue(report.ok, msg="\n" + str(report))

    def test_moment_sum_rule_is_the_default_reference(self):
        moments = check_moments(self.res)
        entry = moments[("0", 0, 0)]
        self.assertEqual(entry["source"][0], "sum-rule")
        self.assertEqual(entry["expected"][0], 1.0)
        self.assertLess(entry["rel_error"][0], 0.02)

    def test_tangent_grid_gives_a_points_mesh(self):
        res = solve(gf_problem(self.g, grid=tangent_grid(20.0, 201), error=toy.NOISE, n_iw=30))
        self.assertIsInstance(res.g_w.mesh, MeshReFreqPts)
        self.assertEqual(len(res.g_w.mesh), 201)


class TestBlockAndMatrix(unittest.TestCase):
    def test_block_structure_is_preserved(self):
        gb = toy.block_gf_imfreq(spectra=[toy.two_peaks(toy.real_axis()), toy.one_peak(toy.real_axis())])
        res = solve(gf_problem(gb, grid=linear_grid(-8, 8, 201), error=toy.NOISE, n_iw=20))
        self.assertEqual(res.g_w.n_blocks, 2)
        self.assertEqual([str(n) for n, _ in res.g_w], ["up", "dn"])
        for _, g in res.g_w:
            self.assertEqual(g.target_shape, (2, 2))
        self.assertEqual(len(res.diagnostics), 4)

    def test_diagonal_mode_warns_about_discarded_offdiagonal(self):
        w = toy.real_axis()
        g = toy.gf_imfreq([toy.two_peaks(w), toy.one_peak(w)], offdiag=0.3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gf_problem(g, grid=linear_grid(-8, 8, 201), error=toy.NOISE, n_iw=20)
        messages = [str(c.message) for c in caught]
        self.assertTrue(any("discards off-diagonal weight" in m for m in messages), messages)


class TestDegenerateBlocks(unittest.TestCase):
    """degenerate_blocks continues one block of each group and copies the rest."""

    @classmethod
    def setUpClass(cls):
        w = toy.real_axis()
        g = toy.gf_imfreq(toy.two_peaks(w))
        cls.gb = BlockGf(name_list=["up", "dn"], block_list=[g, g.copy()])
        cls.grid = linear_grid(-toy.WMAX, toy.WMAX, 201)
        cls.res = solve(gf_problem(cls.gb, grid=cls.grid, error=toy.NOISE, n_iw=20,
                                   degenerate_blocks=[[0, 1]]))

    def test_only_the_representative_is_continued(self):
        calls = []
        import ana_cont.continuation as cont

        original = cont.AnalyticContinuationProblem.solve
        try:
            cont.AnalyticContinuationProblem.solve = lambda self, *a, **k: (
                calls.append(1) or original(self, *a, **k)
            )
            solve(gf_problem(self.gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                             degenerate_blocks=[[0, 1]]))
        finally:
            cont.AnalyticContinuationProblem.solve = original
        self.assertEqual(len(calls), 1)

    def test_the_copy_is_the_representative(self):
        np.testing.assert_array_equal(self.res.a_w["dn"].data, self.res.a_w["up"].data)
        np.testing.assert_array_equal(self.res.g_w["dn"].data, self.res.g_w["up"].data)

    def test_the_copy_keeps_its_own_diagnostics_and_validates(self):
        # without a record of its own the copied block would be silently zero in
        # backtransform() and absent from the report
        self.assertEqual(self.res.diag("dn", 0, 0).copied_from, "up")
        self.assertIsNone(self.res.diag("up", 0, 0).copied_from)
        report = validate(self.res)
        self.assertIn(("dn", 0, 0), report.residuals)
        self.assertTrue(report.ok, msg=str(report))
        self.assertGreater(np.abs(report.backtransform["dn"].data).max(), 0.1)

    def test_matrix_offdiagonals_are_copied_too(self):
        w = toy.real_axis()
        g = toy.gf_imfreq([toy.two_peaks(w), toy.one_peak(w)], offdiag=0.3)
        gb = BlockGf(name_list=["up", "dn"], block_list=[g, g.copy()])
        res = solve(gf_problem(gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                               mode="poorman", degenerate_blocks=[[0, 1]]))
        np.testing.assert_array_equal(res.a_w["dn"].data, res.a_w["up"].data)
        self.assertEqual(len(res.diagnostics), 6)
        self.assertEqual(res.diag("dn", 0, 1).copied_from, "up")

    def test_indices_and_names_are_the_same_group(self):
        prob = gf_problem(self.gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                          degenerate_blocks=[["up", "dn"]])
        self.assertEqual(prob.recipe.degenerate_blocks, (("up", "dn"),))
        self.assertEqual(prob.recipe.representative, {"dn": "up"})

    def test_an_out_of_range_index_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            gf_problem(self.gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                       degenerate_blocks=[[0, 2]])
        self.assertIn("2 blocks", str(caught.exception))

    def test_an_unknown_block_name_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            gf_problem(self.gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                       degenerate_blocks=[["up", "nope"]])
        self.assertIn("not in the input", str(caught.exception))

    def test_a_block_in_two_groups_is_rejected(self):
        with self.assertRaises(ValueError):
            gf_problem(self.gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                       degenerate_blocks=[[0, 1], [1]])


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.g = toy.gf_imfreq(toy.two_peaks(toy.real_axis()))
        self.grid = linear_grid(-8, 8, 201)

    def test_negative_error_rejected(self):
        with self.assertRaises(ValueError):
            gf_problem(self.g, grid=self.grid, error=-1e-4)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            gf_problem(self.g, grid=self.grid, error=1e-4, mode="magic")

    def test_bosonic_mesh_rejected(self):
        from triqs.gfs import Gf
        from triqs.mesh import MeshImFreq

        g = Gf(mesh=MeshImFreq(beta=10.0, statistic="Boson", n_iw=10), target_shape=[])
        with self.assertRaises(NotImplementedError):
            gf_problem(g, grid=self.grid, error=1e-4)

    def test_dlr_mesh_rejected_with_a_pointer_to_the_fix(self):
        from triqs.gfs import Gf
        from triqs.mesh import MeshDLRImFreq

        g = Gf(mesh=MeshDLRImFreq(beta=20.0, statistic="Fermion", w_max=5.0, eps=1e-8),
               target_shape=[])
        with self.assertRaises(TypeError) as caught:
            gf_problem(g, grid=self.grid, error=1e-4)
        self.assertIn("convert", str(caught.exception))

    def test_blocks_must_share_one_mesh(self):
        # one frequency selection is applied to every block, so a block with a
        # different mesh would be sliced at the wrong points
        w = toy.real_axis()
        gb = BlockGf(
            name_list=["up", "dn"],
            block_list=[toy.gf_imfreq(toy.two_peaks(w), n_iw=40),
                        toy.gf_imfreq(toy.two_peaks(w), n_iw=100)],
        )
        with self.assertRaises(ValueError) as caught:
            gf_problem(gb, grid=self.grid, error=1e-4, n_iw=20)
        self.assertIn("share", str(caught.exception))

    def test_task_lookup_rejects_an_unknown_key(self):
        prob = gf_problem(self.g, grid=self.grid, error=1e-4, n_iw=20)
        self.assertEqual(prob.task("0").key, ("0", 0, 0))
        with self.assertRaises(KeyError):
            prob.task("nope")

    def test_error_accepts_a_gf(self):
        err = self.g.copy()
        err.data[:] = 1e-4
        prob = gf_problem(self.g, grid=self.grid, error=err, n_iw=20)
        np.testing.assert_allclose(prob.task("0").error, 1e-4)


if __name__ == "__main__":
    unittest.main()
