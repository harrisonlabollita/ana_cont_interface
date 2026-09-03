import unittest
import warnings

import numpy as np

import toy
from triqs_ana_cont_interface import gf_problem, linear_grid, solve


PACKAGE = "triqs_ana_cont_interface"


def messages(fn, ours_only=False):
    """Warnings raised by fn; ours_only drops upstream noise such as ana_cont's
    np.trapz deprecations, which are attributed to their own source file."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(c.message) for c in caught if not ours_only or PACKAGE in c.filename]


class TestWarnings(unittest.TestCase):
    def setUp(self):
        self.w = toy.real_axis()
        self.grid = linear_grid(-toy.WMAX, toy.WMAX, 201)

    def assertWarns_containing(self, fragment, fn):
        found = messages(fn)
        self.assertTrue(
            any(fragment in m for m in found),
            msg="no warning containing {!r}; got {}".format(fragment, found),
        )

    def test_blocks_that_are_not_degenerate(self):
        from triqs.gfs import BlockGf

        gb = BlockGf(
            name_list=["up", "dn"],
            block_list=[toy.gf_imfreq(toy.two_peaks(self.w)),
                        toy.gf_imfreq(toy.one_peak(self.w))],
        )
        self.assertWarns_containing(
            "declared degenerate but differ",
            lambda: gf_problem(gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                               degenerate_blocks=[[0, 1]]),
        )

    def test_independent_noise_on_degenerate_blocks_is_not_flagged(self):
        # two blocks with the same spectrum but independent noise differ by
        # sqrt(2) error bars; that must not read as a broken degeneracy
        gb = toy.block_gf_imfreq(spectra=toy.two_peaks(self.w))
        found = messages(
            lambda: gf_problem(gb, grid=self.grid, error=toy.NOISE, n_iw=20,
                               degenerate_blocks=[[0, 1]]),
            ours_only=True,
        )
        self.assertEqual([m for m in found if "degenerate" in m], [])

    def test_non_hermitian_input(self):
        g = toy.gf_imfreq(toy.two_peaks(self.w), noise=0.0)
        g.data[toy.N_IW] += 1e-2  # break G(-iw) = conj(G(iw))
        self.assertWarns_containing(
            "not hermitian",
            lambda: gf_problem(g, grid=self.grid, error=toy.NOISE, n_iw=20),
        )

    def test_error_bars_below_machine_noise(self):
        g = toy.gf_imfreq(toy.two_peaks(self.w))
        self.assertWarns_containing(
            "below 1e-8",
            lambda: gf_problem(g, grid=self.grid, error=1e-12, n_iw=20),
        )

    def test_all_matsubara_frequencies_used(self):
        g = toy.gf_imfreq(toy.two_peaks(self.w), n_iw=350)
        self.assertWarns_containing(
            "positive Matsubara frequencies",
            lambda: gf_problem(g, grid=self.grid, error=toy.NOISE),
        )

    def test_all_imaginary_time_points_used(self):
        g = toy.gf_imtime(toy.two_peaks(self.w), n_tau=2001)
        self.assertWarns_containing(
            "imaginary-time points",
            lambda: gf_problem(g, grid=self.grid, error=toy.NOISE),
        )

    def test_noiseless_input(self):
        g = toy.gf_imfreq(toy.two_peaks(self.w), noise=0.0)
        prob = gf_problem(g, grid=self.grid, error=toy.NOISE, n_iw=20)
        self.assertWarns_containing("below the error bars", lambda: solve(prob))

    def test_no_warnings_for_well_posed_input(self):
        g = toy.gf_imfreq(toy.two_peaks(self.w))
        ours = messages(
            lambda: solve(gf_problem(g, grid=self.grid, error=toy.NOISE, n_iw=20)),
            ours_only=True,
        )
        self.assertEqual(ours, [], msg="unexpected warnings: {}".format(ours))


if __name__ == "__main__":
    unittest.main()
