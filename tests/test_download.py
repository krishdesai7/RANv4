from __future__ import annotations

import numpy as np
import pytest
from ran.data.download import _get_var
from ran.rantypes import LOG_RHO_FLOOR


def _raw(n: int = 64, dtype=np.double) -> dict[str, np.ndarray]:
    """One generator's raw arrays, with degenerate jets in known rows.

    Rows 0-2 are single-constituent jets (no width, no tau2) and rows 3-5 are
    groomed down to nothing (no soft-drop mass). Every other row is ordinary.
    """
    rng = np.random.default_rng(0)
    widths = rng.uniform(0.01, 0.6, n)
    widths[:3] = 0.0
    sdms = rng.uniform(1.0, 30.0, n)
    sdms[3:6] = 0.0
    jets = np.zeros((n, 4))
    jets[:, 0] = rng.uniform(200.0, 400.0, n)
    jets[:, 3] = rng.uniform(0.0, 300.0, n)
    return {
        "gen_widths": widths.astype(dtype),
        "gen_tau2s": (widths * rng.uniform(0.0, 0.5, n)).astype(dtype),
        "gen_sdms": sdms.astype(dtype),
        "gen_jets": jets.astype(dtype),
        "gen_mults": rng.integers(1, 80, n).astype(dtype),
        "gen_zgs": rng.uniform(0.0, 0.5, n).astype(dtype),
    }


@pytest.mark.parametrize("var", ["m", "M", "w", "tau21", "zg", "sdm"])
def test_every_variable_is_finite_float64(var: str) -> None:
    got = _get_var(_raw(), var, "gen")
    assert got.dtype == np.double
    assert np.isfinite(got).all()


class TestDegenerateJets:
    """The two observables that are undefined for some jets."""

    def test_one_constituent_jets_take_the_tau21_convention(self) -> None:
        got = _get_var(_raw(), "tau21", "gen")
        np.testing.assert_array_equal(got[:3], 0.0)

    def test_tau21_is_the_plain_ratio_where_the_jet_has_width(self) -> None:
        raw = _raw()
        got = _get_var(raw, "tau21", "gen")
        expected = raw["gen_tau2s"][3:] / raw["gen_widths"][3:]
        np.testing.assert_array_equal(got[3:], expected)

    def test_massless_groomed_jets_take_the_log_floor(self) -> None:
        got = _get_var(_raw(), "sdm", "gen")
        np.testing.assert_array_equal(got[3:6], LOG_RHO_FLOOR)

    def test_sdm_is_the_plain_log_where_the_jet_has_groomed_mass(self) -> None:
        raw = _raw()
        got = _get_var(raw, "sdm", "gen")
        rho_sq = (raw["gen_sdms"][6:] / raw["gen_jets"][6:, 0]) ** 2
        np.testing.assert_allclose(got[6:], np.log(rho_sq), rtol=1e-15)

    def test_float32_inputs_behave_identically(self) -> None:
        """An epsilon in the denominator would not survive this.

        1e-50 is below the smallest float32 denormal, so it rounds to zero and
        leaves 0/0 -- the degenerate jets come back NaN instead of taking the
        convention. Guarding the operation rather than the operand does not
        care what the raw arrays were stored as.
        """
        narrow = _get_var(_raw(dtype=np.single), "tau21", "gen")
        assert np.isfinite(narrow).all()
        np.testing.assert_array_equal(narrow[:3], 0.0)


def test_unknown_variable_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown variable"):
        _get_var(_raw(), "nsubjettiness", "gen")
