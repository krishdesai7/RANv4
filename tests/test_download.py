from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ran.data import download
from ran.data.download import _constituents, _get_var
from ran.rantypes import LOG_RHO_FLOOR, SUBSTRUCTURE_VARIABLES

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_CONSTITUENTS = 12


def _raw(n: int = 64, dtype: type[np.floating] = np.double) -> dict[str, NDArray]:
    """One generator's raw arrays, with degenerate jets in known rows.

    Rows 0-2 are single-constituent jets (no width, no tau2) and rows 3-5 are
    groomed down to nothing (no soft-drop mass). Every other row is ordinary.
    """
    rng: np.random.Generator = np.random.default_rng(seed=0)
    widths: NDArray[np.double] = rng.uniform(low=0.01, high=0.6, size=n)
    widths[:3] = 0.0
    sdms: NDArray[np.double] = rng.uniform(low=1.0, high=30.0, size=n)
    sdms[3:6] = 0.0
    jets: NDArray[np.double] = np.zeros(shape=(n, 4))
    jets[:, 0] = rng.uniform(low=200.0, high=400.0, size=n)
    jets[:, 3] = rng.uniform(low=0.0, high=300.0, size=n)
    # (jets, constituents, 4) with columns (pt, y, phi, pid/10), zero-padded on
    # the constituent axis. Row 6 is padded to empty: every constituent-derived
    # observable divides by a pt sum, so a jet with no pt is the degenerate
    # case there, the way a zero width is for tau21.
    particles: NDArray[np.double] = np.zeros(shape=(n, MAX_CONSTITUENTS, 4))
    lengths: NDArray[np.long] = rng.integers(low=1, high=MAX_CONSTITUENTS + 1, size=n)
    for row, length in enumerate(iterable=lengths):
        particles[row, :length, 0] = rng.uniform(low=0.5, high=40.0, size=length)
        particles[row, :length, 3] = rng.integers(low=0, high=16, size=length) / 10.0
    particles[6] = 0.0
    return {
        "gen_widths": widths.astype(dtype),
        "gen_tau2s": (widths * rng.uniform(low=0.0, high=0.5, size=n)).astype(dtype),
        "gen_sdms": sdms.astype(dtype),
        "gen_jets": jets.astype(dtype),
        "gen_mults": rng.integers(low=1, high=80, size=n).astype(dtype),
        "gen_zgs": rng.uniform(low=0.0, high=0.5, size=n).astype(dtype),
        "gen_lhas": rng.uniform(low=0.0, high=0.8, size=n).astype(dtype),
        "gen_ang2s": rng.uniform(low=0.0, high=0.25, size=n).astype(dtype),
        "gen_particles": particles.astype(dtype),
    }


@pytest.mark.parametrize(argnames="var", argvalues=SUBSTRUCTURE_VARIABLES)
def test_every_variable_is_finite_float64(var: str) -> None:
    got: NDArray[np.double] = _get_var(_raw(), var, "gen")
    assert got.dtype == np.double
    assert np.isfinite(got).all()


class TestDegenerateJets:
    """The two observables that are undefined for some jets."""

    def test_one_constituent_jets_take_the_tau21_convention(self) -> None:
        got: NDArray[np.double] = _get_var(_raw(), "tau21", "gen")
        np.testing.assert_array_equal(actual=got[:3], desired=0.0)

    def test_tau21_is_the_plain_ratio_where_the_jet_has_width(self) -> None:
        raw: dict[str, NDArray] = _raw()
        got: NDArray[np.double] = _get_var(raw, "tau21", "gen")
        expected: NDArray[np.double] = raw["gen_tau2s"][3:] / raw["gen_widths"][3:]
        np.testing.assert_array_equal(actual=got[3:], desired=expected)

    def test_massless_groomed_jets_take_the_log_floor(self) -> None:
        got: NDArray[np.double] = _get_var(_raw(), "sdm", "gen")
        np.testing.assert_array_equal(actual=got[3:6], desired=LOG_RHO_FLOOR)

    def test_sdm_is_the_plain_log_where_the_jet_has_groomed_mass(self) -> None:
        raw: dict[str, NDArray] = _raw()
        got: NDArray[np.double] = _get_var(raw, "sdm", "gen")
        rho_sq: NDArray[np.double] = (raw["gen_sdms"][6:] / raw["gen_jets"][6:, 0]) ** 2
        np.testing.assert_allclose(actual=got[6:], desired=np.log(rho_sq), rtol=1e-15)

    def test_float32_inputs_behave_identically(self) -> None:
        """An epsilon in the denominator would not survive this.

        1e-50 is below the smallest float32 denormal, so it rounds to zero and
        leaves 0/0 -- the degenerate jets come back NaN instead of taking the
        convention. Guarding the operation rather than the operand does not
        care what the raw arrays were stored as.
        """
        narrow: NDArray[np.double] = _get_var(_raw(dtype=np.single), "tau21", "gen")
        assert np.isfinite(narrow).all()
        np.testing.assert_array_equal(narrow[:3], 0.0)


class TestConstituentObservables:
    """The four computed from `particles` rather than read from a stored array.

    `particles` is `(jets, constituents, 4)`. Indexing it as `[:, 0]` picks one
    constituent's features instead of every constituent's pt, which broadcasts
    fine whenever a jet has exactly four constituents -- so these check values,
    not just shapes.
    """

    @staticmethod
    def _one_jet(pt: list[float], pid: list[int]) -> dict[str, NDArray[np.double]]:
        particles: NDArray[np.double] = np.zeros((1, len(pt) + 2, 4))
        particles[0, : len(pt), 0] = pt
        particles[0, : len(pt), 3] = np.asarray(a=pid) / 10.0
        return {"gen_particles": particles}

    # pid index 1 carries charge +1, index 2 charge -1, index 0 charge 0.
    POSITIVE, NEGATIVE, NEUTRAL = 1, 2, 0

    def test_charge_is_the_root_pt_weighted_sum(self) -> None:
        raw: dict[str, NDArray[np.double]] = self._one_jet(
            [10, 20, 30, 40], [1, 2, 0, 1]
        )
        root: NDArray[np.double] = np.sqrt([10.0, 20.0, 30.0, 40.0])
        expected: NDArray[np.double] = (root[0] - root[1] + root[3]) / root.sum()
        np.testing.assert_allclose(actual=_get_var(raw, "q", "gen"), desired=[expected])

    def test_charged_fraction_is_pt_weighted(self) -> None:
        raw: dict[str, NDArray[np.double]] = self._one_jet(
            [10, 20, 30, 40], [1, 2, 0, 1]
        )
        np.testing.assert_allclose(
            actual=_get_var(raw, "f_ch", "gen"), desired=[(10 + 20 + 40) / 100]
        )

    def test_dispersion_is_one_for_a_single_constituent(self) -> None:
        """p_T^D runs from 1/sqrt(n) for n equal constituents up to 1."""
        np.testing.assert_allclose(
            actual=_get_var(self._one_jet(pt=[25.0], pid=[0]), "ptd", "gen"),
            desired=[1.0],
        )
        np.testing.assert_allclose(
            actual=_get_var(self._one_jet(pt=[5.0] * 4, pid=[0] * 4), "ptd", "gen"),
            desired=[0.5],
        )

    def test_padding_never_enters_the_charged_count(self) -> None:
        """`n_ch` is the one observable a zero-pt row could reach.

        Every other constituent observable weights by pt, so padding cancels
        itself. A count does not, and the padded rows carry pid index 0 only
        by convention -- so the mask is on pt.
        """
        raw: dict[str, NDArray[np.double]] = self._one_jet(
            pt=[10, 20, 30], pid=[1, 2, 0]
        )
        raw["gen_particles"][0, 3:, 3] = self.POSITIVE / 10.0  # charged, pt still 0
        np.testing.assert_array_equal(
            actual=_get_var(raw, "n_ch", "gen"), desired=[2.0]
        )

    def test_an_empty_jet_yields_zero_rather_than_nan(self) -> None:
        empty: dict[str, NDArray[np.double]] = {"gen_particles": np.zeros((1, 5, 4))}
        for var in ("q", "f_ch", "ptd", "n_ch"):
            got: NDArray[np.double] = _get_var(empty, var, "gen")
            assert np.isfinite(got).all(), var
            np.testing.assert_array_equal(actual=got, desired=[0.0])

    def test_the_charge_table_decodes_to_signed_values(self) -> None:
        """`PID_CHARGE` packs a 2-bit field per particle index; 0 means -1.

        Two separate things keep that subtraction signed: a `intp` index (so
        the shift promotes to int64) and the final `astype(int8)`. With a
        `uint8` index and no cast -- as this started out -- the whole
        expression stays `uint32` and every *negatively* charged constituent
        comes back as 2**32 - 1, which `f_ch` then weights by. Either guard
        alone is sufficient, so this pins the decoded values rather than
        either mechanism.
        """
        expected: list[int] = [0, 1, -1, 0, -1, 1, -1, 1, 1, -1, 1, -1, 0, 0, -1, -1]
        particles: NDArray[np.double] = np.zeros(shape=(1, 16, 4))
        particles[0, :, 0] = 1.0  # unit pt, so nothing is masked away
        particles[0, :, 3] = np.arange(16) / 10.0
        _, charge = _constituents({"gen_particles": particles}, "gen")
        np.testing.assert_array_equal(actual=charge[0], desired=expected)
        assert charge.dtype == np.int8

    @pytest.mark.parametrize(
        argnames=("bad", "match"),
        argvalues=[
            (np.zeros(shape=(2, 4)), "expected"),
            (np.zeros(shape=(2, 5, 3)), "expected"),
            (np.full(shape=(1, 2, 4), fill_value=1.6), "PID_CHARGE only encodes"),
        ],
    )
    def test_malformed_particles_are_refused(self, bad, match: str) -> None:
        with pytest.raises(expected_exception=ValueError, match=match):
            _constituents({"gen_particles": bad}, "gen")


def _shard(rows: int, gen_width: int, sim_width: int, seed: int) -> dict:
    """One raw shard, padded to a different constituent count per level."""
    rng: np.random.Generator = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for ptype, width in (("gen", gen_width), ("sim", sim_width)):
        particles: NDArray[np.double] = np.zeros(shape=(rows, width, 4))
        lengths: NDArray[np.long] = rng.integers(low=1, high=width + 1, size=rows)
        for row, length in enumerate(lengths):
            particles[row, :length, 0] = rng.uniform(low=0.5, high=40.0, size=length)
            particles[row, :length, 3] = (
                rng.integers(low=0, high=16, size=length) / 10.0
            )
        jets: NDArray[np.double] = np.zeros(shape=(rows, 4))
        jets[:, 0] = rng.uniform(low=200.0, high=400.0, size=rows)
        jets[:, 3] = rng.uniform(low=0.0, high=300.0, size=rows)
        widths: NDArray[np.double] = rng.uniform(low=0.01, high=0.6, size=rows)
        out |= {
            f"{ptype}_particles": particles,
            f"{ptype}_jets": jets,
            f"{ptype}_widths": widths,
            f"{ptype}_tau2s": widths * rng.uniform(low=0.0, high=0.5, size=rows),
            f"{ptype}_sdms": rng.uniform(low=1.0, high=30.0, size=rows),
            f"{ptype}_zgs": rng.uniform(low=0.0, high=0.5, size=rows),
            f"{ptype}_mults": rng.integers(low=1, high=80, size=rows).astype(
                dtype=np.double
            ),
            f"{ptype}_lhas": rng.uniform(low=0.0, high=0.8, size=rows),
            f"{ptype}_ang2s": rng.uniform(low=0.0, high=0.25, size=rows),
        }
    return out


def test_shards_padded_to_different_widths_still_concatenate(
    tmp_path, monkeypatch
) -> None:
    """The constituent axis is padded per array, not globally.

    File 0 of the real release carries 116 constituents for `gen` and 94 for
    `sim`, so the width varies between shards as well as between levels. An
    output buffer preallocated from the first shard cannot hold a wider one --
    it raises rather than truncating -- which is why the shards are reduced to
    observables before anything is concatenated.
    """
    rows, n_files = 20, 3
    widths: list[tuple[int, int]] = [(7, 4), (11, 9), (5, 13)]
    for generator in download.GENERATORS:
        for i, (gen_w, sim_w) in enumerate(iterable=widths):
            np.savez(
                tmp_path / f"{generator}_Zjet_pTZ-200GeV_{i}.npz",
                **_shard(rows, gen_width=gen_w, sim_width=sim_w, seed=i),
            )
    monkeypatch.setattr(download, "N_FILES", n_files)
    download.download_jet_data(cache_dir=tmp_path)

    for var in SUBSTRUCTURE_VARIABLES:
        with np.load(file=tmp_path / f"{download.CACHE_FILENAMES[var]}.npz") as cached:
            arrays: dict[str, NDArray[np.double]] = {
                key: cached[key] for key in cached.files
            }
        assert set(arrays) == {"z_true", "x_data", "z_gen", "x_sim"}, var
        for key, values in arrays.items():
            assert values.shape == (rows * n_files,), f"{var}/{key}"
            assert np.isfinite(values).all(), f"{var}/{key}"

    assert not list(tmp_path.glob("*Zjet*")), "raw shards were not cleaned up"


def test_unknown_variable_is_refused() -> None:
    with pytest.raises(expected_exception=ValueError, match="Unknown variable"):
        _get_var(_raw(), "nsubjettiness", "gen")
