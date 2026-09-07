"""Tests for `ran.report`'s template loading and value-formatting primitives."""

from __future__ import annotations

import pytest
from ran import report


def test_the_template_ships_with_the_package() -> None:
    """It is package data, not a repo-root file the wheel would drop."""
    assert "<<DETECTOR_TABLE>>" in report.load_template()


def test_every_token_the_generator_fills_is_in_the_template() -> None:
    tokens = set(report.TEMPLATE_TOKEN.findall(report.load_template()))
    assert tokens == {
        "<<RUN_NAME>>",
        "<<CONFIG_ROWS>>",
        "<<TIMINGS_ROWS>>",
        "<<DETECTOR_TABLE>>",
        "<<PARTICLE_TABLE>>",
        "<<FIGURE_DIR>>",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.2089450359, "0.2089450359"),
        (8.29431403648554e-05, "0.0000829431"),
        (0.0, "0"),
        (-504.2501717, "-504.2501717"),
    ],
)
def test_numbers_are_emitted_as_plain_decimals(value: float, expected: str) -> None:
    """siunitx S columns fix a table-format and cannot absorb an exponent."""
    rendered: str = report.decimal(value)
    assert "e" not in rendered.lower()
    assert rendered == expected
    assert float(rendered) == pytest.approx(value, rel=1e-6)


def test_underscores_in_a_value_are_escaped() -> None:
    r"""`f_ch` inside \texttt{} was a hard `Missing $ inserted` compile error."""
    assert report.latex_text("f_ch") == r"f\_ch"
    assert report.latex_text("100%") == r"100\%"


def test_the_token_guard_ignores_the_templates_own_documentation() -> None:
    """The header comment documents the contract and contains `<<...>>`."""
    assert report.TEMPLATE_TOKEN.findall("% Replace the six <<...>> tokens") == []


def test_config_rows_pair_two_entries_per_line() -> None:
    """The config table is four columns: key, value, key, value."""
    rows: str = report.config_rows({"dim": 12, "seed": 3, "n_layers": 2}, None)

    assert r"\ConfigPair{dim}{\Count{12}}{seed}{\Count{3}}" in rows
    # An odd trailing entry pads rather than shifting every later cell.
    assert r"\ConfigPair{n_layers}{\Count{2}}{}{}" in rows


def test_the_variable_list_spans_the_row() -> None:
    rows: str = report.config_rows({"variables": ["m", "f_ch"]}, None)
    assert r"\ConfigWide{variables}{\ConfigVal{m, f_ch}}" in rows


def test_the_mmd_sigmas_collapse_to_a_median_and_a_bracket() -> None:
    """Five bare floats tell a reader nothing; they are median x _SCALES."""
    median: float = 3.10746693611145
    sigmas: list[float] = [median * s for s in (0.5, 2**-0.5, 1.0, 2**0.5, 2.0)]

    rows: str = report.config_rows({"mmd_sigmas_detector": sigmas}, None)

    assert r"\ConfigWide{mmd_sigmas_detector}" in rows
    assert "3.107" in rows
    assert r"\sqrt2" in rows


def test_sigmas_that_are_not_the_bracket_are_printed_in_full() -> None:
    """A future change to _SCALES must not make the report quietly lie."""
    rows: str = report.config_rows({"mmd_sigmas_detector": [1.0, 2.0, 3.0]}, None)
    assert "1" in rows
    assert "2" in rows
    assert "3" in rows
    assert r"\sqrt2" not in rows


def test_compile_cache_warmth_reaches_the_config_table() -> None:
    """It is what makes the `compile` timing interpretable at all."""
    rows: str = report.config_rows({"dim": 1}, {"compile_cache_warm": True})
    assert "compile_cache_warm" in rows


def test_timing_rows_indent_nested_phases_and_blank_their_share() -> None:
    """A nested phase is already inside its parent's share."""
    payload = {
        "total_seconds": 10.0,
        "phases": [
            {
                "name": "train",
                "seconds": 8.0,
                "depth": 0,
                "detail": None,
                "failed": False,
                "pass": "train",
            },
            {
                "name": "compile",
                "seconds": 4.0,
                "depth": 1,
                "detail": None,
                "failed": False,
                "pass": "train",
            },
        ],
    }

    rows: str = report.timing_rows(payload)

    assert r"\TimingSubphase{compile}" in rows
    assert "compile} & 4 &  &" in rows.replace("\\TimingSubphase{", "")
    assert "80" in rows  # train's share
    assert r"\midrule" in rows
    assert "10" in rows  # the total row


def test_the_pass_is_folded_into_the_detail_cell() -> None:
    """The template's timing table has four columns and no pass column."""
    payload = {
        "total_seconds": 1.0,
        "phases": [
            {
                "name": "data",
                "seconds": 1.0,
                "depth": 0,
                "detail": "cache hit",
                "failed": False,
                "pass": "train",
            }
        ],
    }
    assert "cache hit -- train" in report.timing_rows(payload)
