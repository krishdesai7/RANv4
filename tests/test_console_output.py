from io import StringIO

from rich.console import Console
from rich.progress import Progress


def test_render_metrics_outputs_named_columns_and_values() -> None:
    from ran.evaluate import render_metrics

    output = StringIO()
    console = Console(file=output, color_system=None, width=120)
    metrics = {
        "detector_dim_0": {
            "wasserstein_before": 1.0,
            "wasserstein_after": 0.25,
            "wasserstein_improvement_pct": 75.0,
            "jensenshannon_before": 0.2,
            "jensenshannon_after": 0.1,
            "jensenshannon_improvement_pct": 50.0,
            "triangular_before": 8.0,
            "triangular_after": 2.0,
            "triangular_improvement_pct": 75.0,
        }
    }

    render_metrics("sample-run", metrics, ["dim_0"], console=console)

    rendered = output.getvalue()
    assert "sample-run" in rendered
    assert "Wasserstein" in rendered
    assert "Before" in rendered
    assert "After" in rendered
    assert "75.0%" in rendered


def test_download_file_updates_a_rich_progress_task(monkeypatch, tmp_path) -> None:
    from ran.data import download

    completed = []

    def fake_urlretrieve(url, dest, reporthook) -> None:
        reporthook(0, 25, 100)
        reporthook(2, 25, 100)
        reporthook(4, 25, 100)
        completed.append((url, dest))

    monkeypatch.setattr(download.urllib.request, "urlretrieve", fake_urlretrieve)
    progress = Progress(disable=True)
    task_id = progress.add_task("sample.npz", total=None)

    download._download_file(
        "https://example.test/sample.npz",
        tmp_path / "sample.npz",
        progress,
        task_id,
    )

    assert completed == [("https://example.test/sample.npz", tmp_path / "sample.npz")]
    assert progress.tasks[0].completed == 100
    assert progress.tasks[0].total == 100
