import logging
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- backend isolation
import sys
from types import SimpleNamespace


def _completion_records(caplog, logger_name):
    return [
        record
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.INFO
    ]


def test_raw_download_records_completion(monkeypatch, tmp_path, caplog):
    from ran.data import download
    from rich.progress import Progress

    destination = tmp_path / "sample.npz"

    def fake_urlretrieve(_url, _dest, reporthook):
        reporthook(1, 100, 100)

    monkeypatch.setattr(download.urllib.request, "urlretrieve", fake_urlretrieve)
    progress = Progress(disable=True)
    task_id = progress.add_task(destination.name, total=None)

    with caplog.at_level(logging.INFO, logger="ran.data.download"):
        download._download_file(
            "https://example.test/sample.npz", destination, progress, task_id
        )

    messages = [
        record.getMessage()
        for record in _completion_records(caplog, "ran.data.download")
    ]
    assert any(
        "Downloaded" in message and str(destination) in message for message in messages
    )


def test_evaluation_records_metrics_artifact_completion(monkeypatch, tmp_path, caplog):
    import numpy as np
    from ran import evaluate

    run_dir = tmp_path / "sample-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"dataset": "gaussian", "dim": 1}')
    (run_dir / "generator.keras").touch()
    z = np.array([[0.0], [1.0], [2.0], [3.0]])
    x = z.copy()
    y = np.array([1, 1, 0, 0])

    monkeypatch.setattr(evaluate.keras.saving, "load_model", lambda _path: object())
    monkeypatch.setattr(
        evaluate, "_load_splits", lambda _config: SimpleNamespace(test=object())
    )
    monkeypatch.setattr(evaluate, "_collect_test_data", lambda _test: (z, x, y))
    monkeypatch.setattr(evaluate, "_get_weights", lambda _model, _values: np.ones(2))
    monkeypatch.setattr(evaluate, "_wd_per_dim", lambda *_args, **_kwargs: [1.0])
    monkeypatch.setattr(evaluate, "_js_per_dim", lambda *_args, **_kwargs: [0.5])
    monkeypatch.setattr(
        evaluate, "_triangular_per_dim", lambda *_args, **_kwargs: [0.25]
    )
    monkeypatch.setattr(evaluate, "render_metrics", lambda *_args, **_kwargs: None)

    with caplog.at_level(logging.INFO, logger="ran.evaluate"):
        evaluate.evaluate_run(run_dir)

    out_path = run_dir / "metrics.json"
    messages = [
        record.getMessage() for record in _completion_records(caplog, "ran.evaluate")
    ]
    assert out_path.exists()
    assert any(
        "metrics" in message.lower() and str(out_path) in message
        for message in messages
    )


def test_omnifold_records_metric_and_weight_artifact_completion(tmp_path):
    run_dir = tmp_path / "sample-run"
    run_dir.mkdir()
    script = """
import logging
import sys
from pathlib import Path

import numpy as np

from ran.baselines import omnifold

run_dir = Path(sys.argv[1])
(run_dir / "config.json").write_text("{}")
omnifold._run_and_evaluate = lambda config, niter, epochs: ({}, [], np.ones(2))
omnifold.render_metrics = lambda *args, **kwargs: None
logging.basicConfig(level=logging.INFO)
omnifold.evaluate_single(run_dir)
"""

    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- isolated fixed script
        [sys.executable, "-c", script, str(run_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    metrics_path = run_dir / "metrics_omnifold.json"
    weights_path = run_dir / "omnifold_weights.npz"
    assert completed.returncode == 0, completed.stderr
    assert metrics_path.exists()
    assert weights_path.exists()
    assert str(metrics_path) in completed.stderr
    assert str(weights_path) in completed.stderr


def test_ibu_records_metric_and_weight_artifact_completion(
    monkeypatch, tmp_path, caplog
):
    import numpy as np
    from ran.baselines import ibu

    run_dir = tmp_path / "sample-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")

    def fake_run_and_evaluate(config, n_iterations, purity_threshold):
        del config, n_iterations, purity_threshold
        return {}, [], [np.ones(2)]

    monkeypatch.setattr(
        ibu,
        "_run_and_evaluate",
        fake_run_and_evaluate,
    )
    monkeypatch.setattr(ibu, "render_metrics", lambda *_args, **_kwargs: None)

    with caplog.at_level(logging.INFO, logger="ran.baselines.ibu"):
        ibu.evaluate_single(run_dir)

    metrics_path = run_dir / "metrics_ibu.json"
    weights_path = run_dir / "ibu_weights.npz"
    messages = [
        record.getMessage()
        for record in _completion_records(caplog, "ran.baselines.ibu")
    ]
    assert metrics_path.exists()
    assert weights_path.exists()
    assert any(
        str(metrics_path) in message and str(weights_path) in message
        for message in messages
    )
