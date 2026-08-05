import json
import logging
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- backend isolation
import sys
from types import SimpleNamespace


def _completion_records(caplog, logger_name: str):
    return [
        record
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.INFO
    ]


def test_raw_download_records_completion(monkeypatch, tmp_path, caplog) -> None:
    from ran.data import download
    from rich.progress import Progress

    destination = tmp_path / "sample.npz"

    def fake_urlretrieve(_url, _dest, reporthook) -> None:
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


def test_evaluation_records_metrics_artifact_completion(
    monkeypatch, tmp_path, caplog
) -> None:
    import keras
    import numpy as np
    from ran import evaluate

    run_dir = tmp_path / "sample-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"dataset": "gaussian", "dim": 1}')
    (run_dir / "generator.keras").touch()
    z = np.array([[0.0], [1.0], [2.0], [3.0]])
    x = z.copy()
    y = np.array([1, 1, 0, 0])

    # evaluate_run imports keras inside the function (so ran.evaluate stays
    # keras-free for the OmniFold backend pin), so patch the real module.
    monkeypatch.setattr(keras.saving, "load_model", lambda _path: object())
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


def test_omnifold_records_metric_and_weight_artifact_completion(tmp_path) -> None:
    run_dir = tmp_path / "sample-run"
    run_dir.mkdir()
    script = """
import logging
import sys
from pathlib import Path

import numpy as np

from ran.baselines import omnifold

run_dir = Path(sys.argv[1])
(run_dir / "config.json").write_text('{"dim": 1, "n_samples": 1, "batch_size": 1}')
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
) -> None:
    import numpy as np
    from ran.baselines import _shared as shared
    from ran.baselines import ibu

    run_dir = tmp_path / "sample-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"dim": 2, "n_samples": 2, "batch_size": 1}')

    metric_record: shared.MetricRecord = {
        "wasserstein_before": 1.0,
        "wasserstein_after": 0.5,
        "wasserstein_improvement_pct": 50.0,
        "jensenshannon_before": 0.4,
        "jensenshannon_after": 0.2,
        "jensenshannon_improvement_pct": 50.0,
        "triangular_before": 0.3,
        "triangular_after": 0.1,
        "triangular_improvement_pct": 200.0 / 3.0,
    }

    def fake_run_and_evaluate(
        config: shared.RunConfig, n_iterations: int, purity_threshold: np.double
    ) -> ibu.IBUResult:
        del config, n_iterations, purity_threshold
        return ibu.IBUResult(
            metrics={"detector_mass": metric_record},
            variable_names=("mass", "momentum"),
            weights=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.double),
            outcomes=(
                ibu.VariableOutcome(
                    variable_name="mass",
                    status="completed",
                    n_bins=2,
                ),
                ibu.VariableOutcome(
                    variable_name="momentum",
                    status="completed",
                    n_bins=2,
                ),
            ),
        )

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
    assert json.loads(metrics_path.read_text()) == {"detector_mass": metric_record}
    with np.load(weights_path) as weights:
        assert set(weights.files) == {"weights_0", "weights_1"}
        np.testing.assert_array_equal(weights["weights_0"], [1.0, 2.0])
        np.testing.assert_array_equal(weights["weights_1"], [3.0, 4.0])
    assert any(
        str(metrics_path) in message and str(weights_path) in message
        for message in messages
    )
