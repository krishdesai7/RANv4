import json
import logging
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

    def fake_urlretrieve(_url, filename, reporthook) -> None:
        assert filename == destination
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
    from ran.rantypes import ZXY, Events

    run_dir = tmp_path / "sample-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"dataset": "gaussian", "dim": 1}')
    (run_dir / "generator.keras").touch()
    z = np.array([[0.0], [1.0], [2.0], [3.0]])
    test_data = ZXY(Events(z, z.copy()), np.array([1, 1, 0, 0], dtype=np.ubyte))

    monkeypatch.setattr(keras.saving, "load_model", lambda _path: object())
    monkeypatch.setattr(
        evaluate, "_load_splits", lambda _config: SimpleNamespace(test=object())
    )

    # Both are called by keyword, so the stubs have to name their parameters.
    def fake_collect_test_data(test_ds):
        del test_ds
        return test_data

    def fake_get_weights(model, z_gen):
        del model, z_gen
        return np.ones(2)

    monkeypatch.setattr(evaluate, "_collect_test_data", fake_collect_test_data)
    monkeypatch.setattr(evaluate, "_get_weights", fake_get_weights)
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
            weights=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.single),
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
