import os
import sys

import numpy as np
import pytest

# This test trains a real (tiny) OmniFold model. Local/dev runs and CI should
# NOT train models — that is cluster work — so it is opt-in. Enable with:
#   RAN_RUN_SLOW=1 uv run pytest tests/test_omnifold.py
#
# It must also run in a process that has not already committed Keras to JAX:
# omnifold needs the TensorFlow backend (its loss calls raw tf ops), and the
# backend is fixed at first `keras` import. Hence the second guard — run this
# file on its own, not as part of the full suite.
pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RAN_RUN_SLOW") != "1",
        reason="trains a real OmniFold model; set RAN_RUN_SLOW=1 to run",
    ),
    pytest.mark.skipif(
        "keras" in sys.modules
        and sys.modules["keras"].backend.backend() != "tensorflow",
        reason="needs the TensorFlow backend; run tests/test_omnifold.py alone",
    ),
]


def test_omnifold_unfold_returns_mean_normalized_weights() -> None:
    from ran.baselines.omnifold import omnifold_unfold

    rng = np.random.default_rng(0)
    n = 500
    z_gen = rng.normal(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    x_sim = z_gen + 0.1 * rng.normal(size=(n, 1)).astype(np.float32)
    x_data = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

    w = omnifold_unfold(x_data, x_sim, z_gen, niter=1, epochs=2, batch_size=128)

    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-5)
