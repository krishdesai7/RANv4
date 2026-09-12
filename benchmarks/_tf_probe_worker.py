# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "numpy>=2.5.2",
#     "tensorflow[and-cuda]; sys_platform == 'linux' and platform_machine == 'x86_64'",
#     "tensorflow; sys_platform == 'darwin'",
# ]
# ///
"""The TensorFlow half of `gpu_coexistence.py`. Never imported -- only `uv run`.

This file deliberately has no `ran` import and is not reachable from the
package: it runs under Python 3.13 with the TensorFlow Keras backend, which is
exactly the environment `src/ran` cannot coexist with. The PEP 723 header above
is the whole quarantine mechanism; `uv run --no-project` provisions it.

It answers one question -- **can TensorFlow get usable GPU memory right now?**
-- and is careful to separate three outcomes that a naive probe collapses into
one:

* `ok`            TF ran the op on the GPU. The coexistence works.
* `cpu_fallback`  TF ran, but on the CPU, because it saw no GPU at all. This is
                  the dangerous outcome: nothing raises, the baseline just runs
                  ~50x slower and the result looks fine. An explicit
                  `tf.device("/GPU:0")` is what turns it into an error, so the
                  probe pins the device rather than letting TF place the op.
* `oom`           TF saw the GPU and could not get memory on it. This is the
                  failure the whole benchmark exists to detect.

The op is a real matmul rather than a device query because TF's memory pool is
lazy about the *first* allocation but greedy after it: listing devices succeeds
on a card that has nothing left to give. Only touching it settles the question.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# The size is chosen to need real memory (~3 x 512MB of float32) without being
# a meaningful fraction of a 40GB card on its own: a failure here is a failure
# to get a *foothold*, not a failure to fit a large model.
SIDE: int = 8192
ALLOCATION_BYTES: int = SIDE * SIDE * 4


def _peak_bytes() -> int | None:
    """Peak GPU bytes TF's allocator reached, if it can report them."""
    import tensorflow as tf

    try:
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except (KeyError, ValueError, RuntimeError):
        return None


def probe() -> dict[str, object]:
    import numpy as np
    import tensorflow as tf

    result: dict[str, object] = {
        "tf_version": tf.__version__,
        "python": sys.version.split()[0],
        "requested_bytes": ALLOCATION_BYTES,
    }

    gpus = tf.config.list_physical_devices("GPU")
    result["gpus_visible"] = [d.name for d in gpus]

    if not gpus:
        # Not an error on a laptop; the driver decides whether it is one here.
        result["status"] = "cpu_fallback"
        result["detail"] = "tf.config.list_physical_devices('GPU') returned nothing"
        return result

    try:
        # Pinned, not placed: soft placement would silently fall back to CPU on
        # OOM and report success.
        with tf.device("/GPU:0"):
            a = tf.random.normal((SIDE, SIDE), dtype=tf.float32)
            b = tf.random.normal((SIDE, SIDE), dtype=tf.float32)
            c = tf.linalg.matmul(a, b)
            checksum = float(tf.reduce_sum(c).numpy())
        result["status"] = "ok"
        result["device_used"] = c.device
        result["checksum_finite"] = bool(np.isfinite(checksum))
        result["peak_bytes"] = _peak_bytes()
    except tf.errors.ResourceExhaustedError as exc:
        result["status"] = "oom"
        result["detail"] = str(exc).splitlines()[0][:400]
    except (tf.errors.InternalError, tf.errors.UnknownError, RuntimeError) as exc:
        # A CUDA context that cannot be created at all lands here rather than in
        # ResourceExhaustedError, and on a full card it is the common shape.
        result["status"] = "oom"
        result["detail"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}"
    return result


def main() -> None:
    try:
        result = probe()
    except BaseException as exc:  # noqa: BLE001 -- the driver needs the reason, not a traceback on stderr
        result = {
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
    # One line of JSON on stdout is the entire protocol. TF writes banners to
    # stderr regardless of TF_CPP_MIN_LOG_LEVEL, so stdout must stay clean.
    print(json.dumps(result))


if __name__ == "__main__":
    main()
