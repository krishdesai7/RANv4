"""Can a TensorFlow subprocess get GPU memory while a JAX parent holds the card?

This is the fail-fast check standing in front of any decision to bring OmniFold
into this repository. The quarantine itself is settled -- a PEP 723 script run
through `uv run --no-project` provisions Python 3.13 and TensorFlow in an
interpreter that cannot see `ran`, which is proven and cheap. What is *not*
settled is whether the two processes can share one A100, and that question only
has an answer on a machine with an A100 in it.

The hazard is in the Precision section of CLAUDE.md: **JAX preallocates ~75% of
GPU memory on its first device allocation.** A `ran baseline omnifold` parent
reaches the device long before it spawns a worker -- `load_populations` alone
does it -- so by the time TensorFlow starts, three quarters of the card is
already spoken for and does not come back. Deleting the array does not release
it; the preallocation is the pool, not the array, which is why there is no
"just free it first" arm below.

Five arms, each a separate process because `XLA_PYTHON_CLIENT_*` is read when
the JAX backend initialises and cannot be changed afterwards. Reading them in
one interpreter and re-importing is the obvious way to write this and it
measures nothing: every arm after the first would inherit the first one's pool.

| Arm                   | Parent                    | What a pass means           |
| --------------------- | ------------------------- | --------------------------- |
| `control`             | never touches JAX         | the worker works on this node |
| `preallocate-default` | JAX on GPU, defaults      | no fix needed               |
| `preallocate-false`   | `..._PREALLOCATE=false`   | the cheap fix works         |
| `mem-fraction-0.4`    | `..._MEM_FRACTION=0.4`    | the budgeted fix works      |
| `parent-on-cpu`       | `JAX_PLATFORMS=cpu`       | the fallback design works   |

(the two truncated names are `XLA_PYTHON_CLIENT_PREALLOCATE` and
`XLA_PYTHON_CLIENT_MEM_FRACTION`.)

`control` is the arm to read first. If it fails, nothing below it means
anything -- the node, the CUDA driver or the uv script cache is the problem,
not coexistence -- and the other four results should be discarded rather than
interpreted.

Run it inside a GPU allocation, not on a login node:

    srun -C gpu --qos=shared --gpus=1 --cpus-per-task=32 --time=00:20:00 \
        uv run benchmarks/gpu_coexistence.py

**Warm the worker environment on a login node first.** uv resolves the PEP 723
header at first run and compute nodes generally have no outbound network, so a
cold cache fails the whole benchmark with a download error that looks nothing
like an OOM:

    uv run --no-project benchmarks/_tf_probe_worker.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

WORKER: Path = Path(__file__).parent / "_tf_probe_worker.py"

# Large enough to force a real device allocation, small enough that it is never
# itself the reason the card is full: what fills the card is the preallocation,
# and the point of the benchmark is to attribute the failure to that.
PARENT_ARRAY_BYTES: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class Arm:
    name: str
    touch_jax: bool
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""


ARMS: tuple[Arm, ...] = (
    Arm("control", touch_jax=False, note="worker alone on the card"),
    Arm(
        "preallocate-default", touch_jax=True, note="what `ran baseline` would do today"
    ),
    Arm(
        "preallocate-false",
        touch_jax=True,
        env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        note="JAX grows on demand",
    ),
    Arm(
        "mem-fraction-0.4",
        touch_jax=True,
        env={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.4"},
        note="JAX capped, ~60% left",
    ),
    Arm(
        "parent-on-cpu",
        touch_jax=True,
        env={"JAX_PLATFORMS": "cpu"},
        note="parent never on device",
    ),
)


def nvidia_free_mib() -> tuple[int, int] | None:
    """`(free, total)` MiB on GPU 0, or None where there is no nvidia-smi."""
    # Fixed argv, no shell. `nvidia-smi` is deliberately a bare name: it is
    # resolved on PATH because its location differs between the driver packages
    # a cluster might have installed.
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except OSError, subprocess.SubprocessError, IndexError:
        return None
    try:
        free, total = (int(v.strip()) for v in out.split(","))
    except ValueError:
        return None
    return free, total


def run_worker() -> dict[str, object]:
    """Spawn the TensorFlow worker and parse its one line of JSON.

    `--no-project` is not decoration. Without it uv would try to resolve the
    script against this repository's `pyproject.toml`, whose `requires-python`
    is `>=3.14` -- irreconcilable with the worker's `==3.13.*`. That fails
    loudly rather than silently, which is the good case, but it fails.
    """
    # Fixed argv, no shell; the only interpolated element is a path inside
    # this file's own directory. `uv` is a bare name on purpose -- it is what
    # the user invoked this benchmark with.
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["uv", "run", "--no-project", str(WORKER)],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    line = proc.stdout.strip().splitlines()
    if not line:
        return {
            "status": "error",
            "detail": f"worker exited {proc.returncode} with no JSON",
            "stderr": proc.stderr[-1500:],
        }
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError:
        return {
            "status": "error",
            "detail": "unparseable worker stdout",
            "stdout": proc.stdout[-800:],
            "stderr": proc.stderr[-1500:],
        }


def as_parent(arm: Arm) -> dict[str, object]:
    """The middle process: hold the card the way a real run would, then spawn.

    The array is bound to a local and read after the worker returns so that
    nothing -- not the garbage collector, not a clever XLA rewrite -- can retire
    it while the worker is running. A parent that quietly released the device
    would report a pass that the real baseline could not reproduce.
    """
    report: dict[str, object] = {"arm": arm.name}
    held = None

    if arm.touch_jax:
        import jax
        import jax.numpy as jnp

        n = PARENT_ARRAY_BYTES // 4
        held = jnp.ones((n,), dtype=jnp.float32)
        jax.block_until_ready(held)
        report["parent_platform"] = jax.default_backend()
        report["parent_device"] = str(held.device)
    else:
        report["parent_platform"] = "none"
        report["parent_device"] = "none"

    mem = nvidia_free_mib()
    if mem is not None:
        report["free_mib_before_worker"], report["total_mib"] = mem

    report["worker"] = run_worker()

    if held is not None:
        # Reading the array after the worker returns is what forces the parent
        # to still own it for the worker's whole lifetime. The value is checked
        # loosely -- the question is whether the buffer survived, not arithmetic.
        report["parent_array_live"] = bool(abs(float(held[0]) - 1.0) < 1e-6)
    return report


def verdict(row: dict[str, object]) -> tuple[str, str]:
    status = str(row.get("worker", {}).get("status", "error"))  # type: ignore[union-attr]
    return {
        "ok": ("[green]PASS[/green]", "worker ran on the GPU"),
        "oom": ("[red]FAIL[/red]", "worker could not get memory"),
        "cpu_fallback": (
            "[yellow]SILENT[/yellow]",
            "worker saw no GPU and used the CPU",
        ),
    }.get(status, ("[red]ERROR[/red]", "worker did not report"))


def render(rows: list[dict[str, object]], console: Console) -> None:
    table = Table(title="TensorFlow worker vs. a JAX parent, one GPU", show_lines=False)
    for col in ("arm", "parent", "free MiB", "worker", "result", "what it means"):
        table.add_column(col, overflow="fold")
    for row in rows:
        mark, meaning = verdict(row)
        free = row.get("free_mib_before_worker")
        total = row.get("total_mib")
        table.add_row(
            str(row.get("arm")),
            str(row.get("parent_platform")),
            f"{free} / {total}" if free is not None else "-",
            str(row.get("worker", {}).get("status", "?")),  # type: ignore[union-attr]
            mark,
            meaning,
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        action="append",
        choices=[a.name for a in ARMS],
        help="run only these arms (repeatable); default is all five",
    )
    parser.add_argument("--json", type=Path, help="also write the raw results here")
    parser.add_argument("--as-parent", help=argparse.SUPPRESS)
    args = parser.parse_args()

    by_name = {a.name: a for a in ARMS}

    # Re-exec of self as the middle process. Reached only via subprocess below.
    if args.as_parent:
        print(json.dumps(as_parent(by_name[args.as_parent])))
        return

    console = Console(stderr=True)
    if not WORKER.exists():
        console.print(f"[red]missing worker script:[/red] {WORKER}")
        raise SystemExit(2)

    selected = [by_name[n] for n in (args.arm or [a.name for a in ARMS])]
    rows: list[dict[str, object]] = []

    for arm in selected:
        console.print(f"[cyan]arm[/cyan] {arm.name} -- {arm.note}")
        env = os.environ | arm.env
        # Fixed argv, no shell: this interpreter, this file, a name from ARMS.
        proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, __file__, "--as-parent", arm.name],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=3600,
        )
        out = proc.stdout.strip().splitlines()
        if out:
            rows.append(json.loads(out[-1]) | {"env": arm.env})
        else:
            rows.append(
                {
                    "arm": arm.name,
                    "env": arm.env,
                    "parent_platform": "?",
                    "worker": {
                        "status": "error",
                        "detail": f"parent exited {proc.returncode}",
                    },
                    "stderr": proc.stderr[-1500:],
                }
            )

    render(rows, console)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        console.print(f"wrote {args.json}")

    if (
        rows
        and verdict(rows[0])[0] != "[green]PASS[/green]"
        and selected[0].name == "control"
    ):
        console.print(
            "[yellow]control did not pass -- the node, the driver or the uv "
            "script cache is the problem. Discard the other arms.[/yellow]"
        )


if __name__ == "__main__":
    main()
