from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import] -- isolated type check
from pathlib import Path


def test_builder_contract_hides_unsupported_model_members(tmp_path: Path) -> None:
    """Pyrefly sees narrow builder and persisted-model contracts."""
    probe = tmp_path / "builder_contract.py"
    probe.write_text(
        "from pathlib import Path\n\n"
        "from ran.models import build_generator\n"
        "from ran.workflow import _load_artifacts\n\n"
        "build_generator().unsupported_builder_model_member\n"
        "_load_artifacts(Path('run'))[0].unsupported_loaded_model_member\n"
    )
    root = Path(__file__).parents[1]
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local probe
        [
            str(root / ".venv" / "bin" / "pyrefly"),
            "check",
            "--config",
            str(root / "pyproject.toml"),
            str(probe),
        ],
        check=False,
        capture_output=True,
        cwd=root,
        text=True,
    )

    assert completed.returncode != 0, completed.stdout
    assert "unsupported_builder_model_member" in completed.stdout
    assert "unsupported_loaded_model_member" in completed.stdout
    assert "missing-attribute" in completed.stdout
