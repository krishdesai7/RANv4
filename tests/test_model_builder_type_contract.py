from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import] -- isolated type check
from pathlib import Path


def test_builder_contract_hides_unsupported_model_members(tmp_path: Path) -> None:
    """Pyrefly sees the deliberately narrow public builder contract."""
    probe = tmp_path / "builder_contract.py"
    probe.write_text(
        "from ran.models import build_generator\n\n"
        "build_generator().unsupported_model_member\n"
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
    assert "unsupported_model_member" in completed.stdout
    assert "missing-attribute" in completed.stdout
