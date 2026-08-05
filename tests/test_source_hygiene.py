import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_production_python_uses_neither_builtin_print_nor_fire():
    offenders = []
    for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: print")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "fire":
                            offenders.append(
                                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: fire"
                            )
                if isinstance(node, ast.ImportFrom) and node.module == "fire":
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: fire")
    assert offenders == []
