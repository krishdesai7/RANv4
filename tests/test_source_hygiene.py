import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def _forbidden_kind(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ):
        return "print"
    if isinstance(node, ast.Import) and any(
        alias.name == "fire" for alias in node.names
    ):
        return "fire"
    if isinstance(node, ast.ImportFrom) and node.module == "fire":
        return "fire"
    return None


def test_production_python_uses_neither_builtin_print_nor_fire():
    paths = (
        path
        for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")
        for path in source_root.rglob("*.py")
    )
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {kind}"
        for path in paths
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.stmt, ast.expr))
        if (kind := _forbidden_kind(node)) is not None
    ]
    assert offenders == []
