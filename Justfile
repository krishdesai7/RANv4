# List available recipes.
default:
    @just --list

# Apply safe lint fixes, then format.
fix:
    uv sync
    ruff check --fix
    uv format

# Apply unsafe lint fixes, then format.
fix-unsafe:
    uv sync
    ruff check --fix --unsafe-fixes
    uv format

# Infer annotations and imports, then apply safe fixes.
infer:
    uv sync
    pyrefly infer --return-types --parameter-types --imports --containers
    just fix

# Check formatting without modifying files.
format-check *args:
    uv format --check {{ args }}

# Run Ruff lint checks.
lint:
    ruff check

# Run Pyrefly type checks.
typecheck:
    pyrefly check --min-severity info

# Run complexity checks.
complexity:
    complexipy

# Run tests, optionally forwarding arguments to pytest.
test *args:
    uv run pytest -q {{ args }}

# Audit locked dependencies for known vulnerabilities.
audit:
    uv audit

# Run all local, read-only validation.
check:
    uv sync --locked
    just format-check
    just lint
    just typecheck
    just complexity
    just test

# Run the full CI validation suite.
ci:
    just check
    just audit

# Upgrade all locked dependencies and synchronize the environment.
upgrade:
    uv lock --upgrade
    uv sync
