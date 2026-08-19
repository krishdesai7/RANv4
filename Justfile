# List available recipes.
default:
    @printf '%s\n' \
        'Available recipes:' \
        '  Immutable / read-only:' \
        '    format *args      # Check formatting without modifying files.' \
        '    lint              # Run Ruff lint checks.' \
        '    type-check       # Run Pyrefly type checks.' \
        '    complexity        # Run complexity checks.' \
        '    test *args        # Run tests, optionally forwarding arguments to pytest.' \
        '    audit             # Audit locked dependencies for known vulnerabilities.' \
        '    validate          # Run all local, read-only validation.' \
        '    ci                # Run the full CI validation suite.' \
        '  Mutable / writes:' \
        '    lint-fix         # Apply safe lint fixes, then format.' \
        '    lint-fix-unsafe  # Apply unsafe lint fixes, then format.' \
        '    infer             # Infer annotations and imports, then apply safe fixes.' \
        '    upgrade-deps     # Upgrade all locked dependencies and synchronize the environment.'

# --- Read-only checks ---

# Check formatting without modifying files.
format *args:
    uv format --check {{ args }}

# Run Ruff lint checks.
lint:
    uv run --locked ruff check

# Run Pyrefly type checks.
type-check:
    uv run --locked pyrefly check --min-severity info
    uv check --locked

# Run complexity checks.
complexity:
    uv run --locked complexipy --suggest-refactors

# Run tests, optionally forwarding arguments to pytest.
test *args:
    uv run --locked pytest -q {{ args }}

# Audit locked dependencies for known vulnerabilities.
audit:
    uv audit --locked

# Run all local, read-only validation.
validate:
    uv sync --locked
    just format
    just lint
    just type-check
    just complexity
    just test

# Run the full CI validation suite.
ci:
    just validate
    just audit

# --- Mutable operations (writes) ---

# Apply safe lint fixes, then format.
lint-fix:
    uv sync
    uv run ruff check --fix
    uv format

# Apply unsafe lint fixes, then format.
lint-fix-unsafe:
    uv sync
    uv run ruff check --fix --unsafe-fixes
    uv format

# Infer annotations and imports, then apply safe fixes.
infer:
    uv sync
    uv run pyrefly infer --return-types --parameter-types --imports --containers
    uv check --fix
    just lint-fix

# Upgrade all locked dependencies and synchronize the environment.
upgrade-deps:
    uv lock --upgrade
    uv sync
