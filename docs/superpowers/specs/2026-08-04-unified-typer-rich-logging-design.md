# Unified Typer CLI and Rich Logging Design

## Goal

Replace the repository's six Fire entry points and production `print()` calls
with one unified Typer command tree, standard-library logging rendered by
`rich.logging.RichHandler`, and Rich-native presentation for progress and
tabular reports.

The repository has not been published externally, so preserving the existing
module-level command paths or Fire-compatible option syntax is not required.

## Command Architecture

`src/ran/cli.py` will be the only module that defines Typer applications and
CLI options. It will export `app`, `baseline_app`, and `sweep_app`, and expose
this command tree:

```text
ran
├── train
├── evaluate
├── baseline
│   ├── omnifold
│   └── ibu
├── sweep
│   ├── ran
│   ├── omnifold
│   └── collect
└── leakage-check
```

The root application and both command groups will use
`rich_markup_mode="rich"`. `src/ran/__main__.py` will contain only the small
module entry point that imports `app` and invokes it.

The scientific and orchestration modules will remain independently callable
Python APIs. They will not import Typer or contain Typer decorators. Thin
decorated functions in `ran.cli` will translate CLI values into the types those
APIs consume.

The primary workflow will intentionally change from an implicit root command to
an explicit command:

```bash
uv run -m ran train --config params/1d_default.yaml
```

Other representative invocations will be:

```bash
uv run -m ran evaluate --run-dir runs/2026-...
uv run -m ran baseline omnifold --run-dir runs/2026-...
uv run -m ran baseline ibu --run-dir runs/2026-...
uv run -m ran sweep ran --s-index 3
uv run -m ran sweep omnifold --s-index 3
uv run -m ran sweep collect
uv run -m ran leakage-check --poison
```

Jet variables will use repeatable Typer options, for example
`--variable m --variable w`, instead of Fire's Python-literal tuple syntax.
Typer will provide validation, Rich-formatted help, and conventional Boolean
flags.

## Application Boundaries

The existing primary command body in `ran.__main__` will move to
`ran.workflow.run`. The `train` CLI wrapper will call that function. Evaluation
and baseline `main` functions will likewise receive domain-specific names that
describe their orchestration behavior.

The leakage-check implementation will move from `scripts/leakage_check.py` into
the `ran` package so installed code does not depend on a repository-only script.
The old script entry point will be removed, and the README will direct users to
`ran leakage-check`.

The cubic sweep's existing `run_ran`, `run_omnifold`, and `collect` functions
will stay as the business API and will be called by the three sweep commands.

## Logging Architecture

`src/ran/logging_config.py` will provide
`configure_logging(level: str) -> None`. A root Typer callback will call it once;
importing the package or calling scientific APIs will not configure the root
logger. As with conventional grouped CLIs, the global option precedes the
command, for example `ran --log-level DEBUG train ...`.

Every module that reports operational events will use a module logger:

```python
logger = logging.getLogger(__name__)
```

The shared configuration will use `logging.basicConfig(..., force=True)` with a
`RichHandler` configured for readable tracebacks and without interpreting log
messages as Rich markup. The default level will be `INFO`, with a root
`--log-level` option supporting standard logging level names. Configuration
will be deterministic when invoked repeatedly in tests.

Log levels will follow these rules:

- `INFO`: epoch summaries, cache hits and writes, model/run loading and saving,
  plot creation, baseline progress, and completed evaluation milestones.
- `WARNING`: incomplete sweep points, skipped dimensions, and recoverable
  per-run failures when processing a collection.
- `ERROR` with traceback via `logger.exception`: caught failures that prevent a
  requested operation, such as final metric evaluation.
- `DEBUG`: details useful for diagnosis but too noisy for the normal CLI.

Logging calls will use lazy parameter substitution rather than f-strings, in
line with the enabled Ruff `LOG` rules.

## Rich Presentation

Not every current `print()` call represents a log event. High-frequency download
state and the evaluation metric report are user-interface output:

- Dataset download progress will use a Rich progress display instead of carriage
  return `print()` calls. File completion and extraction milestones will still
  be logged.
- Evaluation metrics will be rendered as Rich tables. The calculations and
  returned metric dictionaries remain unchanged.

This separation keeps logs meaningful while still eliminating production
`print()` calls and retaining readable output in interactive terminals and
captured SLURM logs. Rich will automatically avoid raw ANSI styling when output
is not a capable terminal.

## Error Handling

The refactor will preserve existing failure semantics unless Typer validation
rejects invalid input earlier.

Batch-style commands that process several run directories will log an individual
failure and continue to the next run, matching current behavior. A failure of
the primary run's final metric calculation will include its traceback while
preserving the already-created checkpoints and plots. Invalid datasets,
variables, paths, and option values will produce Typer errors or the existing
domain exceptions rather than being silently coerced.

## Documentation

The README's usage examples, option table, project structure, evaluation,
baseline, sweep, and leakage-check commands will be updated for the unified
command tree. References to Fire and the removed script entry point will be
deleted. Help text in `ran.cli` will be concise and use Rich markup only where it
improves readability.

## Testing Strategy

Implementation will follow test-driven development:

1. Add tests using `typer.testing.CliRunner` that initially fail because the
   unified app does not exist.
2. Verify root help lists `train`, `evaluate`, `baseline`, `sweep`, and
   `leakage-check`, and group help lists the expected nested commands.
3. Test representative option conversion without running expensive training by
   replacing the target application function at the CLI boundary.
4. Test logging configuration installs exactly one `RichHandler`, honors the
   requested level, and is safe to call repeatedly.
5. Update output assertions to use `caplog` or Rich console capture according to
   whether the behavior is a log event or presentation.
6. Exercise help for every command so unsupported Typer annotations are caught.
7. Verify the source tree contains neither Fire imports nor production
   `print()` calls.
8. Run the complete test suite and the repository's formatting, Ruff, type, and
   complexity checks. Pre-existing unrelated validation failures, if any, will
   be reported separately from regressions introduced by this refactor.

The tests will not run full training or network downloads. Existing small-model
training tests remain the integration coverage for the scientific path.

## Non-Goals

- Preserving old module-level CLI invocation paths.
- Preserving Fire's underscore option spellings or Python-literal parsing.
- Adding JSON or structured logging.
- Adding log rotation, remote aggregation, or a permanent per-run log file.
- Redesigning scientific calculations, saved artifact formats, or SLURM job
  behavior beyond updating the invoked command.
