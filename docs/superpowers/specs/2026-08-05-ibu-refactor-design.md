# IBU Evaluation Pipeline Refactor Design

## Goal

Refactor `ran.baselines.ibu._run_and_evaluate` into a short, typed orchestration
function whose collaborators make the IBU data flow, binning policy, numerical
invariants, and per-variable outcomes explicit.

The work remains internal to `src/ran/baselines/ibu.py` apart from focused tests.
It preserves the existing metric JSON layout and `ibu_weights.npz` keys used by
the plotting workflow. Existing purity-binning optimizations and documentation
changes already present in the worktree are not part of this refactor and must
be preserved.

## Root Cause and Methodological Choice

The current implementation uses two different definitions of bin membership:

- response-matrix and test-weight assignment use clipped `np.digitize` results,
  so underflow and overflow events enter the first and last bins;
- prior and observed-data counts use `np.histogram`, which discards values
  outside the finite edge range.

Consequently, the response and histograms can describe different event
populations. This is particularly relevant because purity edges are derived
from generator-level values while reconstructed values can extend beyond that
range. The last purity edge can also stop below the generator maximum when no
further sufficiently pure edge is found.

The refactor will make saturating edge bins the single policy for every IBU
input. A shared bin-assignment helper will clip underflow and overflow into the
first and last bins. Response construction, prior counts, observed-data counts,
and test-MC weight lookup will all use those assignments. Histogram counts will
be produced with `np.bincount`, ensuring that every path describes the same
event population.

This policy retains the existing response and test-weight behavior while
correcting the inconsistent histogram behavior. It will be stated explicitly
in helper docstrings and covered by tests.

## Configuration Boundary

The JSON object loaded from `config.json` is untyped external input. It will be
converted immediately into an immutable IBU-specific configuration dataclass.
The computation will consume that dataclass rather than repeatedly indexing a
`dict[str, Any]`.

The parser will validate before dataset loading that:

- `dataset` is one of the supported dataset names;
- `dim`, `n_samples`, and `batch_size` are positive integers;
- the optional data seed is an integer;
- a jet configuration contains a sequence of string variable names; and
- the number of jet variable names equals `dim`.

After loading, input preparation will verify that the array dimension agrees
with the configured dimension. Dataset-specific nested parameters needed by
the existing shared loader remain in the source mapping, but unstructured
configuration access will be isolated to this boundary.

Gaussian variable names will be generated from the validated dimension. Jet
variable names will come from the validated tuple, so later computation cannot
encounter a name/dimension mismatch.

## Typed Data and Result Models

An immutable internal data container will name the arrays used to build the
response and evaluate the result. It will distinguish:

- generator-level MC used for response construction;
- reconstructed MC used for response construction;
- reconstructed observed data used for unfolding;
- detector-level data and MC from the test split; and
- particle-level data and MC from the test split.

Input preparation will concatenate the train, validation, and test splits once,
derive these arrays, and verify compatible two-dimensional shapes and finite
values. It will also require nonempty response MC, observed data, and test MC,
because the algorithm and normalization are undefined without them.

Metric records will have an explicit `TypedDict` containing the nine existing
floating-point fields: before, after, and percentage improvement for
Wasserstein, Jensen-Shannon, and triangular distance. The top-level metric map
will therefore be typed as a mapping from the existing level-and-variable key to
that record.

The function will return an `IBUResult` dataclass with named fields:

- `metrics`;
- `variable_names`;
- `weights`, a double array with shape `(n_variables, n_test_mc)`; and
- ordered per-variable outcomes.

Each per-variable outcome will contain the variable name, completion status,
number of usable bins, and an optional skip reason. Status is either completed
or skipped. This metadata distinguishes a successful unfolding from an
identity-weight fallback without encoding that distinction indirectly through
missing metrics.

## Functional Decomposition

The implementation will use focused functions rather than a stateful runner
class:

1. A configuration parser validates the external mapping and returns the typed
   configuration.
2. An input-preparation helper loads and validates named response and evaluation
   arrays.
3. A bin-assignment helper maps one-dimensional values to saturating bin
   indices.
4. A count helper converts bin indices into floating-point counts with a known
   number of bins.
5. A per-variable unfolding helper owns purity binning, response construction,
   IBU iterations, and test-MC weight construction. It returns weights plus
   status metadata.
6. A scalar-dimension evaluation helper accepts one-dimensional reference and
   comparison arrays and produces one typed metric record. The existing metric
   functions already accept one-dimensional inputs, so no temporary
   two-dimensional slicing is required.
7. `_run_and_evaluate` coordinates these functions, evaluates both detector and
   particle levels, fills one row of the weight matrix per variable, and
   constructs `IBUResult`.

Binning plus unfolding remains one cohesive per-variable numerical operation.
Metric calculation remains separate so the methodological result is not mixed
with reporting concerns.

## Numerical Invariants and Failure Semantics

The following invariants will be enforced explicitly:

- All arrays entering binning and metric evaluation are finite.
- The response, prior, and observed-data histogram use the same number of bins
  and the same saturating assignments.
- The sum of the prior equals the number of response MC events.
- The sum of the observed-data histogram equals the number of observed events.
- The unfolded histogram and constructed weights are finite and nonnegative.
- Every completed weight vector has the test-MC event count as its length.
- The normalization mean is finite and strictly positive.
- Final normalized weights are finite, nonnegative, and have mean one within
  numerical tolerance.

The unfolded-to-prior ratio will use `np.divide` with an explicitly zeroed
output and a `prior > 0` mask. A zero-prior bin therefore receives zero bin
weight rather than an arbitrarily large value from adding `EPS`. Because the
test MC is included in the response-building MC population, no test event
should occupy a zero-prior bin. If the unfolded result nevertheless assigns
material nonzero mass to such a bin, the code will raise an invariant error
rather than conceal it.

Nonfinite, negative, or non-normalizable weights indicate a numerical failure
and will raise a descriptive exception. They will not be converted into an
ordinary skip.

If purity binning produces fewer than two usable bins, the outcome is instead a
methodological skip. That variable receives an identity weight vector, a
skipped status, and a clear reason. Detector- and particle-level metrics are
still computed with those identity weights, so every configured variable has a
weight row and metric records. The explicit status records that no unfolding
was performed.

## Artifact and Caller Compatibility

`evaluate_single` will consume named fields from `IBUResult` rather than
unpacking a tuple. It will continue writing the existing flat metric dictionary
to `metrics_ibu.json` so `render_metrics` and existing consumers remain valid.

Although weights are held internally as one matrix, `ibu_weights.npz` will
retain the existing `weights_0`, `weights_1`, and subsequent keys. Each row of
the matrix will be written under the corresponding key. The current workflow
loader and plotting functions therefore require no changes.

Skipped-variable status metadata is available in the returned result and is
also emitted through the existing warning log. Persisting a new status artifact
or changing the metric JSON schema is outside this refactor's scope.

## Testing Strategy

Implementation will follow test-driven development with focused IBU tests:

1. Demonstrate that underflow and overflow values are saturated and that prior,
   observed-data, and response populations are conserved.
2. Validate malformed configuration failures, including missing fields,
   nonpositive dimensions, unsupported datasets, invalid jet variable names,
   and variable-count mismatches.
3. Verify a zero-prior bin receives explicit zero ratio behavior and that
   inconsistent unfolded mass in such a bin fails clearly.
4. Verify normalization rejects zero, negative, and nonfinite means or weights.
5. Verify insufficient purity bins produce identity weights, complete metrics,
   and an explicit skipped outcome with a reason.
6. Verify successful per-variable unfolding returns finite, nonnegative,
   mean-one weights.
7. Verify `_run_and_evaluate` returns the named result object with weight shape
   `(dim, n_test_mc)`, ordered names and outcomes, and detector/particle metrics
   for every variable.
8. Update the completion-logging test double to return `IBUResult` and verify
   the existing metric and weight artifacts are still written.
9. Run the complete test suite and the repository's formatting, linting, type,
   and complexity checks.

## Non-Goals

- Refactoring OmniFold or extracting shared baseline abstractions.
- Changing the purity-edge search algorithm or threshold semantics.
- Changing the IBU update equations beyond explicit downstream ratio handling.
- Changing metric definitions or their JSON key layout.
- Changing the baseline weight archive format consumed by plotting.
- Persisting a new result or status artifact.
- Refactoring the shared dataset loader or repository-wide configuration types.
