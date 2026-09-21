# Baselines API (`deconvolve.baselines`)

The `deconvolve.baselines` package contains reference implementations for Iterative Bayesian Unfolding (IBU) and OmniFold.

---

## Shared Protocol (`deconvolve.baselines._shared`)

::: deconvolve.baselines._shared
    options:
      show_root_heading: true
      show_source: true
      members:
        - parse_run_config
        - prepare_populations
        - load_populations
        - evaluate_dimension

---

## IBU (`deconvolve.baselines.ibu`)

::: deconvolve.baselines.ibu
    options:
      show_root_heading: true
      show_source: true
      members:
        - evaluate_runs
        - evaluate_run

---

## OmniFold (`deconvolve.baselines.omnifold`)

::: deconvolve.baselines.omnifold
    options:
      show_root_heading: true
      show_source: true
      members:
        - evaluate_runs
        - evaluate_run
