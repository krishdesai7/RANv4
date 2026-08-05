from ran import run


def test_primary_workflow_has_a_domain_entry_point() -> None:
    assert callable(run)


def test_leakage_check_has_a_package_entry_point() -> None:
    from ran.leakage import run_leakage_check

    assert callable(run_leakage_check)


def test_batch_orchestrators_have_domain_names() -> None:
    from ran.baselines.ibu import evaluate_runs as evaluate_ibu_runs
    from ran.evaluate import evaluate_runs

    assert callable(evaluate_runs)
    assert callable(evaluate_ibu_runs)
