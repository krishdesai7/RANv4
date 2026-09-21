# Quickstart

This walkthrough takes you from zero to a fully trained and evaluated reweighting model in five minutes.

---

## 1. Train a 1D Gaussian Model

Gaussian toy models provide an intuitive testbed where the truth is analytically known. In this example, the Monte Carlo simulation differs from nature by a shift in mean ($\mu = 0.5$ vs $0.0$) and a slight difference in width ($\sigma = 0.9$ vs $1.0$).

Run the training workflow using the pre-configured YAML parameters in `params/1d_default.yaml`:

```shell
ran train --config params/1d_default.yaml
```

During training, RAN logs progress via Rich:

```shell
[INFO] Initializing Gaussian dataset (dim=1, samples=100,000)
[INFO] Generator: 2 layers, 64 hidden units
[INFO] Discriminator: 2 layers, 64 hidden units
[INFO] Epoch  10/100 | D Loss: 0.6912 | G Loss: 0.6945 | MMD: 0.00342
...
[INFO] Best model selected at Epoch 82 (Validation MMD: 0.00018)
[INFO] Run artifacts saved to runs/2026-09-19-164500/
```

### Useful CLI Overrides

You can easily override hyperparameters directly from the command line:

```shell
# Train with 128 hidden units, 3 layers, and 200 epochs
ran train --config params/1d_default.yaml -u 128 -l 3 -e 200

# Specify a custom run directory tag
ran train --config params/1d_default.yaml --tag test-run
```

---

## 2. Evaluate the Run

Once training completes, evaluate the quality of the reweighting on the held-out test split:

```shell
ran evaluate --run-dir runs/2026-09-19-164500
```

This calculates three distance metrics before and after reweighting:

1. **1D Wasserstein-1 Distance**: $\int |F_{\text{ref}}(t) - F_{\text{comp}}(t)| \, dt$
2. **Jensen-Shannon Divergence**: Symmetrized relative entropy
3. **Triangular Discriminator**: $\int \frac{(p(x) - q(x))^2}{p(x) + q(x)} \, dx$

The evaluated metrics are written back to `runs/2026-09-19-164500/metrics.json`.

---

## 3. Generate a Summary Report

Generate publication-ready diagnostic plots and a LaTeX report summarizing the run:

```shell
ran report --run-dir runs/2026-09-19-164500
```

This generates:

- Cumulative distribution function (CDF) comparisons
- Histogram ratio plots
- Loss curves and MMD trajectory over epochs
- `report.tex` and compiled `report.pdf` inside `runs/2026-09-19-164500/artifacts/`

---

## 4. Jet Substructure (High-Dimensional Physics)

To train on real high-energy physics data (12 jet substructure observables from the Zenodo dataset):

```shell
# Train on all 12 jet variables
ran train -D jets

# Or train on specific variables (e.g. mass and width)
ran train -D jets -v m -v w
```

Dataset downloads and cache management are handled automatically under `.cache/`.
