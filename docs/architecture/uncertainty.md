# Uncertainty Quantification

The `deconvolve.uncertainty` package provides variance budgeting and bin-to-bin covariance estimation for unfolded physics measurements.

---

## Three Sources of Uncertainty

In machine learning unfolding, experimental publications often conflate three distinct sources of variance:

| Source                     | Varied By                | Nature                                                                       | Treatment                             |
| :------------------------- | :----------------------- | :--------------------------------------------------------------------------- | :------------------------------------ |
| **Finite Sample**          | Non-parametric Bootstrap | Real physical variation: the 1M events represent one random draw from nature | **Report as measurement uncertainty** |
| **Split & Batch Order**    | `data_seed`              | Algorithm artifact: which events land in train/val/test                      | Eliminate by ensembling               |
| **Network Initialization** | `seed`                   | Algorithm artifact: random initial weights of $g$ and $d$                    | Eliminate by ensembling               |

Only the first source represents a physical measurement uncertainty. Folding initialization or batch-order variance into the quoted error bar inflates the physics uncertainty with algorithmic noise.

---

## Two-Way Crossed Random-Effects ANOVA

To disentangle these sources without double-counting interaction terms, `deconvolve.uncertainty.design` runs a balanced $B \times S$ crossed design (Bootstrap datasets $\times$ Initial seeds):

$$T(D, S) = \mu + \alpha(D) + \beta(S) + \epsilon(D, S)$$

### Naive Sweeps vs. Crossed Grids

A naive pair of one-dimensional sweeps (varying seed at fixed dataset, and varying dataset at fixed seed) measures:

$$\sigma_{\text{naive}}^2 = \sigma_\alpha^2 + \sigma_\beta^2 + 2\sigma_\epsilon^2$$

whereas the true variance is:

$$\sigma_{\text{true}}^2 = \sigma_\alpha^2 + \sigma_\beta^2 + \sigma_\epsilon^2$$

Because $\sigma_\epsilon^2$ (the interaction between initialization and dataset) is significant in min-max adversarial games, the naive sum systematically overestimates the error. The balanced crossed grid solves for $\sigma_\alpha^2$, $\sigma_\beta^2$, and $\sigma_\epsilon^2$ simultaneously.

---

## Common Held-Out Evaluation Set

Bootstrap resamples contain duplicate events and differing sample compositions, making their per-event weight vectors incommensurable.

To allow valid covariance calculation:

- A fixed block of particle-level Monte Carlo events (`z_gen`) is held out **before** resampling.
- Every cell in the $B \times S$ design evaluates its generator $g(z)$ on this identical evaluation block.
- Because the evaluation set is held constant across all cells, its finite size shifts all cells together and cancels out of cross-cell contrasts entirely.
