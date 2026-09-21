# Adversarial Reweighting

Reweighting Adversarial Networks (RAN) frame the correction of simulated distributions as a zero-sum game between two neural networks across two distinct representation spaces: the **particle level** (unobserved truth $z$) and the **detector level** (reconstructed observation $x$).

---

## The Adversarial Formulation

Let:

- $z \in \mathcal{Z}$ denote particle-level (truth) kinematics.
- $x \in \mathcal{X}$ denote detector-level (reconstructed) kinematics.
- $p_{\text{data}}(x)$ denote the distribution of observed nature events at detector level.
- $p_{\text{sim}}(x, z)$ denote the joint distribution of simulated events, with marginal detector distribution $p_{\text{sim}}(x) = \int p_{\text{sim}}(x | z) p_{\text{gen}}(z) \, dz$.

The goal is to find a per-event weight function $w(z)$ defined on particle-level features such that the reweighted simulation matches data at detector level:

$$p_{\text{reweighted}}(x) = \int p_{\text{sim}}(x | z) \, w(z) \, p_{\text{gen}}(z) \, dz = p_{\text{data}}(x)$$

---

## Network Roles

### 1. Generator $g_\theta(z)$

The generator maps particle-level features $z$ to positive weights:

$$g_\theta: \mathbb{R}^{\dim(z)} \to \mathbb{R}^+$$

Parameterized as a multi-layer perceptron with a `softplus` ($\log(1 + e^u)$) activation on the output layer.

To preserve the overall Monte Carlo event yield, weights are normalized per batch:

$$w_i = \frac{g_\theta(z_i)}{\frac{1}{N_{\text{mc}}} \sum_{j=1}^{N_{\text{mc}}} g_\theta(z_j)}$$

### 2. Discriminator $d_\phi(x)$

The discriminator operates strictly at detector level:

$$d_\phi: \mathbb{R}^{\dim(x)} \to (0, 1)$$

Parameterized as an MLP with `sigmoid` activation, estimating the posterior probability that an event originated from data rather than simulation:

$$d_\phi(x) \approx P(y = 1 \mid x)$$

---

## Objective & Loss Functions

The training objective is the weighted binary cross-entropy:

$$\mathcal{L}(d_\phi, g_\theta) = -\mathbb{E}_{x \sim p_{\text{data}}} \lbrack \log d_\phi(x) \rbrack - \mathbb{E}_{(x, z) \sim p_{\text{sim}}} \lbrack w(z) \log(1 - d_\phi(x)) \rbrack$$

### The Alternating Optimization

1. **Discriminator Step**: With $g_\theta$ fixed, maximize $\mathcal{L}$ (minimize binary cross-entropy) with respect to $\phi$:
   $$\min_\phi -\left( \sum_{i \in \text{data}} \log d_\phi(x_i) + \sum_{j \in \text{sim}} w_j \log(1 - d_\phi(x_j)) \right)$$
2. **Generator Step**: With $d_\phi$ fixed, update $\theta$ to fool the discriminator:
   $$\min_\theta \sum_{j \in \text{sim}} w_j \log(1 - d_\phi(x_j))$$

A 5:1 update ratio (5 discriminator steps per generator step) is applied by default to ensure the discriminator accurately estimates the density ratio at each iteration.

---

## Theoretical Equilibrium

At global equilibrium:

- The reweighted simulation perfectly reproduces observed data: $p_{\text{reweighted}}(x) = p_{\text{data}}(x)$.
- The optimal discriminator is completely uninformative:
  $$d^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{reweighted}}(x)} = \frac{1}{2}$$
- The losses for both networks converge to:
  $$\mathcal{L}^* = -\log\left(\frac{1}{2}\right) = \log(2) \approx 0.69315$$
