# lau-optimal-transport-agents

> Sinkhorn algorithm, Wasserstein distances, and barycenters for agent distribution alignment

## What This Does

This crate implements computational optimal transport — algorithms for optimally moving mass from one distribution to another. It provides Sinkhorn's algorithm for entropy-regularized transport, exact Wasserstein distance computation, barycenter computation (the "average" of distributions), and distributional operations (shift, spread, concentrate) for agent state manipulation.

## The Key Idea

Imagine you have two piles of sand (probability distributions) and want to reshape one into the other with minimum effort. That's the optimal transport problem. The Wasserstein distance measures this minimum effort. Unlike KL divergence (which is infinite if distributions have different supports), Wasserstein distance always gives a meaningful answer. The Sinkhorn algorithm solves this efficiently by adding a small amount of entropy, making the problem smooth and differentiable.

## Install

```toml
[dependencies]
lau-optimal-transport-agents = { git = "https://github.com/SuperInstance/lau-optimal-transport-agents" }
```

## Quick Start

```rust
use lau_optimal_transport_agents::*;
use nalgebra::DVector;

// Define two distributions as weighted point clouds
let source = AgentDistribution::from_weights(vec![
    AgentPoint::new(vec![0.0, 0.0], 0.5),
    AgentPoint::new(vec![1.0, 0.0], 0.5),
]);

let target = AgentDistribution::from_weights(vec![
    AgentPoint::new(vec![0.0, 1.0], 0.3),
    AgentPoint::new(vec![1.0, 1.0], 0.7),
]);

// Compute Wasserstein-1 distance
let w1 = source.wasserstein_distance(&target, 1);
println!("W1 distance: {:.4}", w1);

// Compute Wasserstein-2 distance
let w2 = source.wasserstein_distance(&target, 2);
println!("W2 distance: {:.4}", w2);

// Sinkhorn transport plan
let plan = SinkhornSolver::new(0.01, 100).solve(&source, &target);
println!("Transport plan:\n{:?}", plan.matrix);

// Barycenter of three distributions (weighted average in Wasserstein space)
let bary = WassersteinBarycenter::compute(
    &[&source, &target, &third],
    &[0.4, 0.3, 0.3],
);
```

## API Reference

### `AgentDistribution`

| Method | Description |
|--------|-------------|
| `from_weights(points)` | Create from weighted points. |
| `uniform(points)` | Uniform weights. |
| `wasserstein_distance(other, p)` | W_p distance. |
| `shift(offset)` | Translate distribution. |
| `spread(factor)` | Scale dispersion. |
| `concentrate(factor)` | Reduce dispersion toward mean. |
| `barycenter(other, weight)` | Interpolate in Wasserstein space. |

### `AgentPoint`

| Method | Description |
|--------|-------------|
| `new(coords, weight)` | A weighted point in R^n. |
| `coordinates()` | Get coordinates. |
| `weight()` | Get weight. |

### `SinkhornSolver`

| Method | Description |
|--------|-------------|
| `new(regularization, max_iter)` | Create with entropy parameter ε and iteration cap. |
| `solve(source, target)` | Returns `TransportPlan`. |

### `TransportPlan`

| Method | Description |
|--------|-------------|
| `matrix` | The transport matrix T (source × target). |
| `cost()` | Total transport cost Σ c_ij T_ij. |
| `marginals()` | Check row/column marginals match source/target. |

### `WassersteinBarycenter`

| Method | Description |
|--------|-------------|
| `compute(distributions, weights)` | Fixed-point iteration for Wasserstein barycenter. |

## How It Works

1. **Cost Matrix**: Compute pairwise ground costs c(xᵢ, yⱼ) = ‖xᵢ - yⱼ‖ᵖ.
2. **Sinkhorn**: Iteratively project onto row and column marginal constraints: K = exp(-C/ε), then alternate u = a / K·v, v = b / Kᵀ·u until convergence.
3. **Transport Plan**: T = diag(u) · K · diag(v). This is the optimal coupling.
4. **Wasserstein Distance**: W_p = (Σ T_ij c_ij)^{1/p}.
5. **Barycenter**: Iteratively update support weights to minimize Σ w_k W_p²(μ, μ_k).

## The Math

- **Monge-Kantorovich Problem**: min_T Σ c_ij T_ij subject to T1 = a, Tᵀ1 = b, T ≥ 0.
- **Sinkhorn Divergence**: S_ε(μ, ν) = OT_ε(μ,ν) - ½(OT_ε(μ,μ) + OT_ε(ν,ν)). Debiased version.
- **Wasserstein p-distance**: W_p(μ, ν) = (inf_π ∫ ‖x-y‖ᵖ dπ)^{1/p}

## Testing

75 tests covering:
- Sinkhorn convergence on known problems
- Transport plan marginal verification
- Wasserstein distance accuracy
- Barycenter fixed-point convergence
- Distribution operations (shift, spread, concentrate)
- Triangle inequality for W_p
- Edge cases: identical distributions, point masses

## License

MIT
