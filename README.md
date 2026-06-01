# lau-optimal-transport-agents

> Move earth, not just numbers: Wasserstein distances, Sinkhorn, and Monge–Kantorovich optimal transport for agent populations.

## What This Does

This crate provides discrete optimal transport — computing the true geometric distance between probability distributions and agent populations. It implements Wasserstein distances (W₁, W₂, Wₚ), the Sinkhorn algorithm for entropically regularized transport, Monge maps and Brenier maps for deterministic couplings, Wasserstein barycenters for distributional averaging, and JKO gradient flows for distributional dynamics.

Use this when you need to compare, interpolate between, or optimize over probability distributions in a way that respects their geometric structure — not just their pointwise values.

## The Key Idea

The Wasserstein distance measures the minimum cost of transforming one distribution into another, where "cost" is how far you have to move probability mass. It's literally the "earth mover's distance": imagine two piles of dirt (distributions) and find the cheapest way to reshape one into the other. Unlike KL divergence or total variation, Wasserstein distance cares about *where* the mass is — two nearby Gaussians have small Wasserstein distance even if they have zero overlap.

## Install

```bash
cargo add lau-optimal-transport-agents
```

## Quick Start

```rust
use lau_optimal_transport_agents::*;

fn main() {
    // Two discrete distributions
    let mu = DiscreteMeasure::new(
        vec![0.0, 1.0, 2.0],     // support points
        vec![0.25, 0.5, 0.25],   // weights
    );
    let nu = DiscreteMeasure::new(
        vec![1.0, 2.0, 3.0],
        vec![0.25, 0.5, 0.25],
    );

    // W₁ (Earth Mover's Distance)
    let w1 = WassersteinDistance::w1(&mu, &nu);
    println!("W₁ = {:.4}", w1);

    // W₂ (quadratic cost)
    let w2 = WassersteinDistance::w2(&mu, &nu);
    println!("W₂ = {:.4}", w2);

    // Sinkhorn algorithm (entropic regularization)
    let cost = CostMatrix::from_euclidean_squared(&mu.support, &nu.support);
    let mut sinkhorn = SinkhornAlgorithm::new(0.1, 1000, 1e-8);
    let (plan, regularized_cost) = sinkhorn.compute(&cost, &mu, &nu);
    println!("Sinkhorn cost: {:.4}", regularized_cost);

    // Optimal transport plan for 1D (monotone coupling)
    let (cost_val, optimal_plan) = WassersteinDistance::earth_movers(&mu, &nu, &cost);
    println!("Optimal cost: {:.4}, valid marginals: {}",
        cost_val, optimal_plan.marginals_valid(1e-8));

    // Wasserstein barycenter of three measures
    let measures = vec![mu.clone(), nu.clone(), DiscreteMeasure::new(
        vec![0.5, 1.5, 2.5], vec![1.0/3.0; 3]
    )];
    let weights = vec![1.0/3.0; 3];
    let barycenter = WassersteinBarycenter::compute(&measures, &weights, 10, 100);
    println!("Barycenter mean: {:.4}", barycenter.mean());
}
```

## API Reference

### `DiscreteMeasure`
A probability measure on a discrete space with support points and weights.

```rust
// Create from support and weights
let mu = DiscreteMeasure::new(vec![0.0, 1.0, 2.0], vec![0.25, 0.5, 0.25]);

// Uniform distribution
let unif = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0, 3.0]);

// Dirac delta
let delta = DiscreteMeasure::dirac(42.0);

// Statistics
let mean = mu.mean();
let var = mu.variance();
let mass = mu.total_mass();
let is_prob = mu.is_probability();

// CDF and quantile
let cdf_val = mu.cdf(1.5);
let q75 = mu.quantile(0.75);

// Normalize to unit mass
mu.normalize();
```

### `DenseMatrix`
Row-major matrix for transport plans and cost matrices. Supports `get`, `set`, `row_sum`, `col_sum`, `multiply`, `transpose`. Implements `Display` for debugging.

### `CostMatrix`
Ground cost c(xᵢ, yⱼ) between source and target support points.

```rust
// Euclidean distance |x - y|
let cost = CostMatrix::from_euclidean(&source, &target);

// Squared Euclidean |x - y|²
let cost_sq = CostMatrix::from_euclidean_squared(&source, &target);

// Custom cost function
let custom = CostMatrix::from_custom(&source, &target, &|x, y| (x - y).abs().powi(3));
```

### `TransportPlan`
A coupling γ ∈ Π(μ, ν) — a matrix where row sums give source marginals and column sums give target marginals.

```rust
let plan = TransportPlan::new(mu, nu, plan_matrix);
let total = plan.total_cost(&cost);
let valid = plan.marginals_valid(1e-8);
let mass = plan.total_mass();
```

### `WassersteinDistance`
Wₚ distances between discrete measures. Uses the 1D closed-form via quantile functions (not LP solvers), so it's exact for 1D distributions.

| Method | Formula | Complexity |
|--------|---------|------------|
| `w1(μ, ν)` | ∫\|F_μ(x) − F_ν(x)\| dx | O(n log n) |
| `w2(μ, ν)` | (∫₀¹ (F_μ⁻¹(t) − F_ν⁻¹(t))² dt)^{1/2} | O(n log n) |
| `wp(μ, ν, p)` | General Wₚ via quantile formula | O(n log n) |
| `earth_movers(μ, ν, C)` | Exact optimal plan + cost for 1D | O(n log n) |

```rust
let w1 = WassersteinDistance::w1(&mu, &nu);
let w2 = WassersteinDistance::w2(&mu, &nu);
let w3 = WassersteinDistance::wp(&mu, &nu, 3.0);
let (cost, plan) = WassersteinDistance::earth_movers(&mu, &nu, &cost_matrix);
```

### `SinkhornAlgorithm`
Entropically regularized optimal transport via Sinkhorn iterations in the log domain (numerically stable).

The regularized problem: minimize ⟨γ, C⟩ + ε·KL(γ∥abᵀ) subject to marginals.

```rust
let mut sinkhorn = SinkhornAlgorithm::new(
    0.1,    // regularization ε
    1000,   // max iterations
    1e-8,   // convergence tolerance
);
let (plan, cost) = sinkhorn.compute(&cost_matrix, &mu, &nu);

// Check convergence history
let history = sinkhorn.convergence_history();
```

Parameters:
- **Small ε** → closer to exact OT, slower convergence
- **Large ε** → faster, more spread-out transport plan
- Uses log-domain stabilization (logsumexp) to avoid underflow

### `MongeMap`
A deterministic transport map T: X → Y, represented as a mapping from source indices to target indices.

```rust
let monge = MongeMap::new(vec![1, 0, 2]); // source 0→target 1, source 1→target 0, etc.
let valid = monge.is_monge_map(&mu, &nu, 1e-8);
let cost = monge.transport_cost(&cost_matrix);
let plan = monge.to_transport_plan(&mu);
```

### `BrenierMap`
The optimal transport map for quadratic cost — in 1D, this is the monotone rearrangement (sort both distributions and pair by quantile).

```rust
let brenier = BrenierMap::compute(&mu, &nu);
let is_mono = BrenierMap::is_monotone(&mu, &brenier);
```

By Brenier's theorem, for quadratic cost this is the unique gradient of a convex function that pushes μ forward to ν.

### `WassersteinBarycenter`
The Fréchet mean in Wasserstein space — the distribution that minimizes the weighted sum of squared W₂ distances.

```rust
let bary = WassersteinBarycenter::compute(
    &[mu1, mu2, mu3],  // input measures
    &[0.5, 0.3, 0.2],  // weights
    20,                 // support size of output
    100,                // iterations
);

// Verify a candidate is approximately a barycenter
let is_bary = WassersteinBarycenter::is_barycenter(&candidate, &measures, &weights, 0.01);
```

Computed via fixed-point iteration on quantile functions: the barycenter's quantile function is the weighted average of input quantile functions.

### `JKOFlow`
Jordan–Kinderlehrer–Otto gradient flow: discretize a gradient flow in Wasserstein space.

```
μ_{n+1} = argmin_ν { τ·F(ν) + W₂²(μₙ, ν)/2 }
```

```rust
let trajectory = JKOFlow::run(
    &initial_measure,
    &|mu| mu.variance(),  // functional to minimize
    0.1,   // time step τ
    50,    // number of steps
    100,   // grid size for optimization
);
```

### `AgentDistribution` / `AgentPoint`
Agents as a probability distribution over positions, with Wasserstein geometry.

```rust
let agents = AgentDistribution::new(vec![
    AgentPoint { id: "a1".into(), position: 0.0, capability: 1.0 },
    AgentPoint { id: "a2".into(), position: 1.0, capability: 0.8 },
]);

// Manipulate the distribution
agents.shift(1.0, 0.5);    // move all agents
agents.spread(0.1);         // increase variance
agents.concentrate(0.2);    // decrease variance

// Compare populations
let dist = agents.w_distance(&other_agents);

// Interpolate between populations
let mid = agents.barycenter_with(&other_agents, 0.5);
```

## How It Works

**Wasserstein distances** use the closed-form 1D quantile formula rather than linear programming. For Wₚ, the optimal coupling in 1D is always the monotone coupling (sort both distributions, pair by quantile). This makes computation O(n log n) instead of O(n³) for general LP-based transport.

**Sinkhorn** iteratively updates dual variables f, g in the log domain:
- Row update: fᵢ = −ε · logsumexp_j((gⱼ − Cᵢⱼ)/ε) + ε · log(aᵢ)
- Column update: gⱼ = −ε · logsumexp_i((fᵢ − Cᵢⱼ)/ε) + ε · log(bⱼ)

This converges to the solution of the entropy-regularized Kantorovich problem.

**Brenier maps** are computed by sorting both measures by support and pairing sources to targets by cumulative weight. This produces the monotone rearrangement, which is optimal for quadratic cost by the Brenier theorem.

**Barycenters** use the quantile representation: the barycenter's quantile function F⁻¹(t) = Σₖ wₖ Fₖ⁻¹(t) minimizes the Fréchet functional in Wasserstein space.

**JKO flows** discretize the time derivative and solve a proximal step at each iteration: minimize τ·F(ν) + W₂²(μₙ, ν)/2 over a grid of trial measures.

## The Math

### Kantorovich Problem

$$\min_{\gamma \in \Pi(\mu, \nu)} \sum_{i,j} c(x_i, y_j) \, \gamma_{ij}$$

where Π(μ, ν) is the set of couplings with marginals μ and ν.

### Wasserstein-p Distance (1D)

$$W_p(\mu, \nu) = \left(\int_0^1 |F_\mu^{-1}(t) - F_\nu^{-1}(t)|^p \, dt\right)^{1/p}$$

### Entropically Regularized Transport

$$\min_\gamma \langle \gamma, C \rangle + \varepsilon \, \text{KL}(\gamma \| a b^T)$$

Solved by Sinkhorn iterations in O(n²) per iteration.

### Brenier's Theorem

For quadratic cost c(x,y) = |x−y|² on ℝᵈ, there exists a unique (a.e.) convex function ψ such that T = ∇ψ is the optimal transport map pushing μ to ν.

### Wasserstein Barycenter

$$\bar{\mu} = \arg\min_\nu \sum_k w_k \, W_2^2(\nu, \mu_k)$$

In 1D, the solution has quantile function $F_\bar{\mu}^{-1}(t) = \sum_k w_k F_{\mu_k}^{-1}(t)$.

### JKO Scheme

$$\mu_{n+1} = \arg\min_\nu \left\{ \tau \cdot F(\nu) + \frac{1}{2} W_2^2(\mu_n, \nu) \right\}$$

This is the proximal operator in Wasserstein space, converging to gradient flows of F as τ → 0.

## License

MIT
