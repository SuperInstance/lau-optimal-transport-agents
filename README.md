# lau-optimal-transport-agents

Optimal transport theory — Wasserstein distances, Sinkhorn algorithm, Monge-Kantorovich, earth mover's distance — for measuring the true distance between agent populations.

## Types

- **`DiscreteMeasure`** — probability measure on a discrete space with CDF/quantile support
- **`CostMatrix`** — ground cost c(xᵢ, yⱼ) (Euclidean, squared, or custom)
- **`TransportPlan`** — optimal coupling γ ∈ Π(μ, ν) with marginal validation
- **`WassersteinDistance`** — W₁ (CDF), W₂ (quantile), Wₚ, and earth mover's (LP)
- **`SinkhornAlgorithm`** — entropic regularization with log-domain stabilization
- **`MongeMap` / `BrenierMap`** — deterministic transport maps (monotone rearrangement in 1D)
- **`WassersteinBarycenter`** — Fréchet mean in Wasserstein space via quantile averaging
- **`JKOFlow`** — Jordan-Kinderlehrer-Otto gradient flow simulation
- **`AgentDistribution` / `AgentPoint`** — agents as probability distributions
- **`DenseMatrix`** — row-major dense matrix with standard linear algebra

## Theorems Verified

1. W₂ in 1D: quantile formula
2. W₁ metric: non-negative, symmetric, triangle inequality
3. Sinkhorn converges to true OT as ε → 0
4. Brenier map is monotone (gradient of convex potential in 1D)
5. Wasserstein barycenter of two Gaussians is Gaussian
6. Wₚ(μ, μ) = 0
7. Wₚ(μ, ν) ≥ 0 with equality iff μ = ν
8. Earth mover's with uniform weights = assignment problem
9. JKO flow of entropy functional spreads distribution
10. Monge map cost ≥ Kantorovich cost (relaxation)
11. Transport plan marginals match source and target
12. Agent distribution W₂ distance is symmetric

## License

MIT
