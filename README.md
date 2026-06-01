# lau-optimal-transport-agents

Optimal transport theory applied to agent belief distributions.

Moving agent beliefs from one distribution to another has a minimum cost — the **Wasserstein distance**. This is the TRUE distance between beliefs (not KL divergence, which is asymmetric and can be infinite).

## Core Concepts

| Module | Description |
|---|---|
| `wasserstein` | W₁ and W₂ distances (earth mover's), cost matrices |
| `sinkhorn` | Entropically regularized OT via Sinkhorn-Knopp, O(n²) per iteration |
| `kantorovich` | Kantorovich duality: OT as LP with dual formulation |
| `brenier` | Brenier's theorem: optimal map is gradient of convex function |
| `mccann` | McCann interpolation: geodesics in Wasserstein space |
| `barycenter` | Wasserstein barycenter: Fréchet mean of multiple beliefs |
| `multimarginal` | Multi-marginal OT: transport between K distributions |
| `gradient_flow` | Wasserstein gradient flow (connects to Fokker-Planck) |
| `convergence` | Belief convergence rates during learning |
| `agent` | `AgentBelief` and `BeliefHistory` for tracking agent learning |

## Quick Start

```rust
use lau_optimal_transport_agents::agent::{AgentBelief, BeliefHistory};

// Create agent beliefs
let b0 = AgentBelief::new(vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 2.0], "prior");
let b1 = AgentBelief::new(vec![0.2, 0.5, 0.3], vec![0.0, 1.0, 2.0], "posterior");

// Measure belief change in the natural metric
let w1 = b0.w1_to(&b1); // Wasserstein-1
let w2 = b0.w2_to(&b1); // Wasserstein-2

// Interpolate between beliefs (McCann geodesic)
let curve = b0.interpolate_to(&b1);
let midpoint = &curve.evaluate(0.5);
```

## Dependencies

- `nalgebra` — Linear algebra
- `serde` / `serde_json` — Serialization

## Tests

117 tests covering all modules.

```bash
cargo test
```

## License

MIT
