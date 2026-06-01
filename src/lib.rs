//! # lau-optimal-transport-agents
//!
//! Optimal transport theory applied to agent belief distributions.
//!
//! Moving agent beliefs from one distribution to another has a minimum cost —
//! the Wasserstein distance. This is the TRUE distance between beliefs
//! (not KL divergence, which is asymmetric and can be infinite).
//!
//! ## Core Concepts
//!
//! - **Wasserstein-1 / Wasserstein-2**: Earth mover's distances with linear/quadratic costs
//! - **Sinkhorn algorithm**: Entropically regularized OT, O(n²) per iteration
//! - **Kantorovich duality**: OT as a linear program with dual formulation
//! - **Brenier's theorem**: Optimal map between continuous distributions
//! - **McCann interpolation**: Geodesic in Wasserstein space
//! - **Wasserstein barycenter**: Fréchet mean of multiple belief distributions
//! - **Multi-marginal OT**: Transport between K distributions simultaneously
//! - **Wasserstein gradient flow**: Steepest descent in distribution space

pub mod wasserstein;
pub mod sinkhorn;
pub mod kantorovich;
pub mod brenier;
pub mod mccann;
pub mod barycenter;
pub mod multimarginal;
pub mod gradient_flow;
pub mod convergence;
pub mod agent;

pub use wasserstein::{wasserstein_1, wasserstein_2, CostMatrix};
pub use sinkhorn::{sinkhorn, SinkhornResult};
pub use kantorovich::{kantorovich_dual, DualVariables};
pub use brenier::{brenier_map, BrenierMap};
pub use mccann::{mccann_interpolation, McCannCurve};
pub use barycenter::{wasserstein_barycenter, BarycenterResult};
pub use multimarginal::{multimarginal_ot, MultiMarginalResult};
pub use gradient_flow::{wasserstein_gradient_flow, GradientFlowResult};
pub use convergence::{belief_shift, ConvergenceResult};
pub use agent::{AgentBelief, BeliefHistory};
