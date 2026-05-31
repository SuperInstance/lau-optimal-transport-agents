pub mod dense_matrix;
pub mod discrete_measure;
pub mod transport_plan;
pub mod wasserstein;
pub mod sinkhorn;
pub mod monge;
pub mod barycenter;
pub mod jko;
pub mod agent;

pub use dense_matrix::DenseMatrix;
pub use discrete_measure::DiscreteMeasure;
pub use transport_plan::{CostMatrix, TransportPlan};
pub use wasserstein::WassersteinDistance;
pub use sinkhorn::SinkhornAlgorithm;
pub use monge::{MongeMap, BrenierMap};
pub use barycenter::WassersteinBarycenter;
pub use jko::JKOFlow;
pub use agent::{AgentDistribution, AgentPoint};
