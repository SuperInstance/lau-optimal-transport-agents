#![allow(clippy::needless_range_loop)]
use crate::{CostMatrix, DenseMatrix, DiscreteMeasure, TransportPlan};

/// Entropic regularization for fast optimal transport (Sinkhorn algorithm).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SinkhornAlgorithm {
    pub regularization: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    #[serde(skip)]
    pub convergence_history: Vec<f64>,
}

impl SinkhornAlgorithm {
    pub fn new(regularization: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            regularization,
            max_iterations,
            tolerance: tolerance.max(1e-12),
            convergence_history: Vec::new(),
        }
    }

    /// Compute approximate OT via Sinkhorn iterations (log-domain stabilized).
    /// Returns (transport_plan, regularized_distance W̃_ε).
    pub fn compute(
        &mut self,
        cost: &CostMatrix,
        mu: &DiscreteMeasure,
        nu: &DiscreteMeasure,
    ) -> (TransportPlan, f64) {
        let n = mu.n();
        let m = nu.n();
        let eps = self.regularization;

        // Use log-domain Sinkhorn for numerical stability
        // f[i] and g[j] are log-domain dual variables
        let mut f = vec![0.0; n];
        let mut g = vec![0.0; m];

        self.convergence_history.clear();

        for iter in 0..self.max_iterations {
            // Update f: f[i] = -eps * log( sum_j exp( (g[j] - C[i][j]) / eps ) ) + eps * log(a[i])
            // Using logsumexp for stability
            for i in 0..n {
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..m {
                    let val = (g[j] - cost.costs.get(i, j)) / eps;
                    if val > max_val {
                        max_val = val;
                    }
                }
                if max_val.is_finite() {
                    let lse: f64 = max_val + (0..m)
                        .map(|j| ((g[j] - cost.costs.get(i, j)) / eps - max_val).exp())
                        .sum::<f64>()
                        .ln();
                    f[i] = -eps * lse + eps * mu.weights[i].ln();
                }
            }

            // Update g: g[j] = -eps * log( sum_i exp( (f[i] - C[i][j]) / eps ) ) + eps * log(b[j])
            for j in 0..m {
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..n {
                    let val = (f[i] - cost.costs.get(i, j)) / eps;
                    if val > max_val {
                        max_val = val;
                    }
                }
                if max_val.is_finite() {
                    let lse: f64 = max_val + (0..n)
                        .map(|i| ((f[i] - cost.costs.get(i, j)) / eps - max_val).exp())
                        .sum::<f64>()
                        .ln();
                    g[j] = -eps * lse + eps * nu.weights[j].ln();
                }
            }

            // Compute transport plan: P[i][j] = exp( (f[i] + g[j] - C[i][j]) / eps )
            let mut plan = DenseMatrix::zeros(n, m);
            for i in 0..n {
                for j in 0..m {
                    let val = (f[i] + g[j] - cost.costs.get(i, j)) / eps;
                    plan.set(i, j, val.exp());
                }
            }

            // Check marginal error
            let mut max_err: f64 = 0.0;
            for i in 0..n {
                let err = (plan.row_sum(i) - mu.weights[i]).abs();
                max_err = max_err.max(err);
            }
            for j in 0..m {
                let err = (plan.col_sum(j) - nu.weights[j]).abs();
                max_err = max_err.max(err);
            }

            // Record dual variable norm for convergence history
            let dual_norm: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt();
            self.convergence_history.push(dual_norm);

            if max_err < self.tolerance {
                let total = cost.total_cost(&TransportPlan::new(
                    mu.clone(),
                    nu.clone(),
                    plan.clone(),
                ));
                return (TransportPlan::new(mu.clone(), nu.clone(), plan), total);
            }

            if iter == self.max_iterations - 1 {
                let total = cost.total_cost(&TransportPlan::new(
                    mu.clone(),
                    nu.clone(),
                    plan.clone(),
                ));
                return (TransportPlan::new(mu.clone(), nu.clone(), plan), total);
            }
        }

        let plan = DenseMatrix::zeros(n, m);
        (TransportPlan::new(mu.clone(), nu.clone(), plan), 0.0)
    }

    pub fn convergence_history(&self) -> Vec<f64> {
        self.convergence_history.clone()
    }
}
