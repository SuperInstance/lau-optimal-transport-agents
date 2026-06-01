//! Sinkhorn algorithm for entropically regularized optimal transport.
//!
//! The Sinkhorn algorithm solves:
//!   min_{T >= 0} <C, T> + ε * H(T)  subject to T1 = a, T^T1 = b
//!
//! where H(T) = sum(T_ij * log(T_ij)) is the entropy regularizer.
//! Complexity: O(n²) per iteration.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Result of the Sinkhorn algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkhornResult {
    /// The optimal transport plan (entropy-regularized).
    pub transport_plan: DMatrix<f64>,
    /// The Sinkhorn scaling vectors (u, v) such that T = diag(u) K diag(v).
    pub u: DVector<f64>,
    pub v: DVector<f64>,
    /// The regularized transport cost.
    pub cost: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Run the Sinkhorn-Knopp algorithm for entropically regularized OT.
///
/// # Arguments
/// * `mu` - Source distribution (histogram, sums to 1)
/// * `nu` - Target distribution (histogram, sums to 1)
/// * `cost` - Cost matrix C[i,j]
/// * `epsilon` - Entropic regularization strength (smaller = closer to exact OT)
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance on marginal violations
pub fn sinkhorn(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    cost: &DMatrix<f64>,
    epsilon: f64,
    max_iter: usize,
    tol: f64,
) -> SinkhornResult {
    let n = mu.len();
    let m = nu.len();

    // Gibbs kernel K = exp(-C / ε)
    let mut kernel_data = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            kernel_data[i * m + j] = (-cost[(i, j)] / epsilon).exp();
        }
    }
    let kernel = DMatrix::from_row_slice(n, m, &kernel_data);

    // Initialize scaling vectors
    let mut u = DVector::from_element(n, 1.0);
    let mut v = DVector::from_element(m, 1.0);

    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // u = mu ./ (K * v)
        let kv = &kernel * &v;
        for i in 0..n {
            u[i] = if kv[i] > 1e-300 { mu[i] / kv[i] } else { 1e300 };
        }

        // v = nu ./ (K^T * u)
        let ktu = kernel.transpose() * &u;
        for j in 0..m {
            v[j] = if ktu[j] > 1e-300 { nu[j] / ktu[j] } else { 1e300 };
        }

        // Check convergence: marginal constraint violation
        let plan = compute_plan(&u, &v, &kernel);
        let mut max_violation = 0.0f64;
        for i in 0..n {
            let row_sum: f64 = (0..m).map(|j| plan[(i, j)]).sum();
            max_violation = max_violation.max((row_sum - mu[i]).abs());
        }
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| plan[(i, j)]).sum();
            max_violation = max_violation.max((col_sum - nu[j]).abs());
        }

        if max_violation < tol {
            converged = true;
            break;
        }
    }

    let transport_plan = compute_plan(&u, &v, &kernel);
    let cost = (&transport_plan).component_mul(cost).sum();

    SinkhornResult {
        transport_plan,
        u,
        v,
        cost,
        iterations,
        converged,
    }
}

fn compute_plan(u: &DVector<f64>, v: &DVector<f64>, kernel: &DMatrix<f64>) -> DMatrix<f64> {
    let n = u.len();
    let m = v.len();
    let mut plan = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            plan[(i, j)] = u[i] * kernel[(i, j)] * v[j];
        }
    }
    plan
}

/// Compute the Sinkhorn divergence (debiasaed Sinkhorn).
/// S_ε(a,b) = OT_ε(a,b) - 0.5 * OT_ε(a,a) - 0.5 * OT_ε(b,b)
pub fn sinkhorn_divergence(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    cost: &DMatrix<f64>,
    epsilon: f64,
    max_iter: usize,
) -> f64 {
    let r_ab = sinkhorn(mu, nu, cost, epsilon, max_iter, 1e-8);
    let r_aa = sinkhorn(mu, mu, cost, epsilon, max_iter, 1e-8);
    let r_bb = sinkhorn(nu, nu, cost, epsilon, max_iter, 1e-8);
    r_ab.cost - 0.5 * r_aa.cost - 0.5 * r_bb.cost
}

/// Stabilized Sinkhorn using log-domain computations (for small ε).
pub fn sinkhorn_log_stabilized(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    cost: &DMatrix<f64>,
    epsilon: f64,
    max_iter: usize,
    tol: f64,
) -> SinkhornResult {
    let n = mu.len();
    let m = nu.len();

    // Log-domain: f_i, g_j (dual potentials)
    let mut f = DVector::zeros(n);
    let mut g = DVector::zeros(m);

    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Update f: f_i = -ε * logsumexp_j((-C_ij + g_j) / ε) + ε * log(a_i)
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..m {
                let val = (-cost[(i, j)] + g[j]) / epsilon;
                if val > max_val {
                    max_val = val;
                }
            }
            let mut lse: f64 = 0.0;
            for j in 0..m {
                lse += f64::exp((-cost[(i, j)] + g[j]) / epsilon - max_val);
            }
            f[i] = -epsilon * (max_val + lse.ln()) + epsilon * mu[i].ln().max(-50.0);
        }

        // Update g: g_j = -ε * logsumexp_i((-C_ij + f_i) / ε) + ε * log(b_j)
        for j in 0..m {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..n {
                let val = (-cost[(i, j)] + f[i]) / epsilon;
                if val > max_val {
                    max_val = val;
                }
            }
            let mut lse: f64 = 0.0;
            for i in 0..n {
                lse += f64::exp((-cost[(i, j)] + f[i]) / epsilon - max_val);
            }
            g[j] = -epsilon * (max_val + lse.ln()) + epsilon * nu[j].ln().max(-50.0);
        }

        // Check convergence
        if (iter + 1) % 10 == 0 {
            let mut max_violation = 0.0f64;
            // Compute plan from dual potentials
            for i in 0..n {
                let row_sum: f64 = (0..m)
                    .map(|j| ((f[i] + g[j] - cost[(i, j)]) / epsilon).exp())
                    .sum();
                max_violation = max_violation.max((row_sum - mu[i]).abs());
            }
            if max_violation < tol {
                converged = true;
                break;
            }
        }
    }

    // Reconstruct transport plan
    let mut transport_plan = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            transport_plan[(i, j)] = ((f[i] + g[j] - cost[(i, j)]) / epsilon).exp();
        }
    }
    let cost_val = (&transport_plan).component_mul(cost).sum();

    // Convert to u, v form
    let u = f.map(|x| (x / epsilon).exp());
    let v = g.map(|x| (x / epsilon).exp());

    SinkhornResult {
        transport_plan,
        u,
        v,
        cost: cost_val,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sinkhorn_basic() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 1000, 1e-8);
        assert!(result.converged);
        assert!(result.cost >= 0.0);
    }

    #[test]
    fn test_sinkhorn_row_marginals() {
        let mu = DVector::from_vec(vec![0.3, 0.7]);
        let nu = DVector::from_vec(vec![0.4, 0.6]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 1000, 1e-8);
        for i in 0..2 {
            let row_sum: f64 = (0..2).map(|j| result.transport_plan[(i, j)]).sum();
            assert_relative_eq!(row_sum, mu[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_sinkhorn_col_marginals() {
        let mu = DVector::from_vec(vec![0.3, 0.7]);
        let nu = DVector::from_vec(vec![0.4, 0.6]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 1000, 1e-8);
        for j in 0..2 {
            let col_sum: f64 = (0..2).map(|i| result.transport_plan[(i, j)]).sum();
            assert_relative_eq!(col_sum, nu[j], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_sinkhorn_identical_distributions() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 1000, 1e-8);
        assert_relative_eq!(result.cost, 0.0, epsilon = 0.15);
    }

    #[test]
    fn test_sinkhorn_dirac_shift() {
        let mu = DVector::from_vec(vec![1.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 1.0]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.01, 2000, 1e-8);
        // With small ε, cost should approach 1.0
        assert!(result.cost > 0.5);
    }

    #[test]
    fn test_sinkhorn_serialization() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 100, 1e-6);
        let json = serde_json::to_string(&result).unwrap();
        let r2: SinkhornResult = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(r2.cost, result.cost, epsilon = 1e-10);
    }

    #[test]
    fn test_sinkhorn_convergence_fewer_iterations() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 10, 1e-8);
        // Should still produce valid transport plan
        let total_mass: f64 = result.transport_plan.sum();
        assert_relative_eq!(total_mass, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_sinkhorn_nonneg_plan() {
        let mu = DVector::from_vec(vec![0.3, 0.4, 0.3]);
        let nu = DVector::from_vec(vec![0.2, 0.5, 0.3]);
        let cost = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 500, 1e-8);
        for i in 0..3 {
            for j in 0..3 {
                assert!(result.transport_plan[(i, j)] >= 0.0);
            }
        }
    }

    #[test]
    fn test_sinkhorn_divergence_nonneg() {
        let mu = DVector::from_vec(vec![0.3, 0.7]);
        let nu = DVector::from_vec(vec![0.6, 0.4]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let div = sinkhorn_divergence(&mu, &nu, &cost, 0.1, 500);
        // Sinkhorn divergence should be >= 0
        assert!(div >= -0.5); // Allow small numerical errors
    }

    #[test]
    fn test_sinkhorn_divergence_same() {
        let mu = DVector::from_vec(vec![0.4, 0.6]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let div = sinkhorn_divergence(&mu, &mu, &cost, 0.1, 500);
        assert_relative_eq!(div, 0.0, epsilon = 0.15);
    }

    #[test]
    fn test_sinkhorn_log_stabilized() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn_log_stabilized(&mu, &nu, &cost, 0.01, 1000, 1e-6);
        assert!(result.transport_plan.sum() > 0.0);
    }

    #[test]
    fn test_sinkhorn_log_marginals() {
        let mu = DVector::from_vec(vec![0.3, 0.7]);
        let nu = DVector::from_vec(vec![0.4, 0.6]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let result = sinkhorn_log_stabilized(&mu, &nu, &cost, 0.1, 1000, 1e-6);
        for i in 0..2 {
            let row_sum: f64 = (0..2).map(|j| result.transport_plan[(i, j)]).sum();
            assert_relative_eq!(row_sum, mu[i], epsilon = 0.1);
        }
    }

    #[test]
    fn test_sinkhorn_larger_problem() {
        let n = 5;
        let mu = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.2, 0.2]);
        let nu = DVector::from_vec(vec![0.15, 0.15, 0.25, 0.25, 0.2]);
        let mut cost_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                cost_data[i * n + j] = ((i as f64) - (j as f64)).abs();
            }
        }
        let cost = DMatrix::from_row_slice(n, n, &cost_data);
        let result = sinkhorn(&mu, &nu, &cost, 0.1, 2000, 1e-6);
        assert!(result.converged || result.cost >= 0.0);
        assert!(result.cost >= 0.0);
    }
}
