//! Kantorovich duality: OT as a linear program with dual formulation.
//!
//! The primal Kantorovich problem:
//!   min_{T >= 0} <C, T>  s.t. T1 = a, T^T1 = b
//!
//! The dual Kantorovich problem:
//!   max_{f, g} <f, a> + <g, b>  s.t. f_i + g_j <= C_ij
//!
//! Strong duality holds when a and b are probability measures.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Dual variables (Kantorovich potentials).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualVariables {
    /// Potential f for source distribution.
    pub f: DVector<f64>,
    /// Potential g for target distribution.
    pub g: DVector<f64>,
    /// Dual objective value: <f, a> + <g, b>.
    pub dual_objective: f64,
}

/// Solve the Kantorovich dual problem using coordinate ascent.
///
/// Iteratively updates f_i = min_j(C_ij - g_j) and g_j = min_i(C_ij - f_i).
/// These satisfy the dual constraint f_i + g_j <= C_ij with equality at optimality.
pub fn kantorovich_dual(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    cost: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> DualVariables {
    let n = mu.len();
    let m = nu.len();

    let mut f = DVector::zeros(n);
    let mut g = DVector::zeros(m);

    let mut prev_obj = f64::NEG_INFINITY;

    for _ in 0..max_iter {
        // Update f: f_i = min_j(C_ij - g_j)
        for i in 0..n {
            let mut min_val = f64::INFINITY;
            for j in 0..m {
                let val = cost[(i, j)] - g[j];
                if val < min_val {
                    min_val = val;
                }
            }
            f[i] = min_val;
        }

        // Update g: g_j = min_i(C_ij - f_i)
        for j in 0..m {
            let mut min_val = f64::INFINITY;
            for i in 0..n {
                let val = cost[(i, j)] - f[i];
                if val < min_val {
                    min_val = val;
                }
            }
            g[j] = min_val;
        }

        let obj = f.dot(mu) + g.dot(nu);

        if (obj - prev_obj).abs() < tol {
            break;
        }
        prev_obj = obj;
    }

    let dual_objective = f.dot(mu) + g.dot(nu);

    DualVariables {
        f,
        g,
        dual_objective,
    }
}

/// Check dual feasibility: f_i + g_j <= C_ij for all (i,j).
pub fn check_dual_feasibility(dual: &DualVariables, cost: &DMatrix<f64>) -> bool {
    let n = dual.f.len();
    let m = dual.g.len();
    for i in 0..n {
        for j in 0..m {
            if dual.f[i] + dual.g[j] > cost[(i, j)] + 1e-8 {
                return false;
            }
        }
    }
    true
}

/// Compute the complementary slackness: how many (i,j) pairs have f_i + g_j = C_ij.
/// At optimality, this should be at least max(n, m).
pub fn complementary_slackness_count(dual: &DualVariables, cost: &DMatrix<f64>, tol: f64) -> usize {
    let n = dual.f.len();
    let m = dual.g.len();
    let mut count = 0;
    for i in 0..n {
        for j in 0..m {
            if (dual.f[i] + dual.g[j] - cost[(i, j)]).abs() < tol {
                count += 1;
            }
        }
    }
    count
}

/// Derive the transport plan from dual variables.
/// T_ij > 0 only when f_i + g_j = C_ij (complementary slackness).
pub fn plan_from_dual(dual: &DualVariables, cost: &DMatrix<f64>, tol: f64) -> DMatrix<f64> {
    let n = dual.f.len();
    let m = dual.g.len();
    let mut plan = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            if (dual.f[i] + dual.g[j] - cost[(i, j)]).abs() < tol {
                plan[(i, j)] = 1.0;
            }
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dual_feasibility() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 1000, 1e-10);
        assert!(check_dual_feasibility(&dual, &cost));
    }

    #[test]
    fn test_dual_objective_bounded_by_primal() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 1000, 1e-10);
        // Dual should be <= 0 for identical distributions with cost matrix that has 0 diagonal
        assert!(dual.dual_objective <= 0.01);
    }

    #[test]
    fn test_dual_dirac() {
        let mu = DVector::from_vec(vec![1.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 1.0]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 1000, 1e-10);
        // Optimal dual: f = [0, -1], g = [-1, 0], objective = 0*1 + (-1)*0 + (-1)*0 + 0*1 = -1
        assert!(dual.dual_objective <= 0.0);
        assert!(check_dual_feasibility(&dual, &cost));
    }

    #[test]
    fn test_complementary_slackness() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 1000, 1e-10);
        let cs = complementary_slackness_count(&dual, &cost, 0.1);
        assert!(cs >= 2);
    }

    #[test]
    fn test_plan_from_dual() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 1000, 1e-10);
        let plan = plan_from_dual(&dual, &cost, 0.1);
        // Diagonal should have 1.0 entries
        assert!(plan[(0, 0)] > 0.0 || plan[(1, 1)] > 0.0);
    }

    #[test]
    fn test_dual_3x3() {
        let mu = DVector::from_vec(vec![0.3, 0.4, 0.3]);
        let nu = DVector::from_vec(vec![0.2, 0.5, 0.3]);
        let cost = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 1000, 1e-10);
        assert!(check_dual_feasibility(&dual, &cost));
    }

    #[test]
    fn test_dual_serialization() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 100, 1e-10);
        let json = serde_json::to_string(&dual).unwrap();
        let d2: DualVariables = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(d2.dual_objective, dual.dual_objective, epsilon = 1e-10);
    }

    #[test]
    fn test_strong_duality_identical() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let dual = kantorovich_dual(&mu, &nu, &cost, 1000, 1e-10);
        // For identical dists, optimal cost = 0, so dual = 0 (strong duality)
        assert_relative_eq!(dual.dual_objective, 0.0, epsilon = 0.01);
    }
}
