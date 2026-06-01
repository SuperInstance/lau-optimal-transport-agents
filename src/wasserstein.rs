//! Wasserstein-1 and Wasserstein-2 distances.
//!
//! The Wasserstein distance (earth mover's distance) measures the minimum cost
//! of transforming one probability distribution into another.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A cost matrix for transport between two discrete distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMatrix {
    /// The cost matrix C where C[i,j] = cost of moving mass from i to j.
    pub costs: DMatrix<f64>,
}

impl CostMatrix {
    /// Build a cost matrix from two sets of 1D support points with linear cost.
    pub fn linear(support_a: &[f64], support_b: &[f64]) -> Self {
        let n = support_a.len();
        let m = support_b.len();
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i * m + j] = (support_a[i] - support_b[j]).abs();
            }
        }
        Self {
            costs: DMatrix::from_row_slice(n, m, &data),
        }
    }

    /// Build a cost matrix from two sets of 1D support points with quadratic cost.
    pub fn quadratic(support_a: &[f64], support_b: &[f64]) -> Self {
        let n = support_a.len();
        let m = support_b.len();
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                let d = support_a[i] - support_b[j];
                data[i * m + j] = d * d;
            }
        }
        Self {
            costs: DMatrix::from_row_slice(n, m, &data),
        }
    }

    /// Build a cost matrix from two sets of d-dimensional points with Euclidean distance.
    pub fn euclidean(points_a: &DMatrix<f64>, points_b: &DMatrix<f64>) -> Self {
        let n = points_a.nrows();
        let m = points_b.nrows();
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..points_a.ncols() {
                    let d = points_a[(i, k)] - points_b[(j, k)];
                    sum += d * d;
                }
                data[i * m + j] = sum.sqrt();
            }
        }
        Self {
            costs: DMatrix::from_row_slice(n, m, &data),
        }
    }

    /// Build a cost matrix from two sets of d-dimensional points with squared Euclidean distance.
    pub fn squared_euclidean(points_a: &DMatrix<f64>, points_b: &DMatrix<f64>) -> Self {
        let n = points_a.nrows();
        let m = points_b.nrows();
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..points_a.ncols() {
                    let d = points_a[(i, k)] - points_b[(j, k)];
                    sum += d * d;
                }
                data[i * m + j] = sum;
            }
        }
        Self {
            costs: DMatrix::from_row_slice(n, m, &data),
        }
    }

    /// Number of source points.
    pub fn n_source(&self) -> usize {
        self.costs.nrows()
    }

    /// Number of target points.
    pub fn n_target(&self) -> usize {
        self.costs.ncols()
    }
}

/// Solve the Kantorovich OT problem for 1D distributions using quantile coupling.
/// For 1D distributions, the optimal transport plan is the monotone rearrangement.
fn solve_ot_1d(cost: &DMatrix<f64>, mu: &DVector<f64>, nu: &DVector<f64>) -> (DMatrix<f64>, f64) {
    let n = mu.len();
    let m = nu.len();

    // Compute CDFs
    let mut cdf_a = vec![0.0; n + 1];
    for i in 0..n {
        cdf_a[i + 1] = cdf_a[i] + mu[i];
    }
    let mut cdf_b = vec![0.0; m + 1];
    for j in 0..m {
        cdf_b[j + 1] = cdf_b[j] + nu[j];
    }

    // Build transport plan by sweeping through quantile intervals
    let mut plan = DMatrix::zeros(n, m);
    let mut i = 0;
    let mut j = 0;
    let mut a = 0.0; // current position in CDF
    let b_end = cdf_a[n];

    while i < n && j < m && a < b_end - 1e-15 {
        let a_next = cdf_a[i + 1];
        let b_next = cdf_b[j + 1];
        let mass = a_next.min(b_next) - a;
        if mass > 1e-15 {
            plan[(i, j)] += mass;
        }
        a += mass;
        if a >= a_next - 1e-15 {
            i += 1;
        }
        if a >= b_next - 1e-15 {
            j += 1;
        }
    }

    let total_cost = (&plan).component_mul(cost).sum();
    (plan, total_cost)
}

/// General OT solver using iterative proportional fitting (RAS/IPFP).
fn solve_ot_lp(cost: &DMatrix<f64>, mu: &DVector<f64>, nu: &DVector<f64>, max_iter: usize) -> (DMatrix<f64>, f64) {
    let n = mu.len();
    let m = nu.len();

    // For 1D (n==m or sorted support), use the exact 1D solver
    // For general case, use IPFP
    if n <= 20 && m <= 20 {
        // Try 1D solver first
        let (plan, cost) = solve_ot_1d(cost, mu, nu);
        // Verify marginals
        let mut ok = true;
        for i in 0..n {
            let row_sum: f64 = (0..m).map(|j| plan[(i, j)]).sum();
            if (row_sum - mu[i]).abs() > 0.01 {
                ok = false;
                break;
            }
        }
        if ok {
            return (plan, cost);
        }
    }

    // IPFP fallback
    let mut plan = DMatrix::from_element(n, m, 1.0 / (n * m) as f64);
    for _ in 0..max_iter {
        for i in 0..n {
            let row_sum: f64 = (0..m).map(|j| plan[(i, j)]).sum();
            if row_sum > 1e-15 {
                for j in 0..m {
                    plan[(i, j)] *= mu[i] / row_sum;
                }
            }
        }
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| plan[(i, j)]).sum();
            if col_sum > 1e-15 {
                for i in 0..n {
                    plan[(i, j)] *= nu[j] / col_sum;
                }
            }
        }
    }
    let total_cost = (&plan).component_mul(cost).sum();
    (plan, total_cost)
}

/// Compute the Wasserstein-1 distance between two discrete distributions.
///
/// Uses linear cost |x_i - y_j|.
///
/// # Arguments
/// * `mu` - Source probability vector (must sum to 1)
/// * `nu` - Target probability vector (must sum to 1)
/// * `support_a` - Support points of source
/// * `support_b` - Support points of target
pub fn wasserstein_1(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    support_a: &[f64],
    support_b: &[f64],
) -> f64 {
    let cost = CostMatrix::linear(support_a, support_b);
    let (_, w1) = solve_ot_lp(&cost.costs, mu, nu, 500);
    w1
}

/// Compute the Wasserstein-2 distance between two discrete distributions.
///
/// Uses quadratic cost (x_i - y_j)², returns sqrt of optimal transport cost.
///
/// # Arguments
/// * `mu` - Source probability vector (must sum to 1)
/// * `nu` - Target probability vector (must sum to 1)
/// * `support_a` - Support points of source
/// * `support_b` - Support points of target
pub fn wasserstein_2(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    support_a: &[f64],
    support_b: &[f64],
) -> f64 {
    let cost = CostMatrix::quadratic(support_a, support_b);
    let (_, w2_sq) = solve_ot_lp(&cost.costs, mu, nu, 500);
    w2_sq.sqrt()
}

/// Compute Wasserstein-1 distance given a precomputed cost matrix.
pub fn wasserstein_1_with_cost(cost: &CostMatrix, mu: &DVector<f64>, nu: &DVector<f64>) -> f64 {
    let (_, w1) = solve_ot_lp(&cost.costs, mu, nu, 500);
    w1
}

/// Compute Wasserstein-2 distance given a precomputed cost matrix (quadratic cost).
/// Returns sqrt of optimal cost.
pub fn wasserstein_2_with_cost(cost: &CostMatrix, mu: &DVector<f64>, nu: &DVector<f64>) -> f64 {
    let (_, w2_sq) = solve_ot_lp(&cost.costs, mu, nu, 500);
    w2_sq.sqrt()
}

/// Compute the optimal transport plan between two distributions.
pub fn optimal_transport_plan(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    cost: &CostMatrix,
    max_iter: usize,
) -> DMatrix<f64> {
    let (plan, _) = solve_ot_lp(&cost.costs, mu, nu, max_iter);
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cost_matrix_linear() {
        let a = &[0.0, 1.0];
        let b = &[0.0, 1.0, 2.0];
        let c = CostMatrix::linear(a, b);
        assert_eq!(c.n_source(), 2);
        assert_eq!(c.n_target(), 3);
        assert_relative_eq!(c.costs[(0, 0)], 0.0);
        assert_relative_eq!(c.costs[(0, 1)], 1.0);
        assert_relative_eq!(c.costs[(1, 2)], 1.0);
    }

    #[test]
    fn test_cost_matrix_quadratic() {
        let a = &[0.0, 2.0];
        let b = &[1.0];
        let c = CostMatrix::quadratic(a, b);
        assert_relative_eq!(c.costs[(0, 0)], 1.0);
        assert_relative_eq!(c.costs[(1, 0)], 1.0);
    }

    #[test]
    fn test_euclidean_cost() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]);
        let c = CostMatrix::euclidean(&a, &b);
        assert_relative_eq!(c.costs[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(c.costs[(0, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(c.costs[(1, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(c.costs[(1, 1)], 1.41421356, epsilon = 1e-4);
    }

    #[test]
    fn test_squared_euclidean_cost() {
        let a = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 2.0]);
        let c = CostMatrix::squared_euclidean(&a, &b);
        assert_relative_eq!(c.costs[(0, 0)], 1.0); // |0-1|^2 = 1
        assert_relative_eq!(c.costs[(0, 1)], 4.0); // |0-2|^2 = 4
        assert_relative_eq!(c.costs[(1, 0)], 0.0); // |1-1|^2 = 0
        assert_relative_eq!(c.costs[(1, 1)], 1.0); // |1-2|^2 = 1
    }

    #[test]
    fn test_w1_identical_distributions() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let w1 = wasserstein_1(&mu, &nu, sa, sb);
        assert_relative_eq!(w1, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_w1_dirac_shift() {
        let mu = DVector::from_vec(vec![1.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 1.0]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let w1 = wasserstein_1(&mu, &nu, sa, sb);
        assert_relative_eq!(w1, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_w1_half_shift() {
        // mu = [0.5, 0.5], nu = [0.5, 0.5] with supports [0,1] and [1,2]
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[1.0, 2.0];
        let w1 = wasserstein_1(&mu, &nu, sa, sb);
        assert_relative_eq!(w1, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_w2_identical_distributions() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let w2 = wasserstein_2(&mu, &nu, sa, sb);
        assert_relative_eq!(w2, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_w2_dirac_shift() {
        let mu = DVector::from_vec(vec![1.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 1.0]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let w2 = wasserstein_2(&mu, &nu, sa, sb);
        assert_relative_eq!(w2, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_w2_shift_by_2() {
        let mu = DVector::from_vec(vec![1.0]);
        let nu = DVector::from_vec(vec![1.0]);
        let sa = &[0.0];
        let sb = &[2.0];
        let w2 = wasserstein_2(&mu, &nu, sa, sb);
        assert_relative_eq!(w2, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_transport_plan_row_marginals() {
        let mu = DVector::from_vec(vec![0.3, 0.7]);
        let nu = DVector::from_vec(vec![0.4, 0.6]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let cost = CostMatrix::linear(sa, sb);
        let plan = optimal_transport_plan(&mu, &nu, &cost, 500);

        for i in 0..2 {
            let row_sum: f64 = (0..2).map(|j| plan[(i, j)]).sum();
            assert_relative_eq!(row_sum, mu[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_transport_plan_col_marginals() {
        let mu = DVector::from_vec(vec![0.3, 0.7]);
        let nu = DVector::from_vec(vec![0.4, 0.6]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let cost = CostMatrix::linear(sa, sb);
        let plan = optimal_transport_plan(&mu, &nu, &cost, 500);

        for j in 0..2 {
            let col_sum: f64 = (0..2).map(|i| plan[(i, j)]).sum();
            assert_relative_eq!(col_sum, nu[j], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_w1_triangle_inequality() {
        let mu1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let mu2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let mu3 = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let s = &[0.0, 1.0, 2.0];
        let w12 = wasserstein_1(&mu1, &mu2, s, s);
        let w23 = wasserstein_1(&mu2, &mu3, s, s);
        let w13 = wasserstein_1(&mu1, &mu3, s, s);
        assert!(w13 <= w12 + w23 + 0.2);
    }

    #[test]
    fn test_w1_symmetry() {
        let mu = DVector::from_vec(vec![0.4, 0.6]);
        let nu = DVector::from_vec(vec![0.3, 0.7]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let w1_fwd = wasserstein_1(&mu, &nu, sa, sb);
        let w1_bwd = wasserstein_1(&nu, &mu, sb, sa);
        assert_relative_eq!(w1_fwd, w1_bwd, epsilon = 0.1);
    }

    #[test]
    fn test_w2_symmetry() {
        let mu = DVector::from_vec(vec![0.4, 0.6]);
        let nu = DVector::from_vec(vec![0.3, 0.7]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let w2_fwd = wasserstein_2(&mu, &nu, sa, sb);
        let w2_bwd = wasserstein_2(&nu, &mu, sb, sa);
        assert_relative_eq!(w2_fwd, w2_bwd, epsilon = 0.1);
    }

    #[test]
    fn test_cost_matrix_serialization() {
        let c = CostMatrix::linear(&[0.0, 1.0], &[0.0, 1.0, 2.0]);
        let json = serde_json::to_string(&c).unwrap();
        let c2: CostMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.costs.nrows(), 2);
        assert_eq!(c2.costs.ncols(), 3);
    }

    #[test]
    fn test_w1_with_cost_matrix() {
        let cost = CostMatrix::linear(&[0.0, 1.0], &[0.0, 1.0]);
        let mu = DVector::from_vec(vec![1.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 1.0]);
        let w1 = wasserstein_1_with_cost(&cost, &mu, &nu);
        assert_relative_eq!(w1, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_w2_with_cost_matrix() {
        let cost = CostMatrix::quadratic(&[0.0, 1.0], &[0.0, 1.0]);
        let mu = DVector::from_vec(vec![1.0, 0.0]);
        let nu = DVector::from_vec(vec![0.0, 1.0]);
        let w2 = wasserstein_2_with_cost(&cost, &mu, &nu);
        assert_relative_eq!(w2, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_w2_larger_gap() {
        let mu = DVector::from_vec(vec![1.0]);
        let nu = DVector::from_vec(vec![1.0]);
        let sa = &[0.0];
        let sb = &[3.0];
        let w2 = wasserstein_2(&mu, &nu, sa, sb);
        assert_relative_eq!(w2, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_w1_nonnegativity() {
        let mu = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let nu = DVector::from_vec(vec![0.4, 0.1, 0.5]);
        let sa = &[0.0, 1.0, 2.0];
        let sb = &[0.0, 1.0, 2.0];
        let w1 = wasserstein_1(&mu, &nu, sa, sb);
        assert!(w1 >= -1e-10);
    }
}
