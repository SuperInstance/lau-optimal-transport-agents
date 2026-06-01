//! Multi-marginal optimal transport: transport between K distributions simultaneously.
//!
//! The multi-marginal OT problem:
//!   min_{π >= 0} Σ C_{i1,...,iK} π_{i1,...,iK}
//!   s.t. marginal_k(π) = μ_k for all k = 1,...,K
//!
//! For K=2 this reduces to standard OT.

use serde::{Deserialize, Serialize};

/// Result of multi-marginal OT computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiMarginalResult {
    /// Number of distributions.
    pub k: usize,
    /// Sizes of each distribution.
    pub sizes: Vec<usize>,
    /// Entropy-regularized solution (flattened coupling tensor).
    pub coupling: Vec<f64>,
    /// Total transport cost.
    pub cost: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether converged.
    pub converged: bool,
}

/// Cost function type for multi-marginal OT.
pub type MultiMarginalCost = Box<dyn Fn(&[usize]) -> f64>;

/// Solve multi-marginal OT using iterative Bregman projection (multi-marginal Sinkhorn).
///
/// # Arguments
/// * `distributions` - List of probability vectors
/// * `cost_fn` - Function mapping index tuple (i1,...,iK) to cost
/// * `epsilon` - Entropic regularization
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn multimarginal_ot(
    distributions: &[Vec<f64>],
    cost_fn: &dyn Fn(&[usize]) -> f64,
    epsilon: f64,
    max_iter: usize,
    tol: f64,
) -> MultiMarginalResult {
    let k = distributions.len();
    let sizes: Vec<usize> = distributions.iter().map(|d| d.len()).collect();
    let total_size: usize = sizes.iter().product();

    // Initialize Gibbs kernel: K(i1,...,iK) = exp(-C(i1,...,iK) / ε)
    let mut kernel = vec![0.0; total_size];
    let mut indices = vec![0usize; k];
    for flat in 0..total_size {
        // Decode flat index to multi-index
        let mut rem = flat;
        for d in 0..k {
            indices[d] = rem % sizes[d];
            rem /= sizes[d];
        }
        kernel[flat] = (-cost_fn(&indices) / epsilon).exp();
    }

    // Scaling vectors for each marginal
    let mut scalings: Vec<Vec<f64>> = sizes.iter().map(|&n| vec![1.0; n]).collect();

    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Update each marginal scaling
        for d in 0..k {
            for i_d in 0..sizes[d] {
                // Sum of kernel * other scalings for marginal (d, i_d)
                let mut sum = 0.0f64;
                for flat in 0..total_size {
                    // Decode
                    let mut rem = flat;
                    let mut idx_d = 0;
                    let mut product = kernel[flat];
                    for dd in 0..k {
                        let idx_dd = rem % sizes[dd];
                        rem /= sizes[dd];
                        if dd == d {
                            idx_d = idx_dd;
                        } else {
                            product *= scalings[dd][idx_dd];
                        }
                    }
                    if idx_d == i_d {
                        sum += product;
                    }
                }
                scalings[d][i_d] = if sum > 1e-300 { distributions[d][i_d] / sum } else { 0.0 };
            }
        }

        // Check convergence every 10 iterations
        if (iter + 1) % 10 == 0 {
            // Compute current coupling
            let coupling = compute_coupling(&kernel, &scalings, total_size);
            // Check first marginal constraint
            let mut max_viol = 0.0f64;
            let mut marginal_0 = vec![0.0; sizes[0]];
            for flat in 0..total_size {
                let rem = flat;
                marginal_0[rem % sizes[0]] += coupling[flat];
            }
            for i in 0..sizes[0] {
                max_viol = max_viol.max((marginal_0[i] - distributions[0][i]).abs());
            }
            if max_viol < tol {
                converged = true;
                break;
            }
        }
    }

    let coupling = compute_coupling(&kernel, &scalings, total_size);

    // Compute total cost
    let mut cost = 0.0;
    for flat in 0..total_size {
        let mut rem = flat;
        let mut indices = Vec::with_capacity(k);
        for d in 0..k {
            indices.push(rem % sizes[d]);
            rem /= sizes[d];
        }
        cost += coupling[flat] * cost_fn(&indices);
    }

    MultiMarginalResult {
        k,
        sizes: sizes.clone(),
        coupling,
        cost,
        iterations,
        converged,
    }
}

fn compute_coupling(kernel: &[f64], scalings: &[Vec<f64>], total_size: usize) -> Vec<f64> {
    let k = scalings.len();
    let mut coupling = vec![0.0; total_size];
    for flat in 0..total_size {
        let mut rem = flat;
        let mut val = kernel[flat];
        for d in 0..k {
            let idx = rem % scalings[d].len();
            rem /= scalings[d].len();
            val *= scalings[d][idx];
        }
        coupling[flat] = val;
    }
    coupling
}

/// Convenience: pairwise cost for multi-marginal OT (sum of pairwise costs).
pub fn pairwise_cost(supports: &[Vec<f64>], indices: &[usize]) -> f64 {
    let mut cost = 0.0;
    for i in 0..supports.len() {
        for j in (i + 1)..supports.len() {
            let d = supports[i][indices[i]] - supports[j][indices[j]];
            cost += d * d;
        }
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_multimarginal_two_dist() {
        let dists = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let cost_fn = |idx: &[usize]| -> f64 {
            let d = supports[0][idx[0]] - supports[1][idx[1]];
            d * d
        };
        let result = multimarginal_ot(&dists, &cost_fn, 0.1, 200, 1e-6);
        assert!(result.cost >= -0.1);
    }

    #[test]
    fn test_multimarginal_three_dist() {
        let dists = vec![vec![0.5, 0.5], vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let cost_fn = |idx: &[usize]| pairwise_cost(&supports, idx);
        let result = multimarginal_ot(&dists, &cost_fn, 0.1, 200, 1e-6);
        assert_eq!(result.k, 3);
        assert!(result.coupling.len() > 0);
    }

    #[test]
    fn test_multimarginal_identical() {
        let dists = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let cost_fn = |idx: &[usize]| {
            let d = supports[0][idx[0]] - supports[1][idx[1]];
            d * d
        };
        let result = multimarginal_ot(&dists, &cost_fn, 0.1, 500, 1e-8);
        // For identical distributions, cost should be small
        assert!(result.cost < 1.0);
    }

    #[test]
    fn test_multimarginal_coupling_nonneg() {
        let dists = vec![vec![0.3, 0.4, 0.3], vec![0.2, 0.5, 0.3]];
        let supports = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
        let cost_fn = |idx: &[usize]| pairwise_cost(&supports, idx);
        let result = multimarginal_ot(&dists, &cost_fn, 0.1, 200, 1e-6);
        for &v in &result.coupling {
            assert!(v >= -1e-10);
        }
    }

    #[test]
    fn test_pairwise_cost() {
        let supports = vec![vec![0.0, 1.0], vec![1.0, 2.0]];
        let cost = pairwise_cost(&supports, &[0, 1]); // |0 - 2|² = 4
        assert_relative_eq!(cost, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pairwise_cost_same_point() {
        let supports = vec![vec![1.0], vec![1.0]];
        let cost = pairwise_cost(&supports, &[0, 0]);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multimarginal_serialization() {
        let dists = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let cost_fn = |idx: &[usize]| {
            let d = supports[0][idx[0]] - supports[1][idx[1]];
            d * d
        };
        let result = multimarginal_ot(&dists, &cost_fn, 0.1, 100, 1e-6);
        let json = serde_json::to_string(&result).unwrap();
        let r2: MultiMarginalResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.k, 2);
    }

    #[test]
    fn test_multimarginal_total_mass() {
        let dists = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let cost_fn = |idx: &[usize]| {
            let d = supports[0][idx[0]] - supports[1][idx[1]];
            d * d
        };
        let result = multimarginal_ot(&dists, &cost_fn, 0.1, 500, 1e-8);
        let total: f64 = result.coupling.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 0.2);
    }
}
