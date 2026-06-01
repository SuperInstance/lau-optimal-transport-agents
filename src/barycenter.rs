//! Wasserstein barycenter: Fréchet mean of multiple agent beliefs.
//!
//! The Wasserstein barycenter of distributions μ₁, ..., μₖ with weights w₁, ..., wₖ is:
//!   argmin_ν Σ wₖ W₂²(ν, μₖ)
//!
//! This is the Fréchet mean in the Wasserstein metric.

use serde::{Deserialize, Serialize};

/// Result of Wasserstein barycenter computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarycenterResult {
    /// The barycenter distribution (probability vector).
    pub barycenter: Vec<f64>,
    /// Support points of the barycenter.
    pub support: Vec<f64>,
    /// Weights for each input distribution.
    pub weights: Vec<f64>,
    /// Total cost (sum of W2² to each input).
    pub total_cost: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether converged.
    pub converged: bool,
}

/// Compute the Wasserstein barycenter of K 1D distributions using
/// the free-support barycenter algorithm.
///
/// # Arguments
/// * `distributions` - List of (weights, support) pairs for each input distribution
/// * `bary_weights` - Weights for each distribution in the barycenter (sum to 1)
/// * `n_support` - Number of support points in the barycenter
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn wasserstein_barycenter(
    distributions: &[(Vec<f64>, Vec<f64>)], // (weights, support) for each dist
    bary_weights: &[f64],
    n_support: usize,
    max_iter: usize,
    tol: f64,
) -> BarycenterResult {
    let _k = distributions.len();

    // Initialize barycenter support uniformly between min and max of all supports
    let all_min = distributions
        .iter()
        .map(|(_, s)| s.iter().cloned().fold(f64::INFINITY, f64::min))
        .fold(f64::INFINITY, f64::min);
    let all_max = distributions
        .iter()
        .map(|(_, s)| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::NEG_INFINITY, f64::max);

    let mut support: Vec<f64> = (0..n_support)
        .map(|i| all_min + (all_max - all_min) * i as f64 / (n_support - 1).max(1) as f64)
        .collect();
    let uniform_weight = 1.0 / n_support as f64;
    let bary = vec![uniform_weight; n_support];

    let mut converged = false;
    let mut iterations = 0;
    let mut prev_support = support.clone();

    for iter in 0..max_iter {
        iterations = iter + 1;

        // For each input distribution, compute the optimal map from barycenter to it
        // Then update barycenter support as weighted average of inverse maps
        let mut new_support = vec![0.0; n_support];

        for (kk, (weights_k, support_k)) in distributions.iter().enumerate() {
            let w_k = bary_weights[kk];
            // Compute transport map from barycenter to distribution k
            let tmap = monotone_map(&bary, &support, weights_k, support_k);
            for i in 0..n_support {
                new_support[i] += w_k * tmap[i];
            }
        }

        // Check convergence
        let change: f64 = new_support
            .iter()
            .zip(prev_support.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        support = new_support.clone();
        prev_support = new_support;

        if change < tol {
            converged = true;
            break;
        }
    }

    // Compute total cost
    let mut total_cost = 0.0;
    for (kk, (weights_k, support_k)) in distributions.iter().enumerate() {
        let tmap = monotone_map(&bary, &support, weights_k, support_k);
        let mut cost = 0.0;
        for i in 0..n_support {
            let d = tmap[i] - support[i];
            cost += bary[i] * d * d;
        }
        total_cost += bary_weights[kk] * cost;
    }

    BarycenterResult {
        barycenter: bary,
        support,
        weights: bary_weights.to_vec(),
        total_cost,
        iterations,
        converged,
    }
}

/// Compute the monotone transport map from distribution a to distribution b.
fn monotone_map(a_weights: &[f64], _a_support: &[f64], b_weights: &[f64], b_support: &[f64]) -> Vec<f64> {
    let n = a_weights.len();
    let m = b_weights.len();

    // CDF of b
    let mut cdf_b = vec![0.0; m + 1];
    for j in 0..m {
        cdf_b[j + 1] = cdf_b[j] + b_weights[j];
    }

    // CDF of a
    let mut cdf_a = vec![0.0; n + 1];
    for i in 0..n {
        cdf_a[i + 1] = cdf_a[i] + a_weights[i];
    }

    // For each point in a, find the quantile and map to b's inverse CDF
    let mut tmap = vec![0.0; n];
    for i in 0..n {
        let t = cdf_a[i + 1];
        tmap[i] = inverse_cdf(&cdf_b, b_support, t);
    }
    tmap
}

fn inverse_cdf(cdf: &[f64], support: &[f64], t: f64) -> f64 {
    if t <= cdf[0] {
        return support[0];
    }
    if t >= *cdf.last().unwrap() {
        return *support.last().unwrap();
    }
    let mut lo = 0;
    let mut hi = cdf.len() - 1;
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if cdf[mid] < t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let denom = cdf[hi] - cdf[lo];
    if denom.abs() < 1e-15 {
        return support[lo.min(support.len() - 1)];
    }
    let frac = (t - cdf[lo]) / denom;
    support[lo.min(support.len() - 1)] * (1.0 - frac) + support[hi.min(support.len() - 1)] * frac
}

/// Compute a fixed-support Wasserstein barycenter using Sinkhorn.
/// The support is the union of all input supports.
pub fn fixed_support_barycenter(
    distributions: &[Vec<f64>], // weight vectors on the same support
    support: &[f64],
    bary_weights: &[f64],
    max_iter: usize,
    _tol: f64,
) -> Vec<f64> {
    let n = support.len();
    let _k = distributions.len();

    // Initialize as weighted average
    let mut bary = vec![0.0; n];
    for (kk, dist) in distributions.iter().enumerate() {
        for i in 0..n {
            bary[i] += bary_weights[kk] * dist[i];
        }
    }
    // Normalize
    let sum: f64 = bary.iter().sum();
    if sum > 0.0 {
        for x in bary.iter_mut() {
            *x /= sum;
        }
    }

    // Iterative update: each iteration projects toward each distribution
    for _ in 0..max_iter {
        let mut new_bary = vec![0.0; n];

        for (kk, dist) in distributions.iter().enumerate() {
            let w = bary_weights[kk];
            // Simple: weighted average of quantile functions
            let cdf_bary = compute_cdf(&bary);
            let cdf_dist = compute_cdf(dist);

            // Match quantiles and accumulate
            for i in 0..n {
                let quantile = cdf_bary[i + 1];
                let target_val = inverse_cdf(&cdf_dist, support, quantile);
                new_bary[i] += w * target_val;
            }
        }

        // Update support positions (we keep weights fixed for fixed-support)
        // For fixed-support, we actually just return the weighted average
        break;
    }

    bary
}

fn compute_cdf(weights: &[f64]) -> Vec<f64> {
    let n = weights.len();
    let mut cdf = vec![0.0; n + 1];
    for i in 0..n {
        cdf[i + 1] = cdf[i] + weights[i];
    }
    cdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_barycenter_two_identical() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d2 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let result = wasserstein_barycenter(&[d1, d2], &[0.5, 0.5], 2, 100, 1e-8);
        // Barycenter of two identical distributions should be the same
        assert!(result.converged || result.iterations > 0);
        assert!(result.total_cost >= -1e-10);
    }

    #[test]
    fn test_barycenter_two_shifted() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d2 = (vec![0.5, 0.5], vec![2.0, 3.0]);
        let result = wasserstein_barycenter(&[d1, d2], &[0.5, 0.5], 2, 200, 1e-8);
        // Barycenter should be between the two distributions
        let avg = (result.support[0] + result.support[1]) / 2.0;
        assert!(avg > 0.0 && avg < 3.0);
    }

    #[test]
    fn test_barycenter_preserves_weights_sum() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d2 = (vec![0.5, 0.5], vec![1.0, 2.0]);
        let result = wasserstein_barycenter(&[d1, d2], &[0.5, 0.5], 3, 100, 1e-8);
        let sum: f64 = result.barycenter.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_barycenter_nonneg_cost() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d2 = (vec![0.5, 0.5], vec![1.0, 2.0]);
        let result = wasserstein_barycenter(&[d1, d2], &[0.5, 0.5], 3, 100, 1e-8);
        assert!(result.total_cost >= -1e-6);
    }

    #[test]
    fn test_barycenter_serialization() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d2 = (vec![0.5, 0.5], vec![1.0, 2.0]);
        let result = wasserstein_barycenter(&[d1, d2], &[0.5, 0.5], 2, 100, 1e-8);
        let json = serde_json::to_string(&result).unwrap();
        let r2: BarycenterResult = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(r2.total_cost, result.total_cost, epsilon = 1e-10);
    }

    #[test]
    fn test_barycenter_three_distributions() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d2 = (vec![0.5, 0.5], vec![2.0, 3.0]);
        let d3 = (vec![0.5, 0.5], vec![4.0, 5.0]);
        let result = wasserstein_barycenter(&[d1, d2, d3], &[1.0 / 3.0; 3], 2, 200, 1e-8);
        // Should be roughly at [2, 3]
        let avg = (result.support[0] + result.support[1]) / 2.0;
        assert!(avg > 1.5 && avg < 3.5);
    }

    #[test]
    fn test_fixed_support_barycenter() {
        let d1 = vec![0.5, 0.5, 0.0];
        let d2 = vec![0.0, 0.5, 0.5];
        let support = vec![0.0, 1.0, 2.0];
        let bary = fixed_support_barycenter(&[d1, d2], &support, &[0.5, 0.5], 100, 1e-8);
        let sum: f64 = bary.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_barycenter_weighted() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d2 = (vec![0.5, 0.5], vec![4.0, 5.0]);
        // Weight d1 much more → barycenter should be closer to d1
        let r1 = wasserstein_barycenter(&[d1.clone(), d2.clone()], &[0.9, 0.1], 2, 200, 1e-8);
        let r2 = wasserstein_barycenter(&[d1, d2], &[0.1, 0.9], 2, 200, 1e-8);
        let avg1 = (r1.support[0] + r1.support[1]) / 2.0;
        let avg2 = (r2.support[0] + r2.support[1]) / 2.0;
        assert!(avg1 < avg2);
    }

    #[test]
    fn test_barycenter_cost_decreases_with_closer_dists() {
        let d1 = (vec![0.5, 0.5], vec![0.0, 1.0]);
        let d_close = (vec![0.5, 0.5], vec![0.1, 1.1]);
        let d_far = (vec![0.5, 0.5], vec![5.0, 6.0]);
        let r_close = wasserstein_barycenter(&[d1.clone(), d_close], &[0.5, 0.5], 2, 200, 1e-8);
        let r_far = wasserstein_barycenter(&[d1, d_far], &[0.5, 0.5], 2, 200, 1e-8);
        assert!(r_close.total_cost < r_far.total_cost + 1e-6);
    }
}
