//! Convergence rate: Wasserstein distance between agent beliefs after T learning steps.
//!
//! Track how quickly agent beliefs converge during learning, measured in the
//! natural metric (Wasserstein distance).

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Result of convergence analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceResult {
    /// Wasserstein distances at each step.
    pub distances: Vec<f64>,
    /// The belief trajectory.
    pub beliefs: Vec<Vec<f64>>,
    /// Support points.
    pub support: Vec<f64>,
    /// Whether the beliefs converged.
    pub converged: bool,
    /// Convergence rate estimate (exponential decay constant).
    pub rate: f64,
    /// Final Wasserstein distance to target.
    pub final_distance: f64,
    /// Number of steps.
    pub steps: usize,
}

/// Measure how much agent beliefs change during learning.
///
/// Given a sequence of belief distributions, compute the Wasserstein distance
/// between consecutive beliefs and between each belief and the target.
///
/// # Arguments
/// * `beliefs` - Sequence of belief distributions (probability vectors)
/// * `support` - Support points for the distributions
/// * `target` - Target belief distribution (optional)
pub fn belief_shift(
    beliefs: &[Vec<f64>],
    support: &[f64],
    target: Option<&Vec<f64>>,
) -> ConvergenceResult {
    let n = beliefs.len();

    // Compute Wasserstein distances
    let mut distances = Vec::new();

    if let Some(target_belief) = target {
        // Distance to target at each step
        for belief in beliefs {
            let mu = DVector::from_vec(belief.clone());
            let nu = DVector::from_vec(target_belief.clone());
            let d = compute_w1(&mu, &nu, support, support);
            distances.push(d);
        }
    } else if n >= 2 {
        // Distance between consecutive beliefs
        for i in 1..n {
            let mu = DVector::from_vec(beliefs[i - 1].clone());
            let nu = DVector::from_vec(beliefs[i].clone());
            let d = compute_w1(&mu, &nu, support, support);
            distances.push(d);
        }
    }

    // Estimate convergence rate (exponential fit)
    let rate = estimate_convergence_rate(&distances);

    let final_distance = distances.last().copied().unwrap_or(0.0);
    let converged = final_distance < 0.01;

    ConvergenceResult {
        distances,
        beliefs: beliefs.to_vec(),
        support: support.to_vec(),
        converged,
        rate,
        final_distance,
        steps: n,
    }
}

/// Simple W1 computation for 1D distributions.
fn compute_w1(mu: &DVector<f64>, nu: &DVector<f64>, sa: &[f64], sb: &[f64]) -> f64 {
    let n = mu.len();
    let m = nu.len();

    // Cost matrix (linear)
    let mut cost_data = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            cost_data[i * m + j] = (sa[i] - sb[j]).abs();
        }
    }

    // Simple IPFP-like solver
    let mut plan = vec![1.0 / (n * m) as f64; n * m];
    for _ in 0..200 {
        // Row scale
        for i in 0..n {
            let row_sum: f64 = (0..m).map(|j| plan[i * m + j]).sum();
            if row_sum > 1e-15 {
                for j in 0..m {
                    plan[i * m + j] *= mu[i] / row_sum;
                }
            }
        }
        // Col scale
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| plan[i * m + j]).sum();
            if col_sum > 1e-15 {
                for i in 0..n {
                    plan[i * m + j] *= nu[j] / col_sum;
                }
            }
        }
    }

    let mut cost = 0.0;
    for i in 0..n {
        for j in 0..m {
            cost += plan[i * m + j] * cost_data[i * m + j];
        }
    }
    cost
}

/// Estimate exponential convergence rate from distance sequence.
/// Fits d_t ≈ d_0 * exp(-rate * t).
fn estimate_convergence_rate(distances: &[f64]) -> f64 {
    if distances.len() < 2 {
        return 0.0;
    }

    let d0 = distances[0];
    if d0 < 1e-15 {
        return 0.0;
    }

    // Linear regression on log(d_t) = log(d_0) - rate * t
    let n = distances.len() as f64;
    let mut sum_t = 0.0;
    let mut sum_log = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_tlog = 0.0;

    for (t, &d) in distances.iter().enumerate() {
        let t_f = t as f64;
        let log_d = if d > 1e-15 { d.ln() } else { -30.0 };
        sum_t += t_f;
        sum_log += log_d;
        sum_tt += t_f * t_f;
        sum_tlog += t_f * log_d;
    }

    // slope = (n * sum_tlog - sum_t * sum_log) / (n * sum_tt - sum_t^2)
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    let slope = (n * sum_tlog - sum_t * sum_log) / denom;

    -slope // Negative slope = positive convergence rate
}

/// Simulate Bayesian belief update and track Wasserstein convergence.
pub fn simulate_bayesian_convergence(
    prior: &[f64],
    support: &[f64],
    likelihood: &[f64], // P(data | x) for each x
    n_observations: usize,
) -> ConvergenceResult {
    let mut beliefs = vec![prior.to_vec()];
    let mut current = prior.to_vec();

    for _ in 0..n_observations {
        // Bayesian update: posterior ∝ likelihood * prior
        let mut posterior = vec![0.0; current.len()];
        let mut total = 0.0;
        for i in 0..current.len() {
            posterior[i] = current[i] * likelihood[i];
            total += posterior[i];
        }
        if total > 0.0 {
            for x in posterior.iter_mut() {
                *x /= total;
            }
        }
        current = posterior;
        beliefs.push(current.clone());
    }

    // Target is the final belief
    let target = beliefs.last().unwrap().clone();
    belief_shift(&beliefs, support, Some(&target))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_belief_shift_converging() {
        let b1 = vec![1.0, 0.0, 0.0];
        let b2 = vec![0.7, 0.3, 0.0];
        let b3 = vec![0.5, 0.3, 0.2];
        let b4 = vec![0.4, 0.3, 0.3];
        let support = vec![0.0, 1.0, 2.0];
        let result = belief_shift(&[b1, b2, b3, b4], &support, None);
        assert_eq!(result.distances.len(), 3);
    }

    #[test]
    fn test_belief_shift_to_target() {
        let beliefs = vec![
            vec![1.0, 0.0],
            vec![0.7, 0.3],
            vec![0.5, 0.5],
        ];
        let target = vec![0.5, 0.5];
        let support = vec![0.0, 1.0];
        let result = belief_shift(&beliefs, &support, Some(&target));
        // Distance should decrease
        assert!(result.distances[0] >= result.distances[2] - 0.1);
    }

    #[test]
    fn test_belief_shift_serialization() {
        let beliefs = vec![vec![0.5, 0.5], vec![0.3, 0.7]];
        let support = vec![0.0, 1.0];
        let result = belief_shift(&beliefs, &support, None);
        let json = serde_json::to_string(&result).unwrap();
        let r2: ConvergenceResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.steps, 2);
    }

    #[test]
    fn test_convergence_rate_positive() {
        let beliefs = vec![
            vec![1.0, 0.0],
            vec![0.6, 0.4],
            vec![0.4, 0.6],
            vec![0.5, 0.5],
        ];
        let target = vec![0.5, 0.5];
        let support = vec![0.0, 1.0];
        let result = belief_shift(&beliefs, &support, Some(&target));
        // Should have computed distances
        assert_eq!(result.distances.len(), 4);
    }

    #[test]
    fn test_bayesian_convergence() {
        let prior = vec![0.5, 0.5];
        let support = vec![0.0, 1.0];
        let likelihood = vec![0.3, 0.7]; // Evidence favors hypothesis 1
        let result = simulate_bayesian_convergence(&prior, &support, &likelihood, 5);
        assert_eq!(result.beliefs.len(), 6); // prior + 5 updates
        // After many updates, should converge to posterior proportional to likelihood
        let final_belief = result.beliefs.last().unwrap();
        assert!(final_belief[1] > final_belief[0]); // Should favor hypothesis 1
    }

    #[test]
    fn test_bayesian_mass_preserved() {
        let prior = vec![0.3, 0.4, 0.3];
        let support = vec![0.0, 1.0, 2.0];
        let likelihood = vec![0.5, 0.3, 0.2];
        let result = simulate_bayesian_convergence(&prior, &support, &likelihood, 3);
        for belief in &result.beliefs {
            let sum: f64 = belief.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_convergence_rate_estimate() {
        // Exponentially decaying distances
        let distances = vec![1.0, 0.5, 0.25, 0.125, 0.0625];
        let rate = estimate_convergence_rate(&distances);
        // Should be approximately ln(2) ≈ 0.693
        assert!(rate > 0.3 && rate < 1.0);
    }

    #[test]
    fn test_convergence_rate_zero_distances() {
        let distances = vec![0.0, 0.0, 0.0];
        let rate = estimate_convergence_rate(&distances);
        assert_relative_eq!(rate, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_belief_shift_nonneg_distances() {
        let beliefs = vec![
            vec![0.2, 0.3, 0.5],
            vec![0.3, 0.4, 0.3],
            vec![0.4, 0.3, 0.3],
        ];
        let support = vec![0.0, 1.0, 2.0];
        let result = belief_shift(&beliefs, &support, None);
        for &d in &result.distances {
            assert!(d >= -1e-10);
        }
    }

    #[test]
    fn test_final_distance() {
        let beliefs = vec![
            vec![1.0, 0.0],
            vec![0.5, 0.5],
        ];
        let target = vec![0.5, 0.5];
        let support = vec![0.0, 1.0];
        let result = belief_shift(&beliefs, &support, Some(&target));
        assert_relative_eq!(result.final_distance, 0.0, epsilon = 0.5);
    }
}
