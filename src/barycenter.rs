use crate::{DiscreteMeasure, WassersteinDistance};

/// Wasserstein barycenter — the Fréchet mean in Wasserstein space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WassersteinBarycenter;

impl WassersteinBarycenter {
    /// Compute barycenter via fixed-point iteration on quantile functions.
    /// The barycenter's quantile function is the weighted average of input quantiles.
    pub fn compute(
        measures: &[DiscreteMeasure],
        weights: &[f64],
        n_support: usize,
        _iterations: usize,
    ) -> DiscreteMeasure {
        assert_eq!(measures.len(), weights.len());
        assert!(weights.iter().all(|w| *w >= 0.0));
        let w_sum: f64 = weights.iter().sum();
        let norm_weights: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();

        // Uniform grid on [0, 1] for quantile evaluation
        let grid: Vec<f64> = (0..n_support)
            .map(|k| (k as f64 + 0.5) / n_support as f64)
            .collect();

        // Barycenter quantile = weighted average of quantiles
        let support: Vec<f64> = grid
            .iter()
            .map(|&t| {
                measures
                    .iter()
                    .zip(norm_weights.iter())
                    .map(|(m, w)| w * m.quantile(t))
                    .sum()
            })
            .collect();

        let weight = 1.0 / n_support as f64;
        DiscreteMeasure::new(support, vec![weight; n_support])
    }

    /// Check if a candidate is approximately a barycenter.
    pub fn is_barycenter(
        candidate: &DiscreteMeasure,
        measures: &[DiscreteMeasure],
        weights: &[f64],
        tolerance: f64,
    ) -> bool {
        let w_sum: f64 = weights.iter().sum();
        let norm_weights: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();

        let weighted_dist: f64 = measures
            .iter()
            .zip(norm_weights.iter())
            .map(|(m, w)| w * WassersteinDistance::w2(candidate, m).powi(2))
            .sum();

        // Compare with a fresh computation
        let fresh = Self::compute(measures, weights, candidate.n(), 1);
        let fresh_dist: f64 = measures
            .iter()
            .zip(norm_weights.iter())
            .map(|(m, w)| w * WassersteinDistance::w2(&fresh, m).powi(2))
            .sum();

        (weighted_dist - fresh_dist).abs() < tolerance
    }
}
