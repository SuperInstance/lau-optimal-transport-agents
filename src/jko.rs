use crate::{DiscreteMeasure, WassersteinDistance};

/// Jordan-Kinderlehrer-Otto gradient flow in Wasserstein space.
/// μ_{n+1} = argmin_ν {τ·F(ν) + W₂²(μₙ, ν)/2}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JKOFlow;

impl JKOFlow {
    /// Single JKO step.
    /// For 1D measures, we approximate this by moving the measure toward
    /// the minimizer of the functional.
    pub fn step(
        current: &DiscreteMeasure,
        functional: &dyn Fn(&DiscreteMeasure) -> f64,
        tau: f64,
        grid_size: usize,
    ) -> DiscreteMeasure {
        // Discretize: try shifting the measure along a grid and pick the best
        let mean = current.mean();
        let std = (current.variance().max(0.01)).sqrt();
        let range = std * 3.0;

        let mut best_measure = current.clone();
        let mut best_val = f64::INFINITY;

        for k in 0..grid_size {
            let alpha = k as f64 / (grid_size - 1).max(1) as f64;
            let shift = mean - range + 2.0 * range * alpha;
            let spread = std * (0.5 + 1.5 * k as f64 / grid_size as f64);

            let trial = DiscreteMeasure::new(
                current.support.iter().map(|x| shift + (*x - mean) * (spread / std)).collect(),
                current.weights.clone(),
            );

            let f_val = functional(&trial);
            let w2_sq = WassersteinDistance::w2(current, &trial).powi(2);
            let obj = tau * f_val + w2_sq / 2.0;

            if obj < best_val {
                best_val = obj;
                best_measure = trial;
            }
        }

        best_measure
    }

    /// Run multiple JKO steps.
    pub fn run(
        initial: &DiscreteMeasure,
        functional: &dyn Fn(&DiscreteMeasure) -> f64,
        tau: f64,
        steps: usize,
        grid_size: usize,
    ) -> Vec<DiscreteMeasure> {
        let mut trajectory = vec![initial.clone()];
        let mut current = initial.clone();
        for _ in 0..steps {
            current = Self::step(&current, functional, tau, grid_size);
            trajectory.push(current.clone());
        }
        trajectory
    }
}
