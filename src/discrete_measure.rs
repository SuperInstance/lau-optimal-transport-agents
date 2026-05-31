use serde::{Deserialize, Serialize};

/// A probability measure on a discrete space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscreteMeasure {
    pub support: Vec<f64>,
    pub weights: Vec<f64>,
}

impl DiscreteMeasure {
    pub fn new(support: Vec<f64>, weights: Vec<f64>) -> Self {
        assert_eq!(support.len(), weights.len(), "support and weights must have same length");
        Self { support, weights }
    }

    pub fn uniform(support: Vec<f64>) -> Self {
        let n = support.len();
        let w = 1.0 / n as f64;
        Self {
            support,
            weights: vec![w; n],
        }
    }

    pub fn dirac(x: f64) -> Self {
        Self {
            support: vec![x],
            weights: vec![1.0],
        }
    }

    pub fn n(&self) -> usize {
        self.support.len()
    }

    pub fn total_mass(&self) -> f64 {
        self.weights.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        let m = self.total_mass();
        self.support
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            / m
    }

    pub fn variance(&self) -> f64 {
        let m = self.total_mass();
        let mu = self.mean();
        self.support
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| w * (x - mu).powi(2))
            .sum::<f64>()
            / m
    }

    pub fn is_probability(&self) -> bool {
        self.weights.iter().all(|w| *w >= 0.0)
            && (self.total_mass() - 1.0).abs() < 1e-9
    }

    pub fn normalize(&mut self) {
        let m = self.total_mass();
        if m > 0.0 {
            for w in &mut self.weights {
                *w /= m;
            }
        }
    }

    /// Empirical CDF value at x.
    pub fn cdf(&self, x: f64) -> f64 {
        self.support
            .iter()
            .zip(self.weights.iter())
            .filter(|(s, _)| **s <= x)
            .map(|(_, w)| w)
            .sum()
    }

    /// Quantile function (inverse CDF) at level t ∈ [0,1].
    /// Interpolates linearly between support points.
    pub fn quantile(&self, t: f64) -> f64 {
        let mut cumulative = 0.0;
        // Sort indices by support
        let mut indices: Vec<usize> = (0..self.n()).collect();
        indices.sort_by(|&a, &b| self.support[a].partial_cmp(&self.support[b]).unwrap());

        for &idx in &indices {
            let _prev_cum = cumulative;
            cumulative += self.weights[idx];
            if cumulative >= t - 1e-12 {
                return self.support[idx];
            }
        }
        // If we're here, return last support point
        self.support[*indices.last().unwrap()]
    }
}
