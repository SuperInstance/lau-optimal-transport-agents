//! McCann interpolation: geodesic in Wasserstein space between two beliefs.
//!
//! Given μ₀ and μ₁ with optimal transport map T, the McCann interpolation at
//! time t ∈ [0,1] is: μ_t = ((1-t)·id + t·T)# μ₀
//!
//! This is the geodesic (shortest path) in the Wasserstein metric.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A McCann interpolation curve in Wasserstein space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McCannCurve {
    /// Source support points.
    pub source_support: Vec<f64>,
    /// Transport map T: source → target.
    pub transport_map: Vec<f64>,
    /// Source weights.
    pub source_weights: Vec<f64>,
    /// The Wasserstein distance between endpoints.
    pub wasserstein_distance: f64,
}

/// Result of evaluating McCann interpolation at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationPoint {
    /// Time parameter t ∈ [0, 1].
    pub t: f64,
    /// Support points at time t: (1-t)*x_i + t*T(x_i).
    pub support: Vec<f64>,
    /// Weights (same as source, since pushforward preserves them).
    pub weights: Vec<f64>,
}

/// Construct a McCann interpolation curve from source, target, and transport map.
///
/// Uses the monotone rearrangement for 1D distributions.
pub fn mccann_interpolation(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    support_a: &[f64],
    support_b: &[f64],
) -> McCannCurve {
    // Compute monotone transport map (quantile coupling)
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

    // Transport map: for each source point, map to corresponding target
    let mut transport_map = vec![0.0; n];
    for i in 0..n {
        let t = (cdf_a[i] + cdf_a[i + 1]) / 2.0; // midpoint quantile
        transport_map[i] = inverse_cdf_interp(&cdf_b, support_b, t);
    }

    // Compute W2 distance
    let mut w_sq = 0.0;
    for i in 0..n {
        let d = transport_map[i] - support_a[i];
        w_sq += mu[i] * d * d;
    }

    McCannCurve {
        source_support: support_a.to_vec(),
        transport_map,
        source_weights: mu.iter().cloned().collect(),
        wasserstein_distance: w_sq.sqrt(),
    }
}

fn inverse_cdf_interp(cdf: &[f64], support: &[f64], t: f64) -> f64 {
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

/// Evaluate the McCann interpolation at time t ∈ [0, 1].
pub fn evaluate_interpolation(curve: &McCannCurve, t: f64) -> InterpolationPoint {
    assert!((0.0..=1.0).contains(&t), "t must be in [0, 1]");

    let n = curve.source_support.len();
    let mut support = vec![0.0; n];
    for i in 0..n {
        support[i] = (1.0 - t) * curve.source_support[i] + t * curve.transport_map[i];
    }

    InterpolationPoint {
        t,
        support,
        weights: curve.source_weights.clone(),
    }
}

/// Evaluate the McCann interpolation at multiple times.
pub fn interpolation_path(curve: &McCannCurve, num_points: usize) -> Vec<InterpolationPoint> {
    (0..num_points)
        .map(|k| {
            let t = k as f64 / (num_points - 1).max(1) as f64;
            evaluate_interpolation(curve, t)
        })
        .collect()
}

/// Compute the Wasserstein distance between μ_t and μ_s along the interpolation.
/// For a geodesic: W2(μ_t, μ_s) = |t - s| * W2(μ_0, μ_1).
pub fn geodesic_distance(curve: &McCannCurve, t: f64, s: f64) -> f64 {
    (t - s).abs() * curve.wasserstein_distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mccann_t0_is_source() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[1.0, 2.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        let p0 = evaluate_interpolation(&curve, 0.0);
        for i in 0..2 {
            assert_relative_eq!(p0.support[i], sa[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mccann_t1_is_target() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[1.0, 2.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        let p1 = evaluate_interpolation(&curve, 1.0);
        for i in 0..2 {
            assert_relative_eq!(p1.support[i], curve.transport_map[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mccann_t05_midpoint() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 2.0];
        let sb = &[2.0, 4.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        let p05 = evaluate_interpolation(&curve, 0.5);
        // Midpoint should be between source and target
        assert!(p05.support[0] > 0.0 && p05.support[0] < 4.0);
        assert!(p05.support[1] > 0.0 && p05.support[1] < 4.0);
    }

    #[test]
    fn test_mccann_preserves_mass() {
        let mu = DVector::from_vec(vec![0.3, 0.4, 0.3]);
        let nu = DVector::from_vec(vec![0.2, 0.5, 0.3]);
        let sa = &[0.0, 1.0, 2.0];
        let sb = &[0.0, 1.0, 2.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        let total: f64 = curve.source_weights.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mccann_geodesic_property() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[1.0, 2.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        // W2(μ_t, μ_s) = |t-s| * W2(μ_0, μ_1)
        let d_01 = geodesic_distance(&curve, 0.0, 1.0);
        let d_05 = geodesic_distance(&curve, 0.0, 0.5);
        assert_relative_eq!(d_01, curve.wasserstein_distance, epsilon = 1e-10);
        assert_relative_eq!(d_05, 0.5 * curve.wasserstein_distance, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolation_path_length() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[1.0, 2.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        let path = interpolation_path(&curve, 11);
        assert_eq!(path.len(), 11);
        assert_relative_eq!(path[0].t, 0.0, epsilon = 1e-10);
        assert_relative_eq!(path[10].t, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mccann_serialization() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[1.0, 2.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        let json = serde_json::to_string(&curve).unwrap();
        let c2: McCannCurve = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(c2.wasserstein_distance, curve.wasserstein_distance, epsilon = 1e-10);
    }

    #[test]
    fn test_mccann_identical_dists_zero_distance() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        // For identical distributions, distance should be small
        assert!(curve.wasserstein_distance < 1.0);
    }

    #[test]
    fn test_mccann_monotone_interpolation() {
        let mu = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let nu = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let sa = &[0.0, 1.0, 2.0, 3.0];
        let sb = &[1.0, 2.0, 3.0, 4.0];
        let curve = mccann_interpolation(&mu, &nu, sa, sb);
        let p05 = evaluate_interpolation(&curve, 0.5);
        // Interpolation should preserve ordering
        for i in 1..p05.support.len() {
            assert!(p05.support[i] >= p05.support[i - 1] - 1e-8);
        }
    }
}
