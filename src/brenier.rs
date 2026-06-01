//! Brenier's theorem: the optimal transport map between continuous distributions
//! is the gradient of a convex function.
//!
//! For quadratic cost, the optimal map T from μ to ν satisfies T = ∇φ where φ
//! is a convex function (the Brenier potential). In the discrete 1D case,
//! this reduces to monotone rearrangement.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A Brenier map (optimal transport map) between distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrenierMap {
    /// Source support points.
    pub source_support: Vec<f64>,
    /// Target support points (the pushforward locations).
    pub target_support: Vec<f64>,
    /// The transport map T[i]: maps source[i] → target[i].
    pub transport_map: Vec<f64>,
    /// The Brenier potential values at source points.
    pub potential: Vec<f64>,
}

/// Compute the Brenier map (optimal transport map) between two 1D distributions.
///
/// For 1D distributions with quadratic cost, the optimal map is the monotone
/// rearrangement: T = F_ν^{-1} ∘ F_μ (composition of inverse CDF and CDF).
///
/// # Arguments
/// * `mu` - Source probability vector
/// * `nu` - Target probability vector
/// * `support_a` - Source support points (must be sorted)
/// * `support_b` - Target support points (must be sorted)
pub fn brenier_map(
    mu: &DVector<f64>,
    nu: &DVector<f64>,
    support_a: &[f64],
    support_b: &[f64],
) -> BrenierMap {
    let n = support_a.len();

    // Compute CDF of source
    let mut cdf_a = vec![0.0; n + 1];
    for i in 0..n {
        cdf_a[i + 1] = cdf_a[i] + mu[i];
    }

    // Compute inverse CDF of target (generalized)
    let m = support_b.len();
    let mut cdf_b = vec![0.0; m + 1];
    for j in 0..m {
        cdf_b[j + 1] = cdf_b[j] + nu[j];
    }

    // For each source point, compute T(x_i) = F_b^{-1}(F_a(x_i))
    // Use the midpoint of the quantile interval for each mass point
    let mut transport_map = vec![0.0; n];
    for i in 0..n {
        let t = (cdf_a[i] + cdf_a[i + 1]) / 2.0; // midpoint quantile
        transport_map[i] = inverse_cdf(&cdf_b, support_b, t);
    }

    // Compute Brenier potential (convex function whose gradient is the transport map)
    // φ(x) = ∫_0^x T(s) ds (approximately)
    let mut potential = vec![0.0; n];
    for i in 1..n {
        let dx = support_a[i] - support_a[i - 1];
        potential[i] = potential[i - 1] + transport_map[i - 1] * dx;
    }

    BrenierMap {
        source_support: support_a.to_vec(),
        target_support: support_b.to_vec(),
        transport_map,
        potential,
    }
}

/// Evaluate the inverse CDF at value t.
/// For discrete distributions, returns the support point of the interval containing t.
fn inverse_cdf(cdf: &[f64], support: &[f64], t: f64) -> f64 {
    if t <= cdf[0] {
        return support[0];
    }
    if t >= *cdf.last().unwrap() {
        return *support.last().unwrap();
    }
    // Binary search for the interval
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
    // Return the midpoint of the support values in this CDF interval
    // For discrete OT, this is the representative location
    let idx_lo = lo.min(support.len() - 1);
    let idx_hi = hi.min(support.len() - 1);
    // Weighted midpoint based on how far t is into the interval
    let interval_width = cdf[hi] - cdf[lo];
    if interval_width < 1e-15 {
        support[idx_lo]
    } else {
        let frac = (t - cdf[lo]) / interval_width;
        support[idx_lo] * (1.0 - frac) + support[idx_hi] * frac
    }
}

/// Compute the pushforward of a distribution through a Brenier map.
pub fn pushforward(mu: &DVector<f64>, map: &BrenierMap, target_support: &[f64]) -> DVector<f64> {
    let m = target_support.len();
    let mut nu = vec![0.0; m];

    for (i, &target) in map.transport_map.iter().enumerate() {
        // Find closest target support point
        let mut best_j = 0;
        let mut best_dist = f64::INFINITY;
        for j in 0..m {
            let d = (target - target_support[j]).abs();
            if d < best_dist {
                best_dist = d;
                best_j = j;
            }
        }
        nu[best_j] += mu[i];
    }

    DVector::from_vec(nu)
}

/// Check if a function is convex (the Brenier potential should be convex).
pub fn is_convex(potential: &[f64], support: &[f64]) -> bool {
    if potential.len() < 3 {
        return true;
    }
    for i in 1..potential.len() - 1 {
        let h_left = support[i] - support[i - 1];
        let h_right = support[i + 1] - support[i];
        let slope_left = (potential[i] - potential[i - 1]) / h_left;
        let slope_right = (potential[i + 1] - potential[i]) / h_right;
        if slope_right < slope_left - 1e-8 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_brenier_identity() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let map = brenier_map(&mu, &nu, sa, sb);
        // For identical distributions, transport should not move mass far
        assert!(map.transport_map[0] >= 0.0 && map.transport_map[0] <= 1.0);
        assert!(map.transport_map[1] >= 0.0 && map.transport_map[1] <= 1.0);
    }

    #[test]
    fn test_brenier_shift() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[1.0, 2.0];
        let map = brenier_map(&mu, &nu, sa, sb);
        // Should shift right (toward [1, 2])
        assert!(map.transport_map[0] >= 0.5);
        assert!(map.transport_map[1] >= 1.0);
    }

    #[test]
    fn test_brenier_potential_convex() {
        let mu = DVector::from_vec(vec![0.3, 0.4, 0.3]);
        let nu = DVector::from_vec(vec![0.2, 0.5, 0.3]);
        let sa = &[0.0, 1.0, 2.0];
        let sb = &[0.0, 1.0, 2.0];
        let map = brenier_map(&mu, &nu, sa, sb);
        assert!(is_convex(&map.potential, sa));
    }

    #[test]
    fn test_brenier_monotone() {
        let mu = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let nu = DVector::from_vec(vec![0.4, 0.3, 0.3]);
        let sa = &[0.0, 1.0, 2.0];
        let sb = &[0.0, 1.0, 2.0];
        let map = brenier_map(&mu, &nu, sa, sb);
        // Optimal map should be monotone
        for i in 1..map.transport_map.len() {
            assert!(map.transport_map[i] >= map.transport_map[i - 1] - 1e-8);
        }
    }

    #[test]
    fn test_pushforward_preserves_mass() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let map = brenier_map(&mu, &nu, sa, sb);
        let pushed = pushforward(&mu, &map, sb);
        let total: f64 = pushed.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_brenier_serialization() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let map = brenier_map(&mu, &nu, sa, sb);
        let json = serde_json::to_string(&map).unwrap();
        let m2: BrenierMap = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.transport_map.len(), 2);
    }

    #[test]
    fn test_inverse_cdf() {
        let cdf = &[0.0, 0.25, 0.5, 0.75, 1.0];
        let support = &[0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(inverse_cdf(cdf, support, 0.5), 2.0, epsilon = 1e-10);
        assert_relative_eq!(inverse_cdf(cdf, support, 0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(inverse_cdf(cdf, support, 1.0), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_brenier_5_points() {
        let mu = DVector::from_vec(vec![0.2, 0.2, 0.2, 0.2, 0.2]);
        let nu = DVector::from_vec(vec![0.1, 0.2, 0.4, 0.2, 0.1]);
        let sa: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let sb: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let map = brenier_map(&mu, &nu, &sa, &sb);
        assert_eq!(map.transport_map.len(), 5);
        assert!(is_convex(&map.potential, &sa));
    }

    #[test]
    fn test_brenier_potential_zero_at_origin() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let nu = DVector::from_vec(vec![0.5, 0.5]);
        let sa = &[0.0, 1.0];
        let sb = &[0.0, 1.0];
        let map = brenier_map(&mu, &nu, sa, sb);
        assert_relative_eq!(map.potential[0], 0.0, epsilon = 1e-10);
    }
}
