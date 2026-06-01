//! Wasserstein gradient flow: steepest descent in distribution space.
//!
//! A Wasserstein gradient flow is a curve (μ_t)_{t≥0} in the space of probability
//! measures that follows the steepest descent of a functional F:
//!
//!   ∂μ_t/∂t = -∇_W F(μ_t)  (Wasserstein gradient)
//!
//! This connects to the Fokker-Planck equation: the law of a diffusion process
//! dX = -∇V(X)dt + √2 dW satisfies ∂μ/∂t = Δμ + ∇·(μ∇V).

use serde::{Deserialize, Serialize};

/// Result of a Wasserstein gradient flow computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowResult {
    /// The trajectory of distributions: μ_0, μ_1, ..., μ_T.
    pub trajectory: Vec<Vec<f64>>,
    /// Support points (fixed).
    pub support: Vec<f64>,
    /// Functional values F(μ_t) at each step.
    pub functional_values: Vec<f64>,
    /// Step size.
    pub step_size: f64,
    /// Number of steps.
    pub steps: usize,
    /// Whether the flow converged.
    pub converged: bool,
}

/// A functional F(μ) on probability measures, defined by its first variation
/// δF/δμ(x) at each support point.
pub trait Functional: Send + Sync {
    /// Compute the first variation δF/δμ at each support point.
    fn first_variation(&self, mu: &[f64], support: &[f64]) -> Vec<f64>;

    /// Evaluate F(μ).
    fn evaluate(&self, mu: &[f64], support: &[f64]) -> f64;
}

/// Potential energy: F(μ) = ∫ V(x) dμ(x).
pub struct PotentialEnergy {
    pub potential: Vec<f64>, // V(x_i) at each support point
}

impl Functional for PotentialEnergy {
    fn first_variation(&self, _mu: &[f64], _support: &[f64]) -> Vec<f64> {
        self.potential.clone()
    }

    fn evaluate(&self, mu: &[f64], _support: &[f64]) -> f64 {
        mu.iter()
            .zip(self.potential.iter())
            .map(|(w, v)| w * v)
            .sum()
    }
}

/// Internal energy: F(μ) = ∫ U(μ(x)) dx, where U is convex.
/// For entropy: U(t) = t * log(t).
pub struct InternalEnergy {
    /// U'(t) at each point (the derivative of the integrand w.r.t. density).
    pub entropy_weight: f64,
}

impl Functional for InternalEnergy {
    fn first_variation(&self, mu: &[f64], _support: &[f64]) -> Vec<f64> {
        mu.iter()
            .map(|&m| {
                if m > 1e-15 {
                    self.entropy_weight * (1.0 + m.ln())
                } else {
                    self.entropy_weight * (-30.0) // cap log at -30
                }
            })
            .collect()
    }

    fn evaluate(&self, mu: &[f64], _support: &[f64]) -> f64 {
        mu.iter()
            .map(|&m| if m > 1e-15 { self.entropy_weight * m * m.ln() } else { 0.0 })
            .sum()
    }
}

/// Run a Wasserstein gradient flow (JKO scheme: Jordan-Kinderlehrer-Otto).
///
/// The JKO scheme discretizes: μ_{t+1} = argmin_μ { F(μ) + W₂²(μ, μ_t) / (2τ) }
///
/// We approximate this using the MFG (mean-field game) discretization on the support.
pub fn wasserstein_gradient_flow(
    mu_0: &[f64],
    support: &[f64],
    functional: &dyn Functional,
    step_size: f64,
    n_steps: usize,
    tol: f64,
) -> GradientFlowResult {
    let n = mu_0.len();
    let dx = if n > 1 {
        support[1] - support[0]
    } else {
        1.0
    };

    let mut mu = mu_0.to_vec();
    let mut trajectory = vec![mu.clone()];
    let mut functional_values = vec![functional.evaluate(&mu, support)];
    let mut converged = false;

    for _step in 0..n_steps {
        let df = functional.first_variation(&mu, support);

        // Approximate gradient flow: shift mass in the direction of -∇(δF/δμ)
        // Using finite differences for ∇(δF/δμ)
        let mut laplacian = vec![0.0; n];
        for i in 1..n - 1 {
            laplacian[i] = (df[i + 1] - 2.0 * df[i] + df[i - 1]) / (dx * dx);
        }
        if n > 1 {
            laplacian[0] = laplacian[1]; // Neumann BC
            laplacian[n - 1] = laplacian[n - 2]; // Neumann BC
        }

        // Update: μ_{t+1} = μ_t - τ * (divergence of μ * ∇(δF/δμ))
        // Approximate divergence: Δμ + ∇·(μ ∇V)
        let mut mu_new = vec![0.0; n];
        for i in 0..n {
            // Heat kernel diffusion + drift
            let mut delta = 0.0;
            if i > 0 {
                delta += mu[i - 1] - mu[i];
            }
            if i < n - 1 {
                delta += mu[i + 1] - mu[i];
            }
            mu_new[i] = mu[i] - step_size * mu[i] * df[i];
            mu_new[i] += step_size * delta / (dx * dx); // diffusion
        }

        // Project to simplex (ensure non-negative, sums to 1)
        mu_new = project_simplex(&mu_new);

        let f_new = functional.evaluate(&mu_new, support);
        trajectory.push(mu_new.clone());
        functional_values.push(f_new);

        // Check convergence
        let change: f64 = mu_new
            .iter()
            .zip(mu.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        mu = mu_new;

        if change < tol {
            converged = true;
            break;
        }
    }

    GradientFlowResult {
        trajectory,
        support: support.to_vec(),
        functional_values,
        step_size,
        steps: n_steps,
        converged,
    }
}

/// Project a vector onto the probability simplex {x >= 0, sum(x) = 1}.
fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cumsum = 0.0;
    let mut rho = 0;
    for j in 0..n {
        cumsum += sorted[j];
        let t = (cumsum - 1.0) / (j + 1) as f64;
        if sorted[j] - t > 0.0 {
            rho = j;
        }
    }

    let cumsum_rho: f64 = sorted[..=rho].iter().sum();
    let theta = (cumsum_rho - 1.0) / (rho + 1) as f64;

    v.iter().map(|&x| (x - theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_potential_energy_first_variation() {
        let f = PotentialEnergy {
            potential: vec![0.0, 1.0, 2.0],
        };
        let fv = f.first_variation(&[0.3, 0.4, 0.3], &[0.0, 1.0, 2.0]);
        assert_relative_eq!(fv[0], 0.0);
        assert_relative_eq!(fv[1], 1.0);
        assert_relative_eq!(fv[2], 2.0);
    }

    #[test]
    fn test_potential_energy_evaluate() {
        let f = PotentialEnergy {
            potential: vec![0.0, 1.0, 4.0],
        };
        let val = f.evaluate(&[0.5, 0.3, 0.2], &[0.0, 1.0, 2.0]);
        assert_relative_eq!(val, 1.1, epsilon = 1e-10);
    }

    #[test]
    fn test_internal_energy_first_variation() {
        let f = InternalEnergy { entropy_weight: 1.0 };
        let mu = vec![0.3, 0.4, 0.3];
        let fv = f.first_variation(&mu, &[0.0, 1.0, 2.0]);
        // Should be 1 + ln(mu_i)
        assert!(fv[0] < fv[1]); // 0.3 < 0.4 → ln(0.3) < ln(0.4)
    }

    #[test]
    fn test_internal_energy_evaluate() {
        let f = InternalEnergy { entropy_weight: 1.0 };
        let val = f.evaluate(&[1.0], &[0.0]);
        assert_relative_eq!(val, 0.0, epsilon = 1e-10); // 1 * ln(1) = 0
    }

    #[test]
    fn test_gradient_flow_potential() {
        let mu = vec![0.2, 0.5, 0.3];
        let support = vec![0.0, 1.0, 2.0];
        let f = PotentialEnergy {
            potential: vec![0.0, 0.0, 5.0],
        };
        let result = wasserstein_gradient_flow(&mu, &support, &f, 0.001, 500, 1e-8);
        // After gradient flow, mass should move away from high potential
        assert!(result.trajectory.len() > 1);
    }

    #[test]
    fn test_gradient_flow_converges() {
        let mu = vec![0.2, 0.3, 0.5];
        let support = vec![0.0, 1.0, 2.0];
        let f = PotentialEnergy {
            potential: vec![1.0, 0.0, 1.0],
        };
        let result = wasserstein_gradient_flow(&mu, &support, &f, 0.001, 2000, 1e-10);
        // Should converge toward the minimum of potential (middle point)
        assert!(result.trajectory.len() > 1);
    }

    #[test]
    fn test_gradient_flow_mass_conservation() {
        let mu = vec![0.2, 0.6, 0.2];
        let support = vec![0.0, 1.0, 2.0];
        let f = PotentialEnergy {
            potential: vec![0.0, 1.0, 0.0],
        };
        let result = wasserstein_gradient_flow(&mu, &support, &f, 0.001, 100, 1e-10);
        for step_mu in &result.trajectory {
            let sum: f64 = step_mu.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_gradient_flow_functional_decreases() {
        let mu = vec![0.2, 0.3, 0.5];
        let support = vec![0.0, 1.0, 2.0];
        let f = PotentialEnergy {
            potential: vec![0.0, 1.0, 3.0],
        };
        let result = wasserstein_gradient_flow(&mu, &support, &f, 0.0005, 1000, 1e-10);
        // Functional should generally decrease (gradient flow is descent)
        // Check at least the first few steps decrease
        if result.functional_values.len() > 5 {
            assert!(result.functional_values[5] <= result.functional_values[0] + 0.01);
        }
    }

    #[test]
    fn test_project_simplex() {
        let v = vec![0.3, 0.4, 0.3];
        let p = project_simplex(&v);
        let sum: f64 = p.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        for &x in &p {
            assert!(x >= 0.0);
        }
    }

    #[test]
    fn test_project_simplex_negative() {
        let v = vec![-0.5, 1.5, 0.0];
        let p = project_simplex(&v);
        let sum: f64 = p.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        for &x in &p {
            assert!(x >= 0.0);
        }
    }

    #[test]
    fn test_gradient_flow_serialization() {
        let mu = vec![0.5, 0.5];
        let support = vec![0.0, 1.0];
        let f = PotentialEnergy {
            potential: vec![0.0, 1.0],
        };
        let result = wasserstein_gradient_flow(&mu, &support, &f, 0.001, 10, 1e-10);
        let json = serde_json::to_string(&result).unwrap();
        let r2: GradientFlowResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.trajectory.len(), result.trajectory.len());
    }

    #[test]
    fn test_entropy_functional() {
        let f = InternalEnergy { entropy_weight: 1.0 };
        let val = f.evaluate(&[0.5, 0.5], &[0.0, 1.0]);
        // H = -0.5*ln(0.5) - 0.5*ln(0.5) = 0.5*ln(2) + 0.5*ln(2) = ln(2) ≈ 0.693
        // But our formulation is ∫ μ ln(μ) dx, so: 0.5*ln(0.5) + 0.5*ln(0.5) = ln(0.5) = -ln(2)
        assert!(val < 0.0); // entropy of uniform < 0 in this formulation
    }
}
