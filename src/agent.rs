//! Agent belief representation and history tracking.
//!
//! An agent's belief is a probability distribution over states.
//! This module provides types for tracking how beliefs evolve over time.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::convergence::belief_shift;
use crate::mccann::{mccann_interpolation, McCannCurve};
use crate::wasserstein::{wasserstein_1, wasserstein_2};

/// An agent's belief as a discrete probability distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBelief {
    /// Probability weights (must sum to 1).
    pub weights: Vec<f64>,
    /// Support points (the states the agent can be in).
    pub support: Vec<f64>,
    /// Name/label for this belief.
    pub label: String,
}

impl AgentBelief {
    /// Create a new belief distribution.
    pub fn new(weights: Vec<f64>, support: Vec<f64>, label: &str) -> Self {
        // Normalize
        let sum: f64 = weights.iter().sum();
        let weights = if sum > 0.0 {
            weights.iter().map(|w| w / sum).collect()
        } else {
            weights
        };
        Self {
            weights,
            support,
            label: label.to_string(),
        }
    }

    /// Create a uniform belief over n states.
    pub fn uniform(n: usize, support: Vec<f64>, label: &str) -> Self {
        let w = 1.0 / n as f64;
        Self {
            weights: vec![w; n],
            support,
            label: label.to_string(),
        }
    }

    /// Create a Dirac (point mass) belief.
    pub fn dirac(point: f64, support: Vec<f64>, index: usize) -> Self {
        let mut weights = vec![0.0; support.len()];
        if index < weights.len() {
            weights[index] = 1.0;
        }
        Self {
            weights,
            support,
            label: format!("dirac_{}", point),
        }
    }

    /// Get the DVector representation.
    pub fn to_dvector(&self) -> DVector<f64> {
        DVector::from_vec(self.weights.clone())
    }

    /// Wasserstein-1 distance to another belief.
    pub fn w1_to(&self, other: &AgentBelief) -> f64 {
        let mu = self.to_dvector();
        let nu = other.to_dvector();
        wasserstein_1(&mu, &nu, &self.support, &other.support)
    }

    /// Wasserstein-2 distance to another belief.
    pub fn w2_to(&self, other: &AgentBelief) -> f64 {
        let mu = self.to_dvector();
        let nu = other.to_dvector();
        wasserstein_2(&mu, &nu, &self.support, &other.support)
    }

    /// Compute McCann interpolation to another belief.
    pub fn interpolate_to(&self, other: &AgentBelief) -> McCannCurve {
        let mu = self.to_dvector();
        let nu = other.to_dvector();
        mccann_interpolation(&mu, &nu, &self.support, &other.support)
    }

    /// Mean of the belief distribution.
    pub fn mean(&self) -> f64 {
        self.weights
            .iter()
            .zip(self.support.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
    }

    /// Variance of the belief distribution.
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.weights
            .iter()
            .zip(self.support.iter())
            .map(|(w, x)| w * (x - m).powi(2))
            .sum()
    }

    /// Entropy of the belief.
    pub fn entropy(&self) -> f64 {
        -self
            .weights
            .iter()
            .map(|&w| if w > 1e-15 { w * w.ln() } else { 0.0 })
            .sum::<f64>()
    }

    /// Total mass (should be 1.0 for valid beliefs).
    pub fn total_mass(&self) -> f64 {
        self.weights.iter().sum()
    }
}

/// History of an agent's beliefs over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefHistory {
    /// The agent's name/identifier.
    pub agent_name: String,
    /// Chronological list of beliefs.
    pub history: Vec<AgentBelief>,
}

impl BeliefHistory {
    /// Create a new belief history for an agent.
    pub fn new(agent_name: &str) -> Self {
        Self {
            agent_name: agent_name.to_string(),
            history: Vec::new(),
        }
    }

    /// Record a new belief.
    pub fn record(&mut self, belief: AgentBelief) {
        self.history.push(belief);
    }

    /// Number of recorded beliefs.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get the latest belief.
    pub fn latest(&self) -> Option<&AgentBelief> {
        self.history.last()
    }

    /// Compute W1 distances between consecutive beliefs.
    pub fn consecutive_w1(&self) -> Vec<f64> {
        let mut distances = Vec::new();
        for i in 1..self.history.len() {
            distances.push(self.history[i - 1].w1_to(&self.history[i]));
        }
        distances
    }

    /// Compute W2 distances between consecutive beliefs.
    pub fn consecutive_w2(&self) -> Vec<f64> {
        let mut distances = Vec::new();
        for i in 1..self.history.len() {
            distances.push(self.history[i - 1].w2_to(&self.history[i]));
        }
        distances
    }

    /// Total belief shift (sum of consecutive W1 distances).
    pub fn total_shift(&self) -> f64 {
        self.consecutive_w1().iter().sum()
    }

    /// Wasserstein distance from initial to final belief.
    pub fn total_displacement(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        self.history.first().unwrap().w1_to(self.history.last().unwrap())
    }

    /// Compute the convergence analysis.
    pub fn convergence_analysis(&self) -> Option<crate::convergence::ConvergenceResult> {
        if self.history.is_empty() {
            return None;
        }
        let beliefs: Vec<Vec<f64>> = self.history.iter().map(|b| b.weights.clone()).collect();
        let support = self.history[0].support.clone();
        let target = self.history.last().map(|b| b.weights.clone());
        Some(belief_shift(&beliefs, &support, target.as_ref()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_agent_belief_creation() {
        let b = AgentBelief::new(vec![0.3, 0.7], vec![0.0, 1.0], "test");
        assert_eq!(b.weights.len(), 2);
        assert_relative_eq!(b.total_mass(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_belief_normalizes() {
        let b = AgentBelief::new(vec![3.0, 7.0], vec![0.0, 1.0], "test");
        assert_relative_eq!(b.weights[0], 0.3, epsilon = 1e-10);
        assert_relative_eq!(b.weights[1], 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_belief() {
        let b = AgentBelief::uniform(3, vec![0.0, 1.0, 2.0], "uniform");
        for &w in &b.weights {
            assert_relative_eq!(w, 1.0 / 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dirac_belief() {
        let b = AgentBelief::dirac(1.0, vec![0.0, 1.0, 2.0], 1);
        assert_relative_eq!(b.weights[0], 0.0);
        assert_relative_eq!(b.weights[1], 1.0);
        assert_relative_eq!(b.weights[2], 0.0);
    }

    #[test]
    fn test_belief_mean() {
        let b = AgentBelief::new(vec![0.3, 0.7], vec![0.0, 1.0], "test");
        assert_relative_eq!(b.mean(), 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_belief_variance() {
        let b = AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "test");
        assert_relative_eq!(b.variance(), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_belief_entropy() {
        let b = AgentBelief::uniform(2, vec![0.0, 1.0], "uniform");
        let h = b.entropy();
        assert_relative_eq!(h, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_w1_identical_beliefs() {
        let b1 = AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b1");
        let w1 = b1.w1_to(&b1);
        assert_relative_eq!(w1, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_w2_identical_beliefs() {
        let b1 = AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b1");
        let w2 = b1.w2_to(&b1);
        assert_relative_eq!(w2, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_belief_interpolation() {
        let b1 = AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b1");
        let b2 = AgentBelief::new(vec![0.5, 0.5], vec![1.0, 2.0], "b2");
        let curve = b1.interpolate_to(&b2);
        assert!(curve.wasserstein_distance > 0.0);
    }

    #[test]
    fn test_belief_history_record() {
        let mut h = BeliefHistory::new("agent1");
        h.record(AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b0"));
        h.record(AgentBelief::new(vec![0.6, 0.4], vec![0.0, 1.0], "b1"));
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_belief_history_consecutive_w1() {
        let mut h = BeliefHistory::new("agent1");
        h.record(AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b0"));
        h.record(AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b1"));
        let dists = h.consecutive_w1();
        assert_eq!(dists.len(), 1);
        assert_relative_eq!(dists[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_belief_history_total_shift() {
        let mut h = BeliefHistory::new("agent1");
        h.record(AgentBelief::new(vec![1.0, 0.0], vec![0.0, 1.0], "b0"));
        h.record(AgentBelief::new(vec![0.0, 1.0], vec![0.0, 1.0], "b1"));
        assert!(h.total_shift() > 0.5);
    }

    #[test]
    fn test_belief_history_displacement() {
        let mut h = BeliefHistory::new("agent1");
        h.record(AgentBelief::new(vec![1.0, 0.0], vec![0.0, 1.0], "b0"));
        h.record(AgentBelief::new(vec![0.0, 1.0], vec![0.0, 1.0], "b1"));
        let d = h.total_displacement();
        assert!(d > 0.5);
    }

    #[test]
    fn test_belief_history_latest() {
        let mut h = BeliefHistory::new("agent1");
        assert!(h.latest().is_none());
        h.record(AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b0"));
        assert_eq!(h.latest().unwrap().label, "b0");
    }

    #[test]
    fn test_belief_serialization() {
        let b = AgentBelief::new(vec![0.3, 0.7], vec![0.0, 1.0], "test");
        let json = serde_json::to_string(&b).unwrap();
        let b2: AgentBelief = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(b2.weights[0], 0.3);
        assert_relative_eq!(b2.weights[1], 0.7);
    }

    #[test]
    fn test_history_serialization() {
        let mut h = BeliefHistory::new("agent1");
        h.record(AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b0"));
        let json = serde_json::to_string(&h).unwrap();
        let h2: BeliefHistory = serde_json::from_str(&json).unwrap();
        assert_eq!(h2.len(), 1);
    }

    #[test]
    fn test_convergence_analysis() {
        let mut h = BeliefHistory::new("agent1");
        h.record(AgentBelief::new(vec![1.0, 0.0], vec![0.0, 1.0], "b0"));
        h.record(AgentBelief::new(vec![0.7, 0.3], vec![0.0, 1.0], "b1"));
        h.record(AgentBelief::new(vec![0.5, 0.5], vec![0.0, 1.0], "b2"));
        let result = h.convergence_analysis().unwrap();
        assert!(result.distances.len() > 0);
    }

    #[test]
    fn test_history_empty_analysis() {
        let h = BeliefHistory::new("agent1");
        assert!(h.convergence_analysis().is_none());
    }
}
