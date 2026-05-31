use serde::{Deserialize, Serialize};

use crate::{DiscreteMeasure, WassersteinDistance};

/// An agent point in the distribution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentPoint {
    pub id: String,
    pub position: f64,
    pub capability: f64,
}

/// Agents as a probability distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDistribution {
    pub agents: Vec<AgentPoint>,
    pub measure: DiscreteMeasure,
}

impl AgentDistribution {
    pub fn new(agents: Vec<AgentPoint>) -> Self {
        let n = agents.len();
        let support: Vec<f64> = agents.iter().map(|a| a.position).collect();
        let weights = vec![1.0 / n as f64; n];
        let measure = DiscreteMeasure::new(support, weights);
        Self { agents, measure }
    }

    /// Shift the distribution by amount in the given direction.
    pub fn shift(&mut self, direction: f64, amount: f64) {
        for agent in &mut self.agents {
            agent.position += direction * amount;
        }
        for s in &mut self.measure.support {
            *s += direction * amount;
        }
    }

    /// Increase variance by scaling positions outward from the mean.
    pub fn spread(&mut self, amount: f64) {
        let mean = self.measure.mean();
        for agent in &mut self.agents {
            agent.position = mean + (agent.position - mean) * (1.0 + amount);
        }
        for s in &mut self.measure.support {
            *s = mean + (*s - mean) * (1.0 + amount);
        }
    }

    /// Decrease variance by scaling positions inward toward the mean.
    pub fn concentrate(&mut self, amount: f64) {
        let mean = self.measure.mean();
        let factor = (1.0 - amount).max(0.0);
        for agent in &mut self.agents {
            agent.position = mean + (agent.position - mean) * factor;
        }
        for s in &mut self.measure.support {
            *s = mean + (*s - mean) * factor;
        }
    }

    /// W₂ distance to another agent distribution.
    pub fn w_distance(&self, other: &AgentDistribution) -> f64 {
        WassersteinDistance::w2(&self.measure, &other.measure)
    }

    /// Interpolant barycenter with another distribution.
    pub fn barycenter_with(&self, other: &AgentDistribution, alpha: f64) -> AgentDistribution {
        let n = self.agents.len();
        let mut new_agents = Vec::with_capacity(n);
        let mut new_support = Vec::with_capacity(n);

        // Simple linear interpolation of positions
        for i in 0..n {
            let pos = (1.0 - alpha) * self.agents[i].position + alpha * other.agents.get(i).map(|a| a.position).unwrap_or(self.agents[i].position);
            new_support.push(pos);
            new_agents.push(AgentPoint {
                id: format!("interp_{}", i),
                position: pos,
                capability: (1.0 - alpha) * self.agents[i].capability
                    + alpha * other.agents.get(i).map(|a| a.capability).unwrap_or(self.agents[i].capability),
            });
        }

        let weights = vec![1.0 / n as f64; n];
        let measure = DiscreteMeasure::new(new_support, weights);
        AgentDistribution { agents: new_agents, measure }
    }
}
