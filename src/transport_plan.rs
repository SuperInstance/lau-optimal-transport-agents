use serde::{Deserialize, Serialize};

use crate::{DenseMatrix, DiscreteMeasure};

/// The ground cost matrix c(xᵢ, yⱼ).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CostMatrix {
    pub costs: DenseMatrix,
    source_support: Vec<f64>,
    target_support: Vec<f64>,
}

impl CostMatrix {
    /// c(x,y) = |x - y|
    pub fn from_euclidean(source: &[f64], target: &[f64]) -> Self {
        let mut data = Vec::with_capacity(source.len());
        for x in source {
            let row: Vec<f64> = target.iter().map(|y| (x - y).abs()).collect();
            data.push(row);
        }
        Self {
            costs: DenseMatrix::new(data),
            source_support: source.to_vec(),
            target_support: target.to_vec(),
        }
    }

    /// c(x,y) = |x - y|²
    pub fn from_euclidean_squared(source: &[f64], target: &[f64]) -> Self {
        let mut data = Vec::with_capacity(source.len());
        for x in source {
            let row: Vec<f64> = target.iter().map(|y| (x - y).powi(2)).collect();
            data.push(row);
        }
        Self {
            costs: DenseMatrix::new(data),
            source_support: source.to_vec(),
            target_support: target.to_vec(),
        }
    }

    pub fn from_custom(source: &[f64], target: &[f64], f: &dyn Fn(f64, f64) -> f64) -> Self {
        let mut data = Vec::with_capacity(source.len());
        for x in source {
            let row: Vec<f64> = target.iter().map(|y| f(*x, *y)).collect();
            data.push(row);
        }
        Self {
            costs: DenseMatrix::new(data),
            source_support: source.to_vec(),
            target_support: target.to_vec(),
        }
    }

    pub fn total_cost(&self, plan: &TransportPlan) -> f64 {
        let mut total = 0.0;
        for i in 0..self.costs.rows() {
            for j in 0..self.costs.cols() {
                total += self.costs.get(i, j) * plan.plan.get(i, j);
            }
        }
        total
    }

    pub fn source_support(&self) -> &[f64] {
        &self.source_support
    }

    pub fn target_support(&self) -> &[f64] {
        &self.target_support
    }
}

/// An optimal transport plan γ ∈ Π(μ, ν).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportPlan {
    pub source: DiscreteMeasure,
    pub target: DiscreteMeasure,
    pub plan: DenseMatrix,
}

impl TransportPlan {
    pub fn new(source: DiscreteMeasure, target: DiscreteMeasure, plan: DenseMatrix) -> Self {
        assert_eq!(plan.rows(), source.n());
        assert_eq!(plan.cols(), target.n());
        Self { source, target, plan }
    }

    /// Σᵢⱼ γᵢⱼ cᵢⱼ
    pub fn total_cost(&self, cost: &CostMatrix) -> f64 {
        cost.total_cost(self)
    }

    /// Row sums = source weights, col sums = target weights.
    pub fn marginals_valid(&self, tolerance: f64) -> bool {
        for i in 0..self.plan.rows() {
            if (self.plan.row_sum(i) - self.source.weights[i]).abs() > tolerance {
                return false;
            }
        }
        for j in 0..self.plan.cols() {
            if (self.plan.col_sum(j) - self.target.weights[j]).abs() > tolerance {
                return false;
            }
        }
        true
    }

    pub fn total_mass(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.plan.rows() {
            for j in 0..self.plan.cols() {
                total += self.plan.get(i, j);
            }
        }
        total
    }
}
