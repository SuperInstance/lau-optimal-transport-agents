use serde::{Deserialize, Serialize};
use std::fmt;

/// Dense row-major matrix.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl DenseMatrix {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i][i] = 1.0;
        }
        m
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i][j] = v;
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        self.data[i].iter().sum()
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| self.data[i][j]).sum()
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i]
    }

    pub fn multiply(&self, other: &DenseMatrix) -> DenseMatrix {
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0;
                for k in 0..self.cols {
                    s += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = s;
            }
        }
        result
    }

    pub fn transpose(&self) -> DenseMatrix {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }
}

impl fmt::Display for DenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            for (j, v) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:.6}", v)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
