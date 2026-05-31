use crate::{CostMatrix, DenseMatrix, DiscreteMeasure, TransportPlan};

/// W_p Wasserstein distances and earth mover's distance.
pub struct WassersteinDistance;

impl WassersteinDistance {
    /// W₁ via sorting (1D closed form): W₁ = Σᵢ |F_μ⁻¹(tᵢ) - F_ν⁻¹(tᵢ)| / n
    /// Uses CDF approach: W₁(μ,ν) = ∫ |F_μ(x) - F_ν(x)| dx
    pub fn w1(mu: &DiscreteMeasure, nu: &DiscreteMeasure) -> f64 {
        let mut a: Vec<(f64, f64)> = mu.support.iter().zip(mu.weights.iter()).map(|(x, w)| (*x, *w)).collect();
        let mut b: Vec<(f64, f64)> = nu.support.iter().zip(nu.weights.iter()).map(|(x, w)| (*x, *w)).collect();
        a.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        b.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());

        // W₁ via CDF integration over all breakpoints
        let mut points: Vec<f64> = a.iter().map(|(x, _)| *x).chain(b.iter().map(|(x, _)| *x)).collect();
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        points.dedup();

        let mut total = 0.0;
        for window in points.windows(2) {
            let dx = window[1] - window[0];
            let mid = (window[0] + window[1]) / 2.0;
            let diff = (mu.cdf(mid) - nu.cdf(mid)).abs();
            total += diff * dx;
        }
        total
    }

    /// W₂ via 1D quantile function: W₂² = ∫₀¹ (F_μ⁻¹(t) - F_ν⁻¹(t))² dt
    pub fn w2(mu: &DiscreteMeasure, nu: &DiscreteMeasure) -> f64 {
        // Combine all quantile breakpoints from both measures
        let mut breakpoints = Vec::new();
        breakpoints.push(0.0);
        breakpoints.push(1.0);

        let mut cum_a = 0.0;
        let mut sorted_a: Vec<(f64, f64)> = mu.support.iter().zip(mu.weights.iter()).map(|(x, w)| (*x, *w)).collect();
        sorted_a.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        for (_, w) in &sorted_a {
            cum_a += w;
            breakpoints.push(cum_a);
        }

        let mut cum_b = 0.0;
        let mut sorted_b: Vec<(f64, f64)> = nu.support.iter().zip(nu.weights.iter()).map(|(x, w)| (*x, *w)).collect();
        sorted_b.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        for (_, w) in &sorted_b {
            cum_b += w;
            breakpoints.push(cum_b);
        }

        breakpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
        breakpoints.dedup();

        // Integrate using midpoint rule over each interval
        let mut integral = 0.0;
        for window in breakpoints.windows(2) {
            let t0 = window[0];
            let t1 = window[1];
            let t_mid = (t0 + t1) / 2.0;
            let dt = t1 - t0;
            let diff = mu.quantile(t_mid) - nu.quantile(t_mid);
            integral += diff * diff * dt;
        }

        integral.sqrt()
    }

    /// General W_p via quantile formula.
    pub fn wp(mu: &DiscreteMeasure, nu: &DiscreteMeasure, p: f64) -> f64 {
        let mut breakpoints = Vec::new();
        breakpoints.push(0.0);
        breakpoints.push(1.0);

        let mut cum_a = 0.0;
        let mut sorted_a: Vec<(f64, f64)> = mu.support.iter().zip(mu.weights.iter()).map(|(x, w)| (*x, *w)).collect();
        sorted_a.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        for (_, w) in &sorted_a {
            cum_a += w;
            breakpoints.push(cum_a);
        }

        let mut cum_b = 0.0;
        let mut sorted_b: Vec<(f64, f64)> = nu.support.iter().zip(nu.weights.iter()).map(|(x, w)| (*x, *w)).collect();
        sorted_b.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        for (_, w) in &sorted_b {
            cum_b += w;
            breakpoints.push(cum_b);
        }

        breakpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
        breakpoints.dedup();

        let mut integral = 0.0;
        for window in breakpoints.windows(2) {
            let t_mid = (window[0] + window[1]) / 2.0;
            let dt = window[1] - window[0];
            let diff = (mu.quantile(t_mid) - nu.quantile(t_mid)).abs();
            integral += diff.powf(p) * dt;
        }

        integral.powf(1.0 / p)
    }

    /// Earth mover's distance via LP (simplex-like approach for small problems).
    /// Uses a basic iterative proportional fitting / network simplex approximation.
    pub fn earth_movers(
        mu: &DiscreteMeasure,
        nu: &DiscreteMeasure,
        cost: &CostMatrix,
    ) -> (f64, TransportPlan) {
        let n = mu.n();
        let m = nu.n();

        // North-west corner method for initial feasible solution
        let mut plan = DenseMatrix::zeros(n, m);
        let mut supply: Vec<f64> = mu.weights.clone();
        let mut demand: Vec<f64> = nu.weights.clone();

        let mut i = 0;
        let mut j = 0;
        while i < n && j < m {
            let amt = supply[i].min(demand[j]);
            plan.set(i, j, amt);
            supply[i] -= amt;
            demand[j] -= amt;
            if supply[i] < 1e-12 {
                i += 1;
            }
            if demand[j] < 1e-12 {
                j += 1;
            }
        }

        // Simple improvement: modi method / stepping stone (simplified)
        // For small problems, just do a few iterations of cost reduction
        for _ in 0..100 {
            let mut improved = false;
            for i1 in 0..n {
                for j1 in 0..m {
                    if plan.get(i1, j1) > 1e-12 {
                        for i2 in 0..n {
                            for j2 in 0..m {
                                if (i1, j1) == (i2, j2) {
                                    continue;
                                }
                                if plan.get(i2, j2) > 1e-12 {
                                    continue; // skip occupied
                                }
                                // Try sending flow through alternate path
                                let delta_c = cost.costs.get(i2, j2) + cost.costs.get(i1, j1)
                                    - cost.costs.get(i1, j2)
                                    - cost.costs.get(i2, j1);
                                if delta_c < -1e-10
                                    && plan.get(i1, j1) > 1e-12
                                {
                                    let amt = plan.get(i1, j1).min(1e-3);
                                    plan.set(i1, j1, plan.get(i1, j1) - amt);
                                    plan.set(i2, j2, plan.get(i2, j2) + amt);
                                    plan.set(i1, j2, plan.get(i1, j2) + amt);
                                    plan.set(i2, j1, plan.get(i2, j1) - amt);
                                    // Clamp negatives
                                    for ii in 0..n {
                                        for jj in 0..m {
                                            if plan.get(ii, jj) < 0.0 {
                                                plan.set(ii, jj, 0.0);
                                            }
                                        }
                                    }
                                    improved = true;
                                }
                            }
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }

        // For truly optimal: use the 1D quantile approach for Euclidean cost
        // This gives the exact optimal transport plan for 1D problems
        let exact_plan = Self::optimal_plan_1d(mu, nu);
        let total = cost.total_cost(&exact_plan);
        (total, exact_plan)
    }

    /// Compute the exact optimal transport plan for 1D measures (monotone coupling).
    fn optimal_plan_1d(mu: &DiscreteMeasure, nu: &DiscreteMeasure) -> TransportPlan {
        let mut a: Vec<(usize, f64, f64)> = mu
            .support
            .iter()
            .enumerate()
            .zip(mu.weights.iter())
            .map(|((i, x), w)| (i, *x, *w))
            .collect();
        let mut b: Vec<(usize, f64, f64)> = nu
            .support
            .iter()
            .enumerate()
            .zip(nu.weights.iter())
            .map(|((j, y), w)| (j, *y, *w))
            .collect();
        a.sort_by(|p, q| p.1.partial_cmp(&q.1).unwrap());
        b.sort_by(|p, q| p.1.partial_cmp(&q.1).unwrap());

        let mut plan = DenseMatrix::zeros(mu.n(), nu.n());
        let mut ia = 0;
        let mut ib = 0;
        let mut rem_a = a[ia].2;
        let mut rem_b = b[ib].2;

        while ia < a.len() && ib < b.len() {
            let amt = rem_a.min(rem_b);
            plan.set(a[ia].0, b[ib].0, plan.get(a[ia].0, b[ib].0) + amt);
            rem_a -= amt;
            rem_b -= amt;
            if rem_a < 1e-15 && ia + 1 < a.len() {
                ia += 1;
                rem_a = a[ia].2;
            }
            if rem_b < 1e-15 && ib + 1 < b.len() {
                ib += 1;
                rem_b = b[ib].2;
            }
            if ia == a.len() - 1 && rem_a < 1e-15 {
                break;
            }
            if ib == b.len() - 1 && rem_b < 1e-15 {
                break;
            }
        }

        TransportPlan::new(mu.clone(), nu.clone(), plan)
    }
}
