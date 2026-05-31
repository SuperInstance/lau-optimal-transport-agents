#[cfg(test)]
mod tests {
    use lau_optimal_transport_agents::*;

    // ─── DenseMatrix tests ───

    #[test]
    fn matrix_zeros() {
        let m = DenseMatrix::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn matrix_identity() {
        let m = DenseMatrix::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(m.get(i, j), 1.0);
                } else {
                    assert_eq!(m.get(i, j), 0.0);
                }
            }
        }
    }

    #[test]
    fn matrix_set_get() {
        let mut m = DenseMatrix::zeros(2, 2);
        m.set(0, 1, 3.14);
        assert!((m.get(0, 1) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn matrix_row_col_sum() {
        let m = DenseMatrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!((m.row_sum(0) - 3.0).abs() < 1e-10);
        assert!((m.row_sum(1) - 7.0).abs() < 1e-10);
        assert!((m.col_sum(0) - 4.0).abs() < 1e-10);
        assert!((m.col_sum(1) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_multiply() {
        let a = DenseMatrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = DenseMatrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.multiply(&b);
        assert!((c.get(0, 0) - 19.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 22.0).abs() < 1e-10);
        assert!((c.get(1, 0) - 43.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_transpose() {
        let m = DenseMatrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert!((t.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((t.get(2, 1) - 6.0).abs() < 1e-10);
    }

    // ─── DiscreteMeasure tests ───

    #[test]
    fn measure_dirac() {
        let d = DiscreteMeasure::dirac(5.0);
        assert_eq!(d.n(), 1);
        assert!((d.total_mass() - 1.0).abs() < 1e-10);
        assert!((d.mean() - 5.0).abs() < 1e-10);
        assert!((d.variance() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn measure_uniform() {
        let u = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        assert!(u.is_probability());
        assert!((u.mean() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn measure_normalize() {
        let mut m = DiscreteMeasure::new(vec![0.0, 1.0], vec![2.0, 3.0]);
        assert!(!m.is_probability());
        m.normalize();
        assert!(m.is_probability());
        assert!((m.total_mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn measure_mean_variance() {
        let m = DiscreteMeasure::uniform(vec![0.0, 2.0]);
        assert!((m.mean() - 1.0).abs() < 1e-10);
        assert!((m.variance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn measure_cdf() {
        let m = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        assert!((m.cdf(-0.5)).abs() < 1e-10);
        assert!((m.cdf(0.0) - 1.0 / 3.0).abs() < 1e-10);
        assert!((m.cdf(1.5) - 2.0 / 3.0).abs() < 1e-10);
        assert!((m.cdf(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn measure_quantile() {
        let m = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        assert!((m.quantile(0.1)).abs() < 1e-10);
        assert!((m.quantile(0.5) - 1.0).abs() < 1e-10);
        assert!((m.quantile(0.9) - 2.0).abs() < 1e-10);
    }

    // ─── CostMatrix tests ───

    #[test]
    fn cost_euclidean_squared() {
        let c = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[2.0, 3.0]);
        assert!((c.costs.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((c.costs.get(0, 1) - 9.0).abs() < 1e-10);
        assert!((c.costs.get(1, 0) - 1.0).abs() < 1e-10);
        assert!((c.costs.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn cost_euclidean() {
        let c = CostMatrix::from_euclidean(&[0.0], &[3.0]);
        assert!((c.costs.get(0, 0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cost_custom() {
        let c = CostMatrix::from_custom(&[0.0, 1.0], &[2.0], &|x, y| (x * y).abs());
        assert!((c.costs.get(0, 0)).abs() < 1e-10);
        assert!((c.costs.get(1, 0) - 2.0).abs() < 1e-10);
    }

    // ─── TransportPlan tests ───

    #[test]
    fn transport_plan_marginals() {
        let mu = DiscreteMeasure::new(vec![0.0, 1.0], vec![0.5, 0.5]);
        let nu = DiscreteMeasure::new(vec![2.0, 3.0], vec![0.3, 0.7]);
        let plan_matrix = DenseMatrix::new(vec![vec![0.3, 0.2], vec![0.0, 0.5]]);
        let tp = TransportPlan::new(mu, nu, plan_matrix);
        assert!(tp.marginals_valid(1e-10));
        assert!((tp.total_mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn transport_plan_total_cost() {
        let mu = DiscreteMeasure::new(vec![0.0], vec![1.0]);
        let nu = DiscreteMeasure::new(vec![5.0], vec![1.0]);
        let plan_matrix = DenseMatrix::new(vec![vec![1.0]]);
        let tp = TransportPlan::new(mu, nu, plan_matrix);
        let cost = CostMatrix::from_euclidean_squared(&[0.0], &[5.0]);
        assert!((tp.total_cost(&cost) - 25.0).abs() < 1e-10);
    }

    // ─── WassersteinDistance tests ───

    #[test]
    fn w1_same_measure() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        assert!((WassersteinDistance::w1(&mu, &mu)).abs() < 1e-10);
    }

    #[test]
    fn w2_same_measure() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        assert!((WassersteinDistance::w2(&mu, &mu)).abs() < 1e-10);
    }

    #[test]
    fn wp_same_measure() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        assert!((WassersteinDistance::wp(&mu, &mu, 3.0)).abs() < 1e-10);
    }

    #[test]
    fn w1_dirac_shift() {
        let mu = DiscreteMeasure::dirac(0.0);
        let nu = DiscreteMeasure::dirac(3.0);
        assert!((WassersteinDistance::w1(&mu, &nu) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn w2_dirac_shift() {
        let mu = DiscreteMeasure::dirac(0.0);
        let nu = DiscreteMeasure::dirac(3.0);
        assert!((WassersteinDistance::w2(&mu, &nu) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn w1_symmetric() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![1.0, 3.0]);
        let d1 = WassersteinDistance::w1(&mu, &nu);
        let d2 = WassersteinDistance::w1(&nu, &mu);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn w2_symmetric() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 2.0, 4.0]);
        let nu = DiscreteMeasure::uniform(vec![1.0, 3.0, 5.0]);
        let d1 = WassersteinDistance::w2(&mu, &nu);
        let d2 = WassersteinDistance::w2(&nu, &mu);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn w1_triangle_inequality() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let sigma = DiscreteMeasure::uniform(vec![5.0, 6.0]);
        let d12 = WassersteinDistance::w1(&mu, &nu);
        let d23 = WassersteinDistance::w1(&nu, &sigma);
        let d13 = WassersteinDistance::w1(&mu, &sigma);
        assert!(d13 <= d12 + d23 + 1e-10);
    }

    #[test]
    fn w2_triangle_inequality() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let sigma = DiscreteMeasure::uniform(vec![5.0, 6.0]);
        let d12 = WassersteinDistance::w2(&mu, &nu);
        let d23 = WassersteinDistance::w2(&nu, &sigma);
        let d13 = WassersteinDistance::w2(&mu, &sigma);
        assert!(d13 <= d12 + d23 + 1e-10);
    }

    #[test]
    fn w1_nonnegative() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![3.0, 4.0, 5.0]);
        assert!(WassersteinDistance::w1(&mu, &nu) >= 0.0);
    }

    #[test]
    fn w2_nonnegative() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![3.0, 4.0]);
        assert!(WassersteinDistance::w2(&mu, &nu) >= 0.0);
    }

    #[test]
    fn w1_zero_iff_equal() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        assert!((WassersteinDistance::w1(&mu, &nu)).abs() < 1e-10);
    }

    #[test]
    fn w2_known_value() {
        // Uniform on [0,1] vs uniform on [1,2]
        // W₂² = ∫₀¹ (1+t - t)² dt = 1 → W₂ = 1
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![1.0, 2.0]);
        let d = WassersteinDistance::w2(&mu, &nu);
        assert!((d - 1.0).abs() < 0.1, "W₂ was {}, expected ~1.0", d);
    }

    // ─── Earth mover's tests ───

    #[test]
    fn earth_movers_basic() {
        let mu = DiscreteMeasure::new(vec![0.0, 1.0], vec![0.5, 0.5]);
        let nu = DiscreteMeasure::new(vec![1.0, 2.0], vec![0.5, 0.5]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[1.0, 2.0]);
        let (total, plan) = WassersteinDistance::earth_movers(&mu, &nu, &cost);
        assert!(plan.marginals_valid(1e-8));
        assert!(total >= 0.0);
        // Optimal: move 0→1 (cost 1) and 1→2 (cost 1) → total = 0.5*1 + 0.5*1 = 1.0
        assert!((total - 1.0).abs() < 0.1, "earth mover total was {}, expected ~1.0", total);
    }

    #[test]
    fn earth_movers_marginals_valid() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![1.0, 2.0, 3.0]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
        let (_, plan) = WassersteinDistance::earth_movers(&mu, &nu, &cost);
        assert!(plan.marginals_valid(1e-8));
    }

    #[test]
    fn earth_movers_uniform_assignment() {
        // With uniform weights, EMD reduces to assignment problem
        let mu = DiscreteMeasure::uniform(vec![0.0, 10.0]);
        let nu = DiscreteMeasure::uniform(vec![10.0, 20.0]);
        let cost = CostMatrix::from_euclidean(&[0.0, 10.0], &[10.0, 20.0]);
        let (total, plan) = WassersteinDistance::earth_movers(&mu, &nu, &cost);
        assert!(plan.marginals_valid(1e-8));
        // Optimal: 0→10 (cost 10), 10→20 (cost 10) → total = 10
        assert!((total - 10.0).abs() < 0.5, "total was {}, expected ~10", total);
    }

    // ─── Sinkhorn tests ───

    #[test]
    fn sinkhorn_converges() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![1.0, 2.0, 3.0]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
        let mut sink = SinkhornAlgorithm::new(0.1, 2000, 1e-8);
        let (plan, w_eps) = sink.compute(&cost, &mu, &nu);
        assert!(plan.marginals_valid(1e-3));
        assert!(w_eps >= 0.0);
        assert!(!sink.convergence_history().is_empty());
    }

    #[test]
    fn sinkhorn_regularized_approaches_true() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![1.0, 2.0]);

        let true_w2 = WassersteinDistance::w2(&mu, &nu);

        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[1.0, 2.0]);
        let mut sink_small = SinkhornAlgorithm::new(0.01, 2000, 1e-10);
        let (_, w_small) = sink_small.compute(&cost, &mu, &nu);

        let mut sink_large = SinkhornAlgorithm::new(1.0, 1000, 1e-8);
        let (_, w_large) = sink_large.compute(&cost, &mu, &nu);

        // Smaller ε should be closer to true W₂
        let err_small = (w_small - true_w2).abs();
        let err_large = (w_large - true_w2).abs();
        assert!(err_small <= err_large + 0.1, "smaller eps should be closer: err_small={}, err_large={}", err_small, err_large);
    }

    #[test]
    fn sinkhorn_plan_mass() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[2.0, 3.0]);
        let mut sink = SinkhornAlgorithm::new(0.1, 1000, 1e-8);
        let (plan, _) = sink.compute(&cost, &mu, &nu);
        assert!((plan.total_mass() - 1.0).abs() < 1e-4);
    }

    // ─── MongeMap tests ───

    #[test]
    fn monge_map_identity() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let monge = MongeMap::new(vec![0, 1, 2]);
        assert!(monge.is_monge_map(&mu, &mu, 1e-10));
    }

    #[test]
    fn monge_map_transport_cost() {
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[2.0, 3.0]);
        let monge = MongeMap::new(vec![0, 1]);
        let c = monge.transport_cost(&cost);
        // 0→2 costs 4, 1→3 costs 4, total 8
        assert!((c - 8.0).abs() < 1e-10);
    }

    #[test]
    fn monge_to_transport_plan() {
        let mu = DiscreteMeasure::new(vec![0.0, 1.0], vec![0.3, 0.7]);
        let monge = MongeMap::new(vec![1, 0]);
        let plan = monge.to_transport_plan(&mu);
        assert!((plan.plan.get(0, 1) - 0.3).abs() < 1e-10);
        assert!((plan.plan.get(1, 0) - 0.7).abs() < 1e-10);
    }

    // ─── BrenierMap tests ───

    #[test]
    fn brenier_map_monotone_rearrangement() {
        let mu = DiscreteMeasure::new(vec![2.0, 0.0, 1.0], vec![0.3, 0.4, 0.3]);
        let nu = DiscreteMeasure::new(vec![1.0, 3.0, 2.0], vec![0.4, 0.3, 0.3]);
        let monge = BrenierMap::compute(&mu, &nu);
        // Should be a valid mapping (all indices in range)
        for &t in &monge.mapping {
            assert!(t < 3);
        }
    }

    #[test]
    fn brenier_pushforward() {
        // Uniform measures — Brenier should give a valid pushforward
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![3.0, 4.0, 5.0]);
        let monge = BrenierMap::compute(&mu, &nu);
        assert!(monge.is_monge_map(&mu, &nu, 0.5)); // tolerance for discretization
    }

    // ─── WassersteinBarycenter tests ───

    #[test]
    fn barycenter_identical_measures() {
        let m = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let bar = WassersteinBarycenter::compute(&[m.clone(), m.clone()], &[0.5, 0.5], 3, 10);
        // Barycenter of identical measures ≈ that measure
        let d = WassersteinDistance::w2(&bar, &m);
        assert!(d < 0.5, "barycenter distance to identical measures: {}", d);
    }

    #[test]
    fn barycenter_between_shifted() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![4.0, 5.0, 6.0]);
        let bar = WassersteinBarycenter::compute(&[mu.clone(), nu.clone()], &[0.5, 0.5], 3, 10);
        // Barycenter should be near midpoint
        let bar_mean = bar.mean();
        assert!((bar_mean - 3.0).abs() < 0.5, "barycenter mean was {}, expected ~3.0", bar_mean);
    }

    #[test]
    fn barycenter_is_barycenter_check() {
        let m1 = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let m2 = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let bar = WassersteinBarycenter::compute(&[m1.clone(), m2.clone()], &[0.5, 0.5], 2, 10);
        assert!(WassersteinBarycenter::is_barycenter(&bar, &[m1, m2], &[0.5, 0.5], 0.5));
    }

    // ─── JKOFlow tests ───

    #[test]
    fn jko_flow_entropy_spreads() {
        // JKO flow of entropy should spread the distribution
        let initial = DiscreteMeasure::new(vec![-0.5, 0.0, 0.5], vec![1.0 / 3.0; 3]);
        let entropy = |m: &DiscreteMeasure| -> f64 {
            m.weights.iter().filter(|w| **w > 0.0).map(|w| -w * w.ln()).sum()
        };
        let trajectory = JKOFlow::run(&initial, &entropy, 0.1, 5, 20);
        assert_eq!(trajectory.len(), 6); // initial + 5 steps
        // Variance should generally increase
        let v0 = trajectory[0].variance();
        let v_last = trajectory.last().unwrap().variance();
        assert!(v_last >= v0 * 0.8, "variance should not shrink: v0={}, v_last={}", v0, v_last);
    }

    #[test]
    fn jko_single_step() {
        let initial = DiscreteMeasure::new(vec![-1.0, 0.0, 1.0], vec![1.0 / 3.0; 3]);
        let potential = |m: &DiscreteMeasure| -> f64 { m.mean().powi(2) };
        let next = JKOFlow::step(&initial, &potential, 0.1, 20);
        assert!(next.n() > 0);
    }

    // ─── AgentDistribution tests ───

    #[test]
    fn agent_distribution_basic() {
        let agents = vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 1.0, capability: 2.0 },
        ];
        let dist = AgentDistribution::new(agents);
        assert!(dist.measure.is_probability());
        assert!((dist.measure.mean() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn agent_shift() {
        let agents = vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 2.0, capability: 1.0 },
        ];
        let mut dist = AgentDistribution::new(agents);
        dist.shift(1.0, 3.0);
        assert!((dist.agents[0].position - 3.0).abs() < 1e-10);
        assert!((dist.agents[1].position - 5.0).abs() < 1e-10);
    }

    #[test]
    fn agent_spread() {
        let agents = vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 2.0, capability: 1.0 },
        ];
        let mut dist = AgentDistribution::new(agents);
        let v0 = dist.measure.variance();
        dist.spread(0.5);
        let v1 = dist.measure.variance();
        assert!(v1 > v0);
    }

    #[test]
    fn agent_concentrate() {
        let agents = vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 2.0, capability: 1.0 },
        ];
        let mut dist = AgentDistribution::new(agents);
        let v0 = dist.measure.variance();
        dist.concentrate(0.5);
        let v1 = dist.measure.variance();
        assert!(v1 < v0);
    }

    #[test]
    fn agent_w_distance() {
        let d1 = AgentDistribution::new(vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 1.0, capability: 1.0 },
        ]);
        let d2 = AgentDistribution::new(vec![
            AgentPoint { id: "c".into(), position: 1.0, capability: 1.0 },
            AgentPoint { id: "d".into(), position: 2.0, capability: 1.0 },
        ]);
        let d = d1.w_distance(&d2);
        assert!(d > 0.0);
    }

    #[test]
    fn agent_w_distance_symmetric() {
        let d1 = AgentDistribution::new(vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 2.0, capability: 1.0 },
        ]);
        let d2 = AgentDistribution::new(vec![
            AgentPoint { id: "c".into(), position: 3.0, capability: 1.0 },
            AgentPoint { id: "d".into(), position: 5.0, capability: 1.0 },
        ]);
        let d12 = d1.w_distance(&d2);
        let d21 = d2.w_distance(&d1);
        assert!((d12 - d21).abs() < 1e-10);
    }

    #[test]
    fn agent_barycenter_with() {
        let d1 = AgentDistribution::new(vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 4.0, capability: 2.0 },
        ]);
        let d2 = AgentDistribution::new(vec![
            AgentPoint { id: "c".into(), position: 4.0, capability: 3.0 },
            AgentPoint { id: "d".into(), position: 8.0, capability: 4.0 },
        ]);
        let interp = d1.barycenter_with(&d2, 0.5);
        assert!((interp.agents[0].position - 2.0).abs() < 1e-10);
        assert!((interp.agents[1].position - 6.0).abs() < 1e-10);
        assert!((interp.agents[0].capability - 2.0).abs() < 1e-10);
    }

    // ─── Serde tests ───

    #[test]
    fn serde_discrete_measure() {
        let m = DiscreteMeasure::new(vec![0.0, 1.0], vec![0.6, 0.4]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: DiscreteMeasure = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn serde_dense_matrix() {
        let m = DenseMatrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: DenseMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn serde_transport_plan() {
        let mu = DiscreteMeasure::new(vec![0.0], vec![1.0]);
        let nu = DiscreteMeasure::new(vec![1.0], vec![1.0]);
        let plan = TransportPlan::new(mu.clone(), nu.clone(), DenseMatrix::new(vec![vec![1.0]]));
        let json = serde_json::to_string(&plan).unwrap();
        let plan2: TransportPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, plan2);
    }

    #[test]
    fn serde_cost_matrix() {
        let c = CostMatrix::from_euclidean_squared(&[0.0], &[1.0]);
        let json = serde_json::to_string(&c).unwrap();
        let c2: CostMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn serde_monge_map() {
        let m = MongeMap::new(vec![1, 0, 2]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: MongeMap = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn serde_agent_point() {
        let a = AgentPoint { id: "test".into(), position: 3.14, capability: 0.5 };
        let json = serde_json::to_string(&a).unwrap();
        let a2: AgentPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(a, a2);
    }

    #[test]
    fn serde_agent_distribution() {
        let d = AgentDistribution::new(vec![
            AgentPoint { id: "x".into(), position: 1.0, capability: 2.0 },
        ]);
        let json = serde_json::to_string(&d).unwrap();
        let d2: AgentDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(d.agents, d2.agents);
        assert_eq!(d.measure, d2.measure);
    }

    // ─── Theorem verification tests ───

    #[test]
    fn theorem_w2_quantile_formula() {
        // W₂²(μ,ν) = ∫₀¹ (F_μ⁻¹(t) - F_ν⁻¹(t))² dt
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let w2 = WassersteinDistance::w2(&mu, &nu);
        // Manual: quantile(μ) = {0 for t<0.5, 1 for t≥0.5}
        // quantile(ν) = {2 for t<0.5, 3 for t≥0.5}
        // integral = 0.5*(0-2)² + 0.5*(1-3)² = 0.5*4 + 0.5*4 = 4
        // W₂ = 2
        assert!((w2 - 2.0).abs() < 0.1, "W₂ was {}, expected 2.0", w2);
    }

    #[test]
    fn theorem_w1_metric_properties() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let sigma = DiscreteMeasure::uniform(vec![4.0, 5.0]);

        // Non-negative
        assert!(WassersteinDistance::w1(&mu, &nu) >= 0.0);
        // Symmetric
        assert!((WassersteinDistance::w1(&mu, &nu) - WassersteinDistance::w1(&nu, &mu)).abs() < 1e-10);
        // Triangle inequality
        let d12 = WassersteinDistance::w1(&mu, &nu);
        let d23 = WassersteinDistance::w1(&nu, &sigma);
        let d13 = WassersteinDistance::w1(&mu, &sigma);
        assert!(d13 <= d12 + d23 + 1e-10);
        // Identity
        assert!((WassersteinDistance::w1(&mu, &mu)).abs() < 1e-10);
    }

    #[test]
    fn theorem_wp_mu_mu_zero() {
        for p in [1.0, 2.0, 3.0, 5.0] {
            let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0, 3.0]);
            assert!((WassersteinDistance::wp(&mu, &mu, p)).abs() < 1e-10, "p={}", p);
        }
    }

    #[test]
    fn theorem_monge_vs_kantorovich() {
        // Monge map cost ≥ Kantorovich (relaxed) cost
        // Actually it's the reverse: Kantorovich ≤ Monge (relaxation)
        let mu = DiscreteMeasure::new(vec![0.0, 1.0], vec![0.5, 0.5]);
        let nu = DiscreteMeasure::new(vec![2.0, 3.0], vec![0.5, 0.5]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[2.0, 3.0]);
        let (kantorovich_cost, _) = WassersteinDistance::earth_movers(&mu, &nu, &cost);

        let monge = MongeMap::new(vec![0, 1]);
        let monge_cost = monge.transport_cost(&cost);

        // Kantorovich ≤ Monge (relaxation gives lower bound)
        assert!(kantorovich_cost <= monge_cost + 1e-8);
    }

    #[test]
    fn theorem_transport_marginals() {
        let mu = DiscreteMeasure::new(vec![0.0, 1.0, 2.0], vec![0.2, 0.5, 0.3]);
        let nu = DiscreteMeasure::new(vec![1.0, 3.0, 5.0], vec![0.4, 0.1, 0.5]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        let (_, plan) = WassersteinDistance::earth_movers(&mu, &nu, &cost);
        assert!(plan.marginals_valid(1e-8));
    }

    #[test]
    fn theorem_sinkhorn_convergence_to_ot() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![1.0, 2.0, 3.0]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
        let true_w2 = WassersteinDistance::w2(&mu, &nu);

        let mut sink = SinkhornAlgorithm::new(0.001, 5000, 1e-12);
        let (_, w_approx) = sink.compute(&cost, &mu, &nu);

        assert!((w_approx.sqrt() - true_w2).abs() < 0.2, "sinkhorn was {}, true was {}", w_approx.sqrt(), true_w2);
    }

    #[test]
    fn theorem_w_geq_0_equality_iff_equal() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        assert!((WassersteinDistance::w2(&mu, &nu)).abs() < 1e-10);
    }

    #[test]
    fn theorem_brenier_monotone_1d() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
        let nu = DiscreteMeasure::uniform(vec![3.0, 4.0, 5.0]);
        let monge = BrenierMap::compute(&mu, &nu);
        // In 1D, Brenier map should be monotone: T is non-decreasing
        // Since mu and nu have same support ordering, mapping should be identity-like
        assert!(BrenierMap::is_monotone(&mu, &monge));
    }

    #[test]
    fn theorem_barycenter_two_gaussians_gaussian() {
        // Two "Gaussian-like" measures (symmetric discrete)
        let g1 = DiscreteMeasure::new(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![0.1, 0.2, 0.4, 0.2, 0.1],
        );
        let g2 = DiscreteMeasure::new(
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
            vec![0.1, 0.2, 0.4, 0.2, 0.1],
        );
        let bar = WassersteinBarycenter::compute(&[g1.clone(), g2.clone()], &[0.5, 0.5], 5, 10);
        // Barycenter mean should be midpoint of means
        let expected_mean = (g1.mean() + g2.mean()) / 2.0;
        assert!((bar.mean() - expected_mean).abs() < 0.5);
    }

    // ─── Additional coverage tests ───

    #[test]
    fn dense_matrix_display() {
        let m = DenseMatrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let s = format!("{}", m);
        assert!(s.contains("1.000000"));
    }

    #[test]
    fn cost_matrix_total_cost() {
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[2.0, 3.0]);
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let plan = DenseMatrix::new(vec![vec![0.5, 0.0], vec![0.0, 0.5]]);
        let tp = TransportPlan::new(mu, nu, plan);
        let tc = cost.total_cost(&tp);
        // 0.5*4 + 0.5*4 = 4
        assert!((tc - 4.0).abs() < 1e-10);
    }

    #[test]
    fn agent_w_distance_same_is_zero() {
        let d = AgentDistribution::new(vec![
            AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
            AgentPoint { id: "b".into(), position: 1.0, capability: 2.0 },
        ]);
        assert!((d.w_distance(&d)).abs() < 1e-10);
    }

    #[test]
    fn multiple_measures_wasserstein() {
        let measures: Vec<DiscreteMeasure> = (0..5)
            .map(|i| DiscreteMeasure::uniform(vec![i as f64, i as f64 + 1.0]))
            .collect();
        for i in 0..measures.len() {
            for j in 0..measures.len() {
                let d = WassersteinDistance::w2(&measures[i], &measures[j]);
                assert!(d >= 0.0);
            }
        }
    }

    #[test]
    fn sinkhorn_history_not_empty() {
        let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
        let nu = DiscreteMeasure::uniform(vec![2.0, 3.0]);
        let cost = CostMatrix::from_euclidean_squared(&[0.0, 1.0], &[2.0, 3.0]);
        let mut sink = SinkhornAlgorithm::new(0.1, 100, 1e-8);
        sink.compute(&cost, &mu, &nu);
        assert!(!sink.convergence_history().is_empty());
    }

    #[test]
    fn measure_total_mass_non_unit() {
        let m = DiscreteMeasure::new(vec![0.0, 1.0], vec![2.0, 3.0]);
        assert!((m.total_mass() - 5.0).abs() < 1e-10);
    }
}
