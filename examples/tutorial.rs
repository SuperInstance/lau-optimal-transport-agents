//! # Optimal Transport Agents — Tutorial
//!
//! Progressive lessons covering the core concepts of computational optimal transport,
//! from discrete measures to the Sinkhorn algorithm, Wasserstein geometry, and agent
//! distribution matching.
//!
//! Run with: `cargo run --example tutorial`

use lau_optimal_transport_agents::{
    AgentDistribution, AgentPoint, BrenierMap, CostMatrix, DiscreteMeasure,
    JKOFlow, SinkhornAlgorithm, WassersteinBarycenter,
    WassersteinDistance,
};

// ── Lesson 1: Discrete Probability Measures ──────────────────────────────────
//
// A DiscreteMeasure is a probability distribution with finite support.
// It's the fundamental building block: every transport problem starts with
// a source measure μ and a target measure ν.

fn lesson_1_discrete_measures() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 1: Discrete Probability Measures");
    println!("═══════════════════════════════════════════\n");

    // Create a uniform measure on 5 points: μ = Σ (1/5) δ_{xᵢ}
    let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    println!("Uniform measure μ on {{0, 1, 2, 3, 4}}:");
    println!("  support = {:?}", mu.support);
    println!("  weights = {:?}", mu.weights);
    println!("  total mass = {:.4}", mu.total_mass());
    println!("  is probability? {}", mu.is_probability());

    // A Dirac delta: all mass at a single point
    let dirac = DiscreteMeasure::dirac(2.5);
    println!("\nDirac measure δ_{{2.5}}:");
    println!("  support = {:?}", dirac.support);
    println!("  mean = {:.4}", dirac.mean());
    println!("  variance = {:.6}", dirac.variance());

    // Custom weighted measure
    let skewed = DiscreteMeasure::new(vec![0.0, 1.0, 2.0], vec![0.5, 0.3, 0.2]);
    println!("\nSkewed measure (weights 0.5, 0.3, 0.2):");
    println!("  mean = {:.4}", skewed.mean());
    println!("  variance = {:.4}", skewed.variance());
    println!("  CDF at x=0.5 = {:.4}", skewed.cdf(0.5));
    println!("  CDF at x=1.5 = {:.4}", skewed.cdf(1.5));
    println!("  quantile(0.5) = {:.4}", skewed.quantile(0.5));
    println!("  quantile(0.8) = {:.4}", skewed.quantile(0.8));
    println!();
}

// ── Lesson 2: Cost Matrices ─────────────────────────────────────────────────
//
// The cost matrix C defines the "price" c(xᵢ, yⱼ) of moving one unit of
// mass from source point xᵢ to target point yⱼ. The choice of cost function
// determines the geometry of the transport problem.

fn lesson_2_cost_matrices() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 2: Cost Matrices");
    println!("═══════════════════════════════════════════\n");

    let source = vec![0.0, 1.0];
    let target = vec![0.0, 1.0, 2.0];

    // Euclidean cost: c(x,y) = |x - y|
    let c_euclidean = CostMatrix::from_euclidean(&source, &target);
    println!("Euclidean cost matrix (|x - y|):");
    println!("  source = {:?}, target = {:?}", source, target);
    println!("  {}", c_euclidean.costs);

    // Squared Euclidean cost: c(x,y) = |x - y|²
    let c_squared = CostMatrix::from_euclidean_squared(&source, &target);
    println!("Squared Euclidean cost matrix (|x - y|²):");
    println!("  {}", c_squared.costs);

    // Custom cost function
    let c_custom = CostMatrix::from_custom(&source, &target, &|x, y| {
        (x - y).abs().powi(3) // cubic cost
    });
    println!("Custom cubic cost matrix (|x - y|³):");
    println!("  {}", c_custom.costs);
    println!();
}

// ── Lesson 3: Transport Plans and Optimal Couplings ─────────────────────────
//
// A TransportPlan is a joint distribution γ ∈ Π(μ, ν) — a matrix where
// row sums recover μ and column sums recover ν. The optimal plan minimizes
// the total transport cost Σ γᵢⱼ c(xᵢ, yⱼ).

fn lesson_3_transport_plans() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 3: Transport Plans");
    println!("═══════════════════════════════════════════\n");

    let mu = DiscreteMeasure::uniform(vec![0.0, 1.0]);
    let nu = DiscreteMeasure::uniform(vec![0.5, 1.5]);
    let cost = CostMatrix::from_euclidean(&mu.support, &nu.support);

    // Compute the exact optimal transport plan (monotone coupling for 1D)
    let (total_cost, plan) = WassersteinDistance::earth_movers(&mu, &nu, &cost);

    println!("Source measure μ: support = {:?}", mu.support);
    println!("Target measure ν: support = {:?}", nu.support);
    println!("Optimal transport plan γ:");
    println!("  {}", plan.plan);
    println!("  Total cost = {:.6}", total_cost);
    println!("  Marginals valid? {}", plan.marginals_valid(1e-6));
    println!("  Total mass = {:.6}", plan.total_mass());
    println!();
}

// ── Lesson 4: Wasserstein Distances ─────────────────────────────────────────
//
// The Wasserstein distance W_p(μ, ν) measures the "minimum effort" to reshape
// μ into ν. In 1D it has beautiful closed-form expressions via quantile functions:
//   W_p^p = ∫₀¹ |F_μ⁻¹(t) - F_ν⁻¹(t)|^p dt

fn lesson_4_wasserstein_distances() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 4: Wasserstein Distances");
    println!("═══════════════════════════════════════════\n");

    let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0, 3.0]);
    let nu = DiscreteMeasure::uniform(vec![1.0, 2.0, 3.0, 4.0]);

    // W₁: Earth Mover's Distance via CDF integration
    let w1 = WassersteinDistance::w1(&mu, &nu);
    println!("μ = uniform on {{0, 1, 2, 3}}");
    println!("ν = uniform on {{1, 2, 3, 4}}");
    println!("  W₁(μ, ν) = {:.6}", w1);
    println!("  (Expected: 1.0 — each point shifted by 1)");

    // W₂: Quadratic Wasserstein distance via quantile functions
    let w2 = WassersteinDistance::w2(&mu, &nu);
    println!("  W₂(μ, ν) = {:.6}", w2);

    // General W_p for any p
    let w3 = WassersteinDistance::wp(&mu, &nu, 3.0);
    println!("  W₃(μ, ν) = {:.6}", w3);

    // Compare with Dirac vs Uniform
    let dirac = DiscreteMeasure::dirac(0.0);
    let spread = DiscreteMeasure::uniform(vec![-1.0, 0.0, 1.0]);
    println!("\nδ₀ vs uniform on {{-1, 0, 1}}:");
    println!("  W₁ = {:.6}", WassersteinDistance::w1(&dirac, &spread));
    println!("  W₂ = {:.6}", WassersteinDistance::w2(&dirac, &spread));
    println!();
}

// ── Lesson 5: The Sinkhorn Algorithm ────────────────────────────────────────
//
// The Sinkhorn algorithm solves entropy-regularized optimal transport:
//   min Σ γᵢⱼ cᵢⱼ + ε Σ γᵢⱼ log(γᵢⱼ)
// It's fast, differentiable, and produces dense transport plans.
// Uses log-domain stabilization for numerical stability.

fn lesson_5_sinkhorn_algorithm() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 5: The Sinkhorn Algorithm");
    println!("═══════════════════════════════════════════\n");

    let mu = DiscreteMeasure::uniform(vec![0.0, 1.0, 2.0]);
    let nu = DiscreteMeasure::uniform(vec![0.5, 1.5, 2.5]);
    let cost = CostMatrix::from_euclidean(&mu.support, &nu.support);

    // Low regularization: closer to exact OT
    let mut sinkhorn = SinkhornAlgorithm::new(0.01, 500, 1e-8);
    let (plan, reg_distance) = sinkhorn.compute(&cost, &mu, &nu);

    println!("Sinkhorn with ε = 0.01 (low regularization):");
    println!("  Regularized distance W̃_ε = {:.6}", reg_distance);
    println!("  Transport plan:");
    for i in 0..plan.plan.rows() {
        let row: Vec<String> = (0..plan.plan.cols())
            .map(|j| format!("{:.4}", plan.plan.get(i, j)))
            .collect();
        println!("    [{}]", row.join(", "));
    }
    println!("  Convergence iterations: {}", sinkhorn.convergence_history.len());

    // High regularization: more entropic, "softer" plan
    let mut sinkhorn_soft = SinkhornAlgorithm::new(1.0, 500, 1e-8);
    let (plan_soft, reg_distance_soft) = sinkhorn_soft.compute(&cost, &mu, &nu);

    println!("\nSinkhorn with ε = 1.0 (high regularization):");
    println!("  Regularized distance W̃_ε = {:.6}", reg_distance_soft);
    println!("  Plan is more spread out (entropic):");
    for i in 0..plan_soft.plan.rows() {
        let row: Vec<String> = (0..plan_soft.plan.cols())
            .map(|j| format!("{:.4}", plan_soft.plan.get(i, j)))
            .collect();
        println!("    [{}]", row.join(", "));
    }
    println!();
}

// ── Lesson 6: Monge Maps and the Brenier Theorem ────────────────────────────
//
// A Monge map is a deterministic transport T: X → Y that pushes μ forward to ν.
// For quadratic cost in 1D, the Brenier map is the monotone rearrangement —
// the gradient of a convex function.

fn lesson_6_monge_and_brenier() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 6: Monge Maps & the Brenier Map");
    println!("═══════════════════════════════════════════\n");

    let mu = DiscreteMeasure::new(vec![0.0, 1.0, 2.0], vec![0.4, 0.3, 0.3]);
    let nu = DiscreteMeasure::new(vec![0.5, 1.5, 2.5], vec![0.3, 0.4, 0.3]);
    let cost = CostMatrix::from_euclidean_squared(&mu.support, &nu.support);

    // Compute the Brenier map (monotone rearrangement for 1D)
    let monge = BrenierMap::compute(&mu, &nu);
    println!("Source μ: support = {:?}, weights = {:?}", mu.support, mu.weights);
    println!("Target ν: support = {:?}, weights = {:?}", nu.support, nu.weights);
    println!("Brenier map T (monotone rearrangement):");
    println!("  mapping = {:?}", monge.mapping);
    for i in 0..mu.n() {
        println!("  T({:.1}) = {:.1}", mu.support[i], nu.support[monge.mapping[i]]);
    }

    println!("  Is monotone? {}", BrenierMap::is_monotone(&mu, &monge));
    println!("  Transport cost = {:.6}", monge.transport_cost(&cost));

    // Convert to a transport plan
    let plan = monge.to_transport_plan(&mu);
    println!("\nMonge map as transport plan:");
    println!("  {}", plan.plan);

    // Verify pushforward
    let tolerance = 0.1;
    println!("  Is valid Monge map? {}", monge.is_monge_map(&mu, &nu, tolerance));
    println!();
}

// ── Lesson 7: Wasserstein Barycenters ───────────────────────────────────────
//
// The Wasserstein barycenter is the Fréchet mean in Wasserstein space:
//   b̄ = argmin_b Σ wᵢ W₂²(b, μᵢ)
// Computed via fixed-point iteration on quantile functions.

fn lesson_7_wasserstein_barycenters() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 7: Wasserstein Barycenters");
    println!("═══════════════════════════════════════════\n");

    let mu1 = DiscreteMeasure::uniform(vec![0.0, 1.0]);
    let mu2 = DiscreteMeasure::uniform(vec![4.0, 5.0]);
    let mu3 = DiscreteMeasure::uniform(vec![1.0, 3.0]);

    // Equal-weight barycenter
    let bary = WassersteinBarycenter::compute(
        &[mu1.clone(), mu2.clone(), mu3.clone()],
        &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        10,
        100,
    );

    println!("Three measures:");
    println!("  μ₁ = uniform on {{0, 1}}");
    println!("  μ₂ = uniform on {{4, 5}}");
    println!("  μ₃ = uniform on {{1, 3}}");
    println!("\nEqual-weight barycenter:");
    println!("  support = {:?}", bary.support);
    println!("  mean = {:.4}", bary.mean());

    // Check it's approximately a barycenter
    let is_bary = WassersteinBarycenter::is_barycenter(
        &bary,
        &[mu1, mu2, mu3],
        &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        0.1,
    );
    println!("  Is barycenter? {}", is_bary);

    // Weighted barycenter: lean toward μ₂
    let weighted_bary = WassersteinBarycenter::compute(
        &[
            DiscreteMeasure::uniform(vec![0.0, 1.0]),
            DiscreteMeasure::uniform(vec![4.0, 5.0]),
        ],
        &[0.2, 0.8],
        10,
        100,
    );
    println!("\nWeighted barycenter (0.2·μ₁ + 0.8·μ₂):");
    println!("  support = {:?}", weighted_bary.support);
    println!("  mean = {:.4} (closer to μ₂'s mean = 4.5)", weighted_bary.mean());
    println!();
}

// ── Lesson 8: Agent Distributions and JKO Gradient Flow ─────────────────────
//
// An AgentDistribution models agents as a probability distribution.
// JKO flow (Jordan-Kinderlehrer-Otto) is the gradient descent of a
// functional in Wasserstein space:
//   μ_{n+1} = argmin_ν { τ·F(ν) + W₂²(μₙ, ν)/2 }

fn lesson_8_agents_and_jko() {
    println!("═══════════════════════════════════════════");
    println!("  Lesson 8: Agent Distributions & JKO Flow");
    println!("═══════════════════════════════════════════\n");

    // Create agents at various positions
    let agents = vec![
        AgentPoint { id: "alice".into(), position: 0.0, capability: 0.8 },
        AgentPoint { id: "bob".into(), position: 3.0, capability: 0.5 },
        AgentPoint { id: "carol".into(), position: 5.0, capability: 0.9 },
        AgentPoint { id: "dave".into(), position: 7.0, capability: 0.3 },
    ];
    let mut dist = AgentDistribution::new(agents.clone());
    println!("Agent distribution:");
    for a in &dist.agents {
        println!("  {} at {:.1} (capability: {:.1})", a.id, a.position, a.capability);
    }
    println!("  mean = {:.2}", dist.measure.mean());

    // Manipulate the distribution
    dist.spread(0.3);
    println!("\nAfter spread(0.3):");
    println!("  mean = {:.2} (same center, wider)", dist.measure.mean());

    let mut dist2 = AgentDistribution::new(agents);
    dist2.concentrate(0.5);
    println!("\nAfter concentrate(0.5):");
    println!("  mean = {:.2} (same center, tighter)", dist2.measure.mean());

    // Wasserstein distance between agent distributions
    let w = dist.w_distance(&dist2);
    println!("  W₂(spread, concentrated) = {:.4}", w);

    // JKO gradient flow: minimize variance (agents cluster together)
    let initial = DiscreteMeasure::uniform(vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    println!("\nJKO gradient flow (minimize variance):");
    println!("  Initial mean = {:.2}, variance = {:.2}", initial.mean(), initial.variance());

    let trajectory = JKOFlow::run(
        &initial,
        &|m| m.variance(), // functional: variance
        0.5,               // time step τ
        5,                 // steps
        20,                // grid size for search
    );

    for (step, m) in trajectory.iter().enumerate() {
        println!(
            "  Step {}: mean = {:.2}, variance = {:.2}",
            step, m.mean(), m.variance()
        );
    }

    // Interpolate between two agent distributions
    let d1 = AgentDistribution::new(vec![
        AgentPoint { id: "a".into(), position: 0.0, capability: 1.0 },
        AgentPoint { id: "b".into(), position: 4.0, capability: 0.0 },
    ]);
    let d2 = AgentDistribution::new(vec![
        AgentPoint { id: "a".into(), position: 4.0, capability: 0.0 },
        AgentPoint { id: "b".into(), position: 0.0, capability: 1.0 },
    ]);
    let interp = d1.barycenter_with(&d2, 0.5);
    println!("\nBarycenter interpolation (α = 0.5):");
    for a in &interp.agents {
        println!("  {} at {:.2}", a.id, a.position);
    }
    println!();
}

fn main() {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║  Optimal Transport Agents — Interactive Tutorial  ║");
    println!("║  From discrete measures to Wasserstein geometry   ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    lesson_1_discrete_measures();
    lesson_2_cost_matrices();
    lesson_3_transport_plans();
    lesson_4_wasserstein_distances();
    lesson_5_sinkhorn_algorithm();
    lesson_6_monge_and_brenier();
    lesson_7_wasserstein_barycenters();
    lesson_8_agents_and_jko();

    println!("═════════════════════════════════════════════════");
    println!("  ✓ Tutorial complete — all 8 lessons done!");
    println!("═════════════════════════════════════════════════");
}
