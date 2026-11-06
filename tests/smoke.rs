use cpsat_rs::prelude::*;

#[test]
fn trivial_two_vars() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    model.add((x + y).le(15));
    model.minimize(x + y);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.objective_value(), 0.0);
    assert_eq!(response.value(x), 0);
    assert_eq!(response.value(y), 0);
}

#[test]
fn infeasible_model() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=5, "x");
    // x >= 10 is impossible when domain is [0, 5]
    model.add(LinearExpr::from(x).ge(10));

    let response = CpSolver::solve(&model).unwrap();
    assert_eq!(response.status(), CpSolverStatus::Infeasible);
}

#[test]
fn all_different() {
    let mut model = CpModel::new();
    let vars: Vec<IntVar> = (0..4)
        .map(|i| model.new_int_var(1..=4, format!("x{}", i)))
        .collect();
    model.add_all_different(&vars);
    model.minimize(LinearExpr::sum(&vars));

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    // Minimum sum of {1,2,3,4} = 10
    assert_eq!(response.objective_value(), 10.0);
}

#[test]
fn bool_vars_and_implication() {
    let mut model = CpModel::new();
    let a = model.new_bool_var("a");
    let b = model.new_bool_var("b");
    // a => b
    model.add_implication(a, b);
    // a must be true
    model.add(LinearExpr::from(a.as_int_var()).ge(1));

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_feasible());
    assert!(response.bool_value(a));
    assert!(response.bool_value(b));
}

#[test]
fn maximize_objective() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    model.add((x + y).le(15));
    model.maximize(2_i64 * x + 3_i64 * y);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    // max 2x + 3y s.t. x+y <= 15, x,y in [0,10]
    // y=10, x=5 → 2*5 + 3*10 = 40
    assert_eq!(response.objective_value(), 40.0);
}

#[test]
fn solver_with_params() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=100, "x");
    model.minimize(x);

    let params = SatParameters::default()
        .with_max_time(5.0)
        .with_num_workers(1);

    let response = CpSolver::solve_with_params(&model, &params).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(x), 0);
}

#[test]
fn max_equality_with_expressions() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    let t = model.new_int_var(-100..=100, "t");
    model.add(LinearExpr::from(x).eq(3));
    model.add(LinearExpr::from(y).eq(4));
    // t + 1 == max(2x + 1, y - 2, 5)
    model.add_max_equality(
        t + 1,
        [2 * x + 1, LinearExpr::from(y) - 2, LinearExpr::constant(5)],
    );

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(t), 6);
}

#[test]
fn min_equality_with_expressions() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    let t = model.new_int_var(-100..=100, "t");
    model.add(LinearExpr::from(x).eq(3));
    model.add(LinearExpr::from(y).eq(4));
    // 2t == min(x + 10, 3y - 2, 20)
    model.add_min_equality(2 * t, [x + 10, 3 * y - 2, LinearExpr::constant(20)]);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(t), 5);
}

#[test]
fn min_equality_with_var_slice() {
    let mut model = CpModel::new();
    let x = model.new_int_var(2..=10, "x");
    let y = model.new_int_var(4..=10, "y");
    let t = model.new_int_var(0..=10, "t");
    let vars = [x, y];
    model.add_min_equality(t, vars.iter());
    model.maximize(t);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(t), 10);
}
