//! Iterating over model variables.

use cpsat_rs::prelude::*;

#[test]
fn variables_empty_model() {
    let model = CpModel::new();
    assert_eq!(model.variables().len(), 0);
    assert_eq!(model.variables().next(), None);
}

#[test]
fn variables_yields_all_vars_in_creation_order() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let b = model.new_bool_var("b");
    let c = model.new_constant(5);
    let y = model.new_int_var(-3..=3, "y");

    let vars: Vec<IntVar> = model.variables().collect();
    assert_eq!(model.variables().len(), 4);
    assert_eq!(
        vars.iter().map(|v| v.index()).collect::<Vec<_>>(),
        vec![x.index(), b.index(), c.index(), y.index()]
    );
    assert_eq!(vars[0], x);
    assert_eq!(vars[2], c);
    assert_eq!(vars[3], y);
    assert_eq!(model.variables().next_back(), Some(y));
}

#[test]
fn variables_usable_in_expressions() {
    let mut model = CpModel::new();
    for i in 0..5 {
        model.new_int_var(0..=10, format!("x{i}"));
    }
    let total: LinearExpr = model.variables().sum();
    model.add(total.eq(50));

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_feasible());
    for v in model.variables() {
        assert_eq!(response.value(v), 10);
    }
}
