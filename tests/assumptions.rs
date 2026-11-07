//! Assumptions.

use cpsat_rs::prelude::*;

#[test]
fn add_assumptions_appends_literal_indices() {
    let mut model = CpModel::new();
    let a = model.new_bool_var("a");
    let b = model.new_bool_var("b");
    let c = model.new_bool_var("c");

    model.add_assumptions([a, !b]);
    let more = [c];
    model.add_assumptions(more.iter());
    assert_eq!(
        model.to_proto().assumptions,
        vec![a.index(), (!b).index(), c.index()]
    );
}

#[test]
fn assumptions_are_enforced() {
    let mut model = CpModel::new();
    let a = model.new_bool_var("a");
    let b = model.new_bool_var("b");
    model.add_bool_or([a, b]);
    model.add_assumptions([!a]);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_feasible());
    assert!(!response.bool_value(a));
    assert!(response.bool_value(b));
}

#[test]
fn infeasible_assumptions_are_reported() {
    let mut model = CpModel::new();
    let a = model.new_bool_var("a");
    let b = model.new_bool_var("b");
    let free = model.new_bool_var("free");
    model.add_implication(a, b);
    model.add_assumptions([free, a, !b]);

    let response = CpSolver::solve(&model).unwrap();
    assert_eq!(response.status(), CpSolverStatus::Infeasible);
    let core = response.sufficient_assumptions_for_infeasibility();
    assert!(!core.is_empty());
    assert!(core.iter().all(|l| [a.index(), (!b).index()].contains(l)));
}
