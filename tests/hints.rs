//! Solution hints.

use cpsat_rs::prelude::*;

#[test]
fn clear_hints_removes_all_hints() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    model.add_hint(x, 3);
    model.add_hint(y, 4);
    assert!(model.to_proto().solution_hint.is_some());

    model.clear_hints();
    assert!(model.to_proto().solution_hint.is_none());

    // Clearing twice is a no-op, and hints can be added again afterwards.
    model.clear_hints();
    model.add_hint(y, 7);
    let hint = model.to_proto().solution_hint.as_ref().unwrap();
    assert_eq!(hint.vars, vec![y.index()]);
    assert_eq!(hint.values, vec![7]);

    model.add((x + y).le(15));
    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_feasible());
}
