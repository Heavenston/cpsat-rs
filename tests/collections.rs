//! Constraints taking collections accept any `IntoIterator`: arrays, `&Vec`,
//! slices, and iterator chains.

use cpsat_rs::prelude::*;

#[test]
fn all_different_with_expressions() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=2, "x");
    let y = model.new_int_var(0..=2, "y");
    // x, y and the constant 1 are pairwise distinct → {x, y} == {0, 2}.
    model.add_all_different([LinearExpr::from(x), y.into(), 1.into()]);
    model.minimize(x);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(x), 0);
    assert_eq!(response.value(y), 2);
}

#[test]
fn bool_constraints_from_iterators() {
    let mut model = CpModel::new();
    let bools: Vec<BoolVar> = (0..4)
        .map(|i| model.new_bool_var(format!("b{i}")))
        .collect();
    model.add_exactly_one(&bools);
    model.add_at_most_one(bools.iter());
    model.add_bool_or(bools.iter().skip(2));
    model.add_bool_and([bools[3].negated()]);
    model.maximize(LinearExpr::sum(&bools));

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    let values: Vec<bool> = bools.iter().map(|&b| response.bool_value(b)).collect();
    assert_eq!(values, [false, false, true, false]);
}

#[test]
fn element_from_iterator() {
    let mut model = CpModel::new();
    let values = [5, 7, 9];
    let array: Vec<IntVar> = values.iter().map(|&v| model.new_constant(v)).collect();
    let index = model.new_int_var(0..=2, "index");
    let target = model.new_int_var(0..=10, "target");
    model.add(LinearExpr::from(index).eq(1));
    model.add_element(index, array.iter().copied(), target);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(target), 7);
}

#[test]
fn table_with_nested_collections() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    model.add(LinearExpr::from(x).eq(3));
    model.add_table([x, y], [[1, 2], [3, 4]], false);

    let forbidden = vec![vec![3, 4]];
    let z = model.new_int_var(0..=10, "z");
    model.add_table([x, z].as_slice(), &forbidden, true);
    model.add(LinearExpr::from(z).le(4));
    model.maximize(z);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(y), 4);
    assert_eq!(response.value(z), 3);
}

#[test]
#[should_panic(expected = "add_table: tuple 1 has 1 entries, expected 2")]
fn table_rejects_wrong_tuple_length() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    model.add_table([x, y], vec![vec![1, 2], vec![3]], false);
}

#[test]
fn automaton_from_iterators() {
    let mut model = CpModel::new();
    let vars: Vec<IntVar> = (0..3)
        .map(|i| model.new_int_var(0..=1, format!("x{i}")))
        .collect();
    // Accepts exactly the words containing two consecutive 1s.
    let transitions = [(0, 0, 0), (0, 1, 1), (1, 2, 1), (1, 0, 0), (2, 2, 0)];
    model.add(LinearExpr::from(vars[0]).eq(0));
    model.add_automaton(&vars, 0, [2], transitions.iter());

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    let word: Vec<i64> = vars.iter().map(|&v| response.value(v)).collect();
    assert_eq!(word, [0, 1, 1]);
}

#[test]
fn circuit_from_iterator() {
    let mut model = CpModel::new();
    let n = 3;
    let arcs: Vec<(i32, i32, BoolVar)> = (0..n)
        .flat_map(|t| (0..n).filter(move |&h| h != t).map(move |h| (t, h)))
        .map(|(t, h)| (t, h, model.new_bool_var(format!("arc_{t}_{h}"))))
        .collect();
    let lit = |t, h| arcs.iter().find(|a| a.0 == t && a.1 == h).unwrap().2;
    model.add_bool_and([lit(0, 1)]);
    model.add_circuit(&arcs);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert!(response.bool_value(lit(1, 2)));
    assert!(response.bool_value(lit(2, 0)));
    assert!(!response.bool_value(lit(1, 0)));
}

#[test]
fn cumulative_with_expression_demands() {
    let mut model = CpModel::new();
    let mut intervals = vec![];
    let mut ends = vec![];
    for (i, size) in [3, 5].into_iter().enumerate() {
        let s = model.new_int_var(0..=20, format!("s{i}"));
        let e = model.new_int_var(0..=20, format!("e{i}"));
        intervals.push(model.new_interval_var(s, size, e, format!("t{i}")));
        ends.push(e);
    }
    // Two tasks of demand 2 under capacity 3 cannot overlap.
    model.add_cumulative(&intervals, [2, 2], 3);
    let makespan = model.new_int_var(0..=20, "makespan");
    model.add_max_equality(makespan, &ends);
    model.minimize(makespan);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(makespan), 8);
}

#[test]
fn no_overlap_2d_from_mixed_collections() {
    let mut model = CpModel::new();
    let mut xs = vec![];
    let mut x_intervals = vec![];
    let mut y_intervals = vec![];
    for i in 0..2 {
        let x = model.new_int_var(0..=2, format!("x{i}"));
        let y = model.new_constant(0);
        x_intervals.push(model.new_interval_var(x, 2, x + 2, format!("x_iv{i}")));
        y_intervals.push(model.new_interval_var(y, 2, y + 2, format!("y_iv{i}")));
        xs.push(x);
    }
    // Same row → the two 2x2 squares must be separated along x.
    model.add_no_overlap_2d(x_intervals, y_intervals.iter());
    model.minimize(LinearExpr::sum(&xs));

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.objective_value(), 2.0);
}

#[test]
fn sum_and_weighted_sum_from_iterators() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    let b = model.new_bool_var("b");
    model.add(LinearExpr::weighted_sum([x, y], [2, 3]).eq(13));
    model.add(LinearExpr::sum([LinearExpr::from(x), y.into(), b.into()]).eq(6));
    model.add(LinearExpr::from(b).eq(1));

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(x), 2);
    assert_eq!(response.value(y), 3);
}
