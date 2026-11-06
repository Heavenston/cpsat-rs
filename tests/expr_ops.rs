//! `AddAssign` and `Sum` on `LinearExpr`.

use cpsat_rs::prelude::*;

#[test]
fn add_assign_accepts_vars_constants_and_exprs() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    let b = model.new_bool_var("b");

    let mut expr = LinearExpr::default();
    expr += x;
    expr += &y;
    expr += b;
    expr += 2 * x;
    expr += 5;
    // 3x + y + b + 5 == 19 with y maximized at 10 → 3x + b == 4 → x = 1, b = 1.
    model.add(expr.eq(19));
    model.maximize(y);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    assert_eq!(response.value(y), 10);
    assert_eq!(response.value(x), 1);
    assert!(response.bool_value(b));
}

#[test]
fn sum_over_iterators() {
    let mut model = CpModel::new();
    let vars: Vec<IntVar> = (0..4)
        .map(|i| model.new_int_var(0..=3, format!("v{i}")))
        .collect();

    // Sum of owned vars, borrowed vars, and mapped expressions.
    let total: LinearExpr = vars.iter().copied().sum();
    let total_ref: LinearExpr = vars.iter().sum();
    let weighted: LinearExpr = vars
        .iter()
        .enumerate()
        .map(|(i, &v)| v * (i as i64 + 1))
        .sum();

    model.add(total.eq(6));
    model.add(total_ref.le(6));
    model.maximize(weighted);

    let response = CpSolver::solve(&model).unwrap();
    assert!(response.is_optimal());
    // Best is to push weight onto the last vars: v3 = 3, v2 = 3.
    assert_eq!(response.objective_value(), 21.0);
}

#[test]
fn empty_sum_is_zero() {
    let mut model = CpModel::new();
    let x = model.new_int_var(-5..=5, "x");
    let empty: LinearExpr = std::iter::empty::<IntVar>().sum();
    model.add((empty + x).eq(-3));

    let response = CpSolver::solve(&model).unwrap();
    assert_eq!(response.value(x), -3);
}

#[test]
fn eval_expression_against_solution() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    let b = model.new_bool_var("b");
    model.add(LinearExpr::from(x).eq(3));
    model.add(LinearExpr::from(y).eq(7));
    model.add(LinearExpr::from(b).eq(1));

    let response = CpSolver::solve(&model).unwrap();
    let expr = 2 * x - y * 3 + b + 5;
    assert_eq!(response.eval(&expr), 6 - 21 + 1 + 5);
    assert_eq!(response.eval(expr), -9);
    assert_eq!(response.eval(x), 3);
    assert_eq!(response.eval(b), 1);
    assert_eq!(response.eval(42), 42);
    assert_eq!(response.eval([x, y].iter().sum::<LinearExpr>()), 10);
}

#[test]
#[should_panic(expected = "Cannot read solution values")]
fn eval_panics_on_infeasible() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=1, "x");
    model.add(LinearExpr::from(x).eq(5));
    let response = CpSolver::solve(&model).unwrap();
    let _ = response.eval(x + 1);
}

#[test]
#[should_panic(expected = "Cannot convert negated BoolVar")]
fn add_negated_bool_var_panics() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=1, "x");
    let b = model.new_bool_var("b");
    let _ = LinearExpr::from(x) + !b;
}

#[test]
fn bool_var_operators() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let a = model.new_bool_var("a");
    let b = model.new_bool_var("b");
    model.add(LinearExpr::from(x).eq(4));
    model.add(LinearExpr::from(a).eq(1));
    model.add(LinearExpr::from(b).eq(0));

    let response = CpSolver::solve(&model).unwrap();
    assert_eq!(response.eval(a + b), 1);
    assert_eq!(response.eval(a - b), 1);
    assert_eq!(response.eval(b - a), -1);
    assert_eq!(response.eval(a + x), 5);
    assert_eq!(response.eval(x + a), 5);
    assert_eq!(response.eval(a - x), -3);
    assert_eq!(response.eval(x - a), 3);
    assert_eq!(response.eval(a + 7), 8);
    assert_eq!(response.eval(a * 3), 3);
    assert_eq!(response.eval(3 * a), 3);
    assert_eq!(response.eval(LinearExpr::from(x) - a), 3);
}

#[test]
#[should_panic(expected = "Cannot convert negated BoolVar")]
fn bool_var_operator_on_negated_panics() {
    let mut model = CpModel::new();
    let a = model.new_bool_var("a");
    let b = model.new_bool_var("b");
    let _ = a + !b;
}

#[test]
fn var_constant_and_expr_operators() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    let a = model.new_bool_var("a");
    model.add(LinearExpr::from(x).eq(4));
    model.add(LinearExpr::from(y).eq(6));
    model.add(LinearExpr::from(a).eq(1));

    let response = CpSolver::solve(&model).unwrap();
    let e = || 2 * y + 1; // 13

    // IntVar
    assert_eq!(response.eval(x + e()), 17);
    assert_eq!(response.eval(x - e()), -9);
    assert_eq!(response.eval(x - 3), 1);
    assert_eq!(response.eval(3 + x), 7);
    assert_eq!(response.eval(3 - x), -1);

    // BoolVar
    assert_eq!(response.eval(a + e()), 14);
    assert_eq!(response.eval(a - e()), -12);
    assert_eq!(response.eval(a - 3), -2);
    assert_eq!(response.eval(3 + a), 4);
    assert_eq!(response.eval(3 - a), 2);
}

#[test]
#[should_panic(expected = "LinearExpr constant overflow")]
fn var_minus_constant_overflow_panics() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=1, "x");
    let _ = x - i64::MIN;
}

#[test]
fn constant_lhs_expr_ops_and_compound_assignment() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let y = model.new_int_var(0..=10, "y");
    let a = model.new_bool_var("a");
    model.add(LinearExpr::from(x).eq(4));
    model.add(LinearExpr::from(y).eq(6));
    model.add(LinearExpr::from(a).eq(1));

    let response = CpSolver::solve(&model).unwrap();
    let e = || x + y + 1; // 11

    assert_eq!(response.eval(5 + e()), 16);
    assert_eq!(response.eval(5 - e()), -6);
    assert_eq!(response.eval(3 * e()), 33);

    let mut expr = e();
    expr -= y; // x + 1
    expr -= &x; // 1
    expr -= a; // 1 - a
    expr -= 2 * y; // 1 - a - 2y
    expr -= 4; // -3 - a - 2y
    assert_eq!(response.eval(&expr), -16);
    expr *= -2;
    assert_eq!(response.eval(expr), 32);
}

#[test]
#[should_panic(expected = "LinearExpr coefficient overflow on multiplication")]
fn mul_assign_overflow_panics() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=1, "x");
    let mut expr = x * i64::MAX;
    expr *= 2;
}

#[test]
fn negate_vars() {
    let mut model = CpModel::new();
    let x = model.new_int_var(0..=10, "x");
    let a = model.new_bool_var("a");
    model.add(LinearExpr::from(x).eq(4));
    model.add(LinearExpr::from(a).eq(1));

    let response = CpSolver::solve(&model).unwrap();
    assert_eq!(response.eval(-x), -4);
    assert_eq!(response.eval(-a), -1);
    assert_eq!(response.eval(-x + 10), 6);
    assert_eq!(response.eval(-a - x), -5);
}

#[test]
#[should_panic(expected = "Cannot convert negated BoolVar")]
fn negate_negated_bool_var_panics() {
    let mut model = CpModel::new();
    let a = model.new_bool_var("a");
    let _ = -!a;
}
