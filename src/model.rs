//! CpModel builder — the main entry point for constructing optimization models.

use crate::expressions::{BoundedLinearExpr, LinearExpr};
use crate::proto::{
    constraint_proto, AllDifferentConstraintProto, AutomatonConstraintProto, BoolArgumentProto,
    CircuitConstraintProto, ConstraintProto, CpModelProto, CpObjectiveProto,
    CumulativeConstraintProto, ElementConstraintProto, IntegerVariableProto,
    IntervalConstraintProto, LinearArgumentProto, LinearConstraintProto,
    NoOverlap2DConstraintProto, NoOverlapConstraintProto, PartialVariableAssignment,
    TableConstraintProto,
};
use crate::vars::{BoolVar, IntVar, IntervalVar};
use prost::Message;
use std::borrow::Borrow;

/// Domain specification for integer variables.
pub trait IntoDomain {
    /// Convert to the flat domain representation used by CP-SAT proto.
    /// Format: pairs of [lb, ub] intervals, flattened into a single Vec.
    fn into_domain(self) -> Vec<i64>;
}

impl IntoDomain for (i64, i64) {
    fn into_domain(self) -> Vec<i64> {
        assert!(
            self.0 <= self.1,
            "Domain ({}, {}) has lb > ub",
            self.0,
            self.1
        );
        vec![self.0, self.1]
    }
}

impl IntoDomain for std::ops::RangeInclusive<i64> {
    fn into_domain(self) -> Vec<i64> {
        assert!(
            self.start() <= self.end(),
            "Domain {}..={} is empty",
            self.start(),
            self.end()
        );
        vec![*self.start(), *self.end()]
    }
}

impl IntoDomain for std::ops::Range<i64> {
    fn into_domain(self) -> Vec<i64> {
        assert!(
            self.start < self.end,
            "Domain {}..{} is empty",
            self.start,
            self.end
        );
        vec![self.start, self.end - 1]
    }
}

impl IntoDomain for i64 {
    fn into_domain(self) -> Vec<i64> {
        vec![self, self]
    }
}

impl IntoDomain for Vec<(i64, i64)> {
    fn into_domain(self) -> Vec<i64> {
        assert!(!self.is_empty(), "Domain interval list must not be empty");
        for &(a, b) in &self {
            assert!(a <= b, "Domain interval ({a}, {b}) has lb > ub");
        }
        // Proto requires sorted, disjoint, non-adjacent intervals:
        // for all i < n-1: max_i + 1 < min_{i+1}
        for w in self.windows(2) {
            assert!(
                w[0].1 + 1 < w[1].0,
                "Domain intervals ({}, {}) and ({}, {}) are not sorted/disjoint/non-adjacent \
                 (require max + 1 < next_min)",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
        self.into_iter().flat_map(|(a, b)| [a, b]).collect()
    }
}

impl IntoDomain for std::ops::RangeInclusive<i32> {
    fn into_domain(self) -> Vec<i64> {
        assert!(
            self.start() <= self.end(),
            "Domain {}..={} is empty",
            self.start(),
            self.end()
        );
        vec![i64::from(*self.start()), i64::from(*self.end())]
    }
}

/// A CP-SAT model builder.
///
/// Construct variables, add constraints, set an objective, then solve.
///
/// # Example
/// ```no_run
/// use cpsat_rs::prelude::*;
///
/// let mut model = CpModel::new();
/// let x = model.new_int_var(0..=10, "x");
/// let y = model.new_int_var(0..=10, "y");
/// model.add((x + y).le(15));
/// model.minimize(x + y);
/// let response = CpSolver::solve(&model).unwrap();
/// ```
pub struct CpModel {
    pub(crate) proto: CpModelProto,
}

impl Clone for CpModel {
    fn clone(&self) -> Self {
        Self {
            proto: self.proto.clone(),
        }
    }
}

impl CpModel {
    /// Create a new empty model.
    pub fn new() -> Self {
        Self {
            proto: CpModelProto::default(),
        }
    }

    // ───── Variable creation ─────

    /// Add an integer variable with the given domain.
    ///
    /// # Panics
    ///
    /// Panics if the domain is invalid (empty range, lb > ub, or
    /// non-sorted/non-disjoint intervals for `Vec<(i64, i64)>`).
    pub fn new_int_var(&mut self, domain: impl IntoDomain, name: impl AsRef<str>) -> IntVar {
        let idx = self.proto.variables.len() as i32;
        self.proto.variables.push(IntegerVariableProto {
            name: name.as_ref().to_string(),
            domain: domain.into_domain(),
        });
        IntVar(idx)
    }

    /// Add a Boolean variable (domain [0, 1]).
    pub fn new_bool_var(&mut self, name: impl AsRef<str>) -> BoolVar {
        let idx = self.proto.variables.len() as i32;
        self.proto.variables.push(IntegerVariableProto {
            name: name.as_ref().to_string(),
            domain: vec![0, 1],
        });
        BoolVar(idx)
    }

    /// Add a constant variable.
    pub fn new_constant(&mut self, value: i64) -> IntVar {
        let idx = self.proto.variables.len() as i32;
        self.proto.variables.push(IntegerVariableProto {
            name: String::new(),
            domain: vec![value, value],
        });
        IntVar(idx)
    }

    /// Add an interval variable defined by start, size, and end expressions.
    ///
    /// Automatically adds the required `start + size == end` and `size >= 0`
    /// constraints (the proto does NOT enforce these implicitly).
    ///
    /// # Panics
    ///
    /// Panics if coefficient arithmetic overflows i64 during constraint generation.
    pub fn new_interval_var(
        &mut self,
        start: impl Into<LinearExpr>,
        size: impl Into<LinearExpr>,
        end: impl Into<LinearExpr>,
        name: impl AsRef<str>,
    ) -> IntervalVar {
        let start_expr = start.into();
        let size_expr = size.into();
        let end_expr = end.into();

        let idx = self.proto.constraints.len() as i32;
        self.proto.constraints.push(ConstraintProto {
            name: name.as_ref().to_string(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::Interval(
                IntervalConstraintProto {
                    start: Some(start_expr.to_proto()),
                    size: Some(size_expr.to_proto()),
                    end: Some(end_expr.to_proto()),
                },
            )),
        });

        // start + size == end  →  start + size - end == 0
        self.add_interval_linking(&start_expr, &size_expr, &end_expr, &[]);

        // size >= 0 (only if not already guaranteed by domain)
        self.add_size_nonneg(&size_expr, &[]);

        IntervalVar(idx)
    }

    /// Add an optional interval variable, active only when `is_present` is true.
    ///
    /// Automatically adds the required `is_present => start + size == end` and
    /// `is_present => size >= 0` constraints.
    pub fn new_optional_interval_var(
        &mut self,
        start: impl Into<LinearExpr>,
        size: impl Into<LinearExpr>,
        end: impl Into<LinearExpr>,
        is_present: BoolVar,
        name: impl AsRef<str>,
    ) -> IntervalVar {
        let start_expr = start.into();
        let size_expr = size.into();
        let end_expr = end.into();
        let enforcement = vec![is_present.index()];

        let idx = self.proto.constraints.len() as i32;
        self.proto.constraints.push(ConstraintProto {
            name: name.as_ref().to_string(),
            enforcement_literal: enforcement.clone(),
            constraint: Some(constraint_proto::Constraint::Interval(
                IntervalConstraintProto {
                    start: Some(start_expr.to_proto()),
                    size: Some(size_expr.to_proto()),
                    end: Some(end_expr.to_proto()),
                },
            )),
        });

        // enforcement => start + size == end
        self.add_interval_linking(&start_expr, &size_expr, &end_expr, &enforcement);

        // enforcement => size >= 0
        self.add_size_nonneg(&size_expr, &enforcement);

        IntervalVar(idx)
    }

    /// Add `start + size - end == 0` (optionally enforced).
    fn add_interval_linking(
        &mut self,
        start: &LinearExpr,
        size: &LinearExpr,
        end: &LinearExpr,
        enforcement: &[i32],
    ) {
        // start + size - end == 0
        // Collect all terms: start terms + size terms + (-end terms)
        let mut vars = Vec::new();
        let mut coeffs = Vec::new();
        for &(v, c) in &start.terms {
            vars.push(v.0);
            coeffs.push(c);
        }
        for &(v, c) in &size.terms {
            vars.push(v.0);
            coeffs.push(c);
        }
        for &(v, c) in &end.terms {
            vars.push(v.0);
            coeffs.push(
                c.checked_neg()
                    .expect("Interval linking: end coefficient overflow on negation"),
            );
        }
        let offset = start
            .constant
            .checked_add(size.constant)
            .and_then(|v| v.checked_sub(end.constant))
            .expect("Interval linking: constant overflow in start + size - end");
        let neg_offset = offset.saturating_neg();

        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: enforcement.to_vec(),
            constraint: Some(constraint_proto::Constraint::Linear(
                LinearConstraintProto {
                    vars,
                    coeffs,
                    domain: vec![neg_offset, neg_offset], // sum(terms) == -offset
                },
            )),
        });
    }

    /// Add `size >= 0` (optionally enforced). Only adds the constraint if
    /// the size expression is not a plain constant >= 0.
    fn add_size_nonneg(&mut self, size: &LinearExpr, enforcement: &[i32]) {
        // If size is a constant >= 0, no constraint needed
        if size.terms.is_empty() && size.constant >= 0 {
            return;
        }
        let vars: Vec<i32> = size.terms.iter().map(|(v, _)| v.0).collect();
        let coeffs: Vec<i64> = size.terms.iter().map(|(_, c)| *c).collect();

        // size >= 0  →  sum(coeffs * vars) + constant >= 0
        //            →  sum(coeffs * vars) >= -constant
        // OR-Tools linear proto supports empty vars (pure constant check).
        let lb = size
            .constant
            .checked_neg()
            .expect("Interval size_nonneg: constant overflow on negation");
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: enforcement.to_vec(),
            constraint: Some(constraint_proto::Constraint::Linear(
                LinearConstraintProto {
                    vars,
                    coeffs,
                    domain: vec![lb, i64::MAX],
                },
            )),
        });
    }

    // ───── Constraints ─────

    /// Add a bounded linear constraint: lb <= expr <= ub.
    ///
    /// # Panics
    ///
    /// Panics if adjusting bounds by the expression constant overflows i64.
    pub fn add(&mut self, bounded: BoundedLinearExpr) -> &mut Self {
        let expr = bounded.expr;
        // Move constant to the bound side: sum(c*v) + k in [lb, ub]
        // becomes sum(c*v) in [lb - k, ub - k]
        let adj_lb = if bounded.lb == i64::MIN {
            i64::MIN
        } else {
            bounded.lb.checked_sub(expr.constant).unwrap_or_else(|| {
                panic!(
                    "Constraint bound overflow: lb={} - constant={} overflows i64",
                    bounded.lb, expr.constant
                )
            })
        };
        let adj_ub = if bounded.ub == i64::MAX {
            i64::MAX
        } else {
            bounded.ub.checked_sub(expr.constant).unwrap_or_else(|| {
                panic!(
                    "Constraint bound overflow: ub={} - constant={} overflows i64",
                    bounded.ub, expr.constant
                )
            })
        };

        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::Linear(
                LinearConstraintProto {
                    vars: expr.terms.iter().map(|(v, _)| v.0).collect(),
                    coeffs: expr.terms.iter().map(|(_, c)| *c).collect(),
                    domain: vec![adj_lb, adj_ub],
                },
            )),
        });
        self
    }

    /// All expressions must take different values.
    pub fn add_all_different<E: Into<LinearExpr>>(
        &mut self,
        exprs: impl IntoIterator<Item = E>,
    ) -> &mut Self {
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::AllDiff(
                AllDifferentConstraintProto {
                    exprs: exprs.into_iter().map(|e| e.into().to_proto()).collect(),
                },
            )),
        });
        self
    }

    /// Intervals must not overlap in time (disjunctive constraint).
    pub fn add_no_overlap<I: Borrow<IntervalVar>>(
        &mut self,
        intervals: impl IntoIterator<Item = I>,
    ) -> &mut Self {
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::NoOverlap(
                NoOverlapConstraintProto {
                    intervals: interval_indices(intervals),
                },
            )),
        });
        self
    }

    /// 2D no-overlap: x_intervals[i] and y_intervals[i] define rectangles
    /// that must not overlap.
    /// # Panics
    ///
    /// Panics if `x_intervals` and `y_intervals` have different lengths.
    pub fn add_no_overlap_2d<X: Borrow<IntervalVar>, Y: Borrow<IntervalVar>>(
        &mut self,
        x_intervals: impl IntoIterator<Item = X>,
        y_intervals: impl IntoIterator<Item = Y>,
    ) -> &mut Self {
        let x_intervals = interval_indices(x_intervals);
        let y_intervals = interval_indices(y_intervals);
        assert_eq!(
            x_intervals.len(),
            y_intervals.len(),
            "add_no_overlap_2d: x_intervals ({}) and y_intervals ({}) must have the same length",
            x_intervals.len(),
            y_intervals.len()
        );
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::NoOverlap2d(
                NoOverlap2DConstraintProto {
                    x_intervals,
                    y_intervals,
                },
            )),
        });
        self
    }

    /// Cumulative constraint: at any point in time, the sum of demands
    /// of active intervals must not exceed capacity.
    /// # Panics
    ///
    /// Panics if `intervals` and `demands` have different lengths.
    pub fn add_cumulative<I: Borrow<IntervalVar>, D: Into<LinearExpr>>(
        &mut self,
        intervals: impl IntoIterator<Item = I>,
        demands: impl IntoIterator<Item = D>,
        capacity: impl Into<LinearExpr>,
    ) -> &mut Self {
        let intervals = interval_indices(intervals);
        let demands: Vec<_> = demands.into_iter().map(|d| d.into().to_proto()).collect();
        assert_eq!(
            intervals.len(),
            demands.len(),
            "add_cumulative: intervals ({}) and demands ({}) must have the same length",
            intervals.len(),
            demands.len()
        );
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::Cumulative(
                CumulativeConstraintProto {
                    intervals,
                    demands,
                    capacity: Some(capacity.into().to_proto()),
                },
            )),
        });
        self
    }

    /// Circuit constraint: find a Hamiltonian circuit.
    /// Each arc is (tail_node, head_node, literal).
    pub fn add_circuit<A: Borrow<(i32, i32, BoolVar)>>(
        &mut self,
        arcs: impl IntoIterator<Item = A>,
    ) -> &mut Self {
        let mut tails = vec![];
        let mut heads = vec![];
        let mut literals = vec![];
        for arc in arcs {
            let &(t, h, l) = arc.borrow();
            tails.push(t);
            heads.push(h);
            literals.push(l.index());
        }
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::Circuit(
                CircuitConstraintProto {
                    tails,
                    heads,
                    literals,
                },
            )),
        });
        self
    }

    /// Table constraint: vars must take one of the allowed tuples.
    /// # Panics
    ///
    /// Panics if any tuple has a different length than `vars`.
    pub fn add_table<V, T, X>(
        &mut self,
        vars: impl IntoIterator<Item = V>,
        tuples: impl IntoIterator<Item = T>,
        negated: bool,
    ) -> &mut Self
    where
        V: Borrow<IntVar>,
        T: IntoIterator<Item = X>,
        X: Borrow<i64>,
    {
        let vars = var_indices(vars);
        let mut values = vec![];
        for (i, tuple) in tuples.into_iter().enumerate() {
            let before = values.len();
            values.extend(tuple.into_iter().map(|x| *x.borrow()));
            let len = values.len() - before;
            assert_eq!(
                len,
                vars.len(),
                "add_table: tuple {i} has {len} entries, expected {} (one per variable)",
                vars.len()
            );
        }
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::Table(TableConstraintProto {
                vars,
                values,
                negated,
                exprs: vec![],
            })),
        });
        self
    }

    /// Automaton constraint.
    ///
    /// Each transition is `(tail_state, head_state, label)` matching
    /// the OR-Tools Python API convention.
    pub fn add_automaton<V, F, T>(
        &mut self,
        vars: impl IntoIterator<Item = V>,
        starting_state: i64,
        final_states: impl IntoIterator<Item = F>,
        transitions: impl IntoIterator<Item = T>,
    ) -> &mut Self
    where
        V: Borrow<IntVar>,
        F: Borrow<i64>,
        T: Borrow<(i64, i64, i64)>,
    {
        let mut tails = vec![];
        let mut heads = vec![];
        let mut labels = vec![];
        for transition in transitions {
            let &(tail, head, label) = transition.borrow();
            tails.push(tail);
            heads.push(head);
            labels.push(label);
        }
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::Automaton(
                AutomatonConstraintProto {
                    vars: var_indices(vars),
                    starting_state,
                    final_states: final_states.into_iter().map(|f| *f.borrow()).collect(),
                    transition_tail: tails,
                    transition_head: heads,
                    transition_label: labels,
                    exprs: vec![],
                },
            )),
        });
        self
    }

    /// Element constraint: target == array[index].
    ///
    /// # Panics
    ///
    /// Panics if `array` is empty.
    pub fn add_element<V: Borrow<IntVar>>(
        &mut self,
        index: IntVar,
        array: impl IntoIterator<Item = V>,
        target: IntVar,
    ) -> &mut Self {
        let vars = var_indices(array);
        assert!(!vars.is_empty(), "add_element: array must not be empty");
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::Element(
                ElementConstraintProto {
                    index: index.0,
                    target: target.0,
                    vars,
                    linear_index: None,
                    linear_target: None,
                    exprs: vec![],
                },
            )),
        });
        self
    }

    /// At least one literal must be true.
    ///
    /// # Panics
    ///
    /// Panics if `literals` is empty (would create an always-false constraint).
    pub fn add_bool_or<L: Borrow<BoolVar>>(
        &mut self,
        literals: impl IntoIterator<Item = L>,
    ) -> &mut Self {
        let literals = literal_indices(literals);
        assert!(
            !literals.is_empty(),
            "add_bool_or: literals must not be empty"
        );
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::BoolOr(BoolArgumentProto {
                literals,
            })),
        });
        self
    }

    /// All literals must be true.
    pub fn add_bool_and<L: Borrow<BoolVar>>(
        &mut self,
        literals: impl IntoIterator<Item = L>,
    ) -> &mut Self {
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::BoolAnd(BoolArgumentProto {
                literals: literal_indices(literals),
            })),
        });
        self
    }

    /// Exactly one literal must be true.
    ///
    /// # Panics
    ///
    /// Panics if `literals` is empty (would create an always-false constraint).
    pub fn add_exactly_one<L: Borrow<BoolVar>>(
        &mut self,
        literals: impl IntoIterator<Item = L>,
    ) -> &mut Self {
        let literals = literal_indices(literals);
        assert!(
            !literals.is_empty(),
            "add_exactly_one: literals must not be empty"
        );
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::ExactlyOne(
                BoolArgumentProto { literals },
            )),
        });
        self
    }

    /// At most one literal can be true.
    pub fn add_at_most_one<L: Borrow<BoolVar>>(
        &mut self,
        literals: impl IntoIterator<Item = L>,
    ) -> &mut Self {
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::AtMostOne(BoolArgumentProto {
                literals: literal_indices(literals),
            })),
        });
        self
    }

    /// Implication: if a is true then b must be true.
    pub fn add_implication(&mut self, a: BoolVar, b: BoolVar) -> &mut Self {
        // Encoded as: a => b  ≡  ¬a ∨ b
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![a.index()],
            constraint: Some(constraint_proto::Constraint::BoolAnd(BoolArgumentProto {
                literals: vec![b.index()],
            })),
        });
        self
    }

    /// Max constraint: `target == max(exprs)`.
    ///
    /// Accepts anything convertible to [`LinearExpr`] for both the target and
    /// the arguments, e.g. `&[IntVar]`, `Vec<LinearExpr>`, or `[x + 1, 2 * y]`.
    pub fn add_max_equality<E: Into<LinearExpr>>(
        &mut self,
        target: impl Into<LinearExpr>,
        exprs: impl IntoIterator<Item = E>,
    ) -> &mut Self {
        let target = target.into();
        let exprs: Vec<LinearExpr> = exprs.into_iter().map(Into::into).collect();
        self.push_lin_max(&target, &exprs);
        self
    }

    /// Min constraint: `target == min(exprs)`.
    ///
    /// Encoded as `-target == max(-exprs)`.
    ///
    /// # Panics
    ///
    /// Panics if any coefficient or constant in `target` or `exprs` is
    /// `i64::MIN` (cannot be negated).
    pub fn add_min_equality<E: Into<LinearExpr>>(
        &mut self,
        target: impl Into<LinearExpr>,
        exprs: impl IntoIterator<Item = E>,
    ) -> &mut Self {
        let target = -target.into();
        let exprs: Vec<LinearExpr> = exprs.into_iter().map(|e| -e.into()).collect();
        self.push_lin_max(&target, &exprs);
        self
    }

    fn push_lin_max(&mut self, target: &LinearExpr, exprs: &[LinearExpr]) {
        self.proto.constraints.push(ConstraintProto {
            name: String::new(),
            enforcement_literal: vec![],
            constraint: Some(constraint_proto::Constraint::LinMax(LinearArgumentProto {
                target: Some(target.to_proto()),
                exprs: exprs.iter().map(LinearExpr::to_proto).collect(),
            })),
        });
    }

    // ───── Objective ─────

    /// Minimize the given expression.
    pub fn minimize(&mut self, expr: impl Into<LinearExpr>) {
        let e = expr.into();
        self.proto.objective = Some(CpObjectiveProto {
            vars: e.terms.iter().map(|(v, _)| v.0).collect(),
            coeffs: e.terms.iter().map(|(_, c)| *c).collect(),
            offset: e.constant as f64,
            scaling_factor: 0.0,
            domain: vec![],
            integer_after_offset: 0,
            integer_scaling_factor: 0,
            integer_before_offset: 0,
            scaling_was_exact: false,
        });
    }

    /// Maximize the given expression.
    ///
    /// # Panics
    ///
    /// Panics if any coefficient is `i64::MIN` (cannot be negated).
    pub fn maximize(&mut self, expr: impl Into<LinearExpr>) {
        let e = expr.into();
        // Maximize f(x) = minimize -f(x)
        self.proto.objective = Some(CpObjectiveProto {
            vars: e.terms.iter().map(|(v, _)| v.0).collect(),
            coeffs: e
                .terms
                .iter()
                .map(|(_, c)| {
                    c.checked_neg().unwrap_or_else(|| {
                        panic!("maximize: coefficient {c} cannot be negated (i64 overflow)")
                    })
                })
                .collect(),
            offset: -(e.constant as f64),
            scaling_factor: -1.0,
            domain: vec![],
            integer_after_offset: 0,
            integer_scaling_factor: 0,
            integer_before_offset: 0,
            scaling_was_exact: false,
        });
    }

    // ───── Hints ─────

    /// Provide a solution hint for a variable.
    pub fn add_hint(&mut self, var: IntVar, value: i64) {
        let hint = self
            .proto
            .solution_hint
            .get_or_insert_with(|| PartialVariableAssignment {
                vars: vec![],
                values: vec![],
            });
        hint.vars.push(var.0);
        hint.values.push(value);
    }

    /// Remove all solution hints previously added with [`add_hint`](Self::add_hint).
    pub fn clear_hints(&mut self) {
        self.proto.solution_hint = None;
    }

    // ───── Serialization ─────

    /// Get a reference to the underlying proto.
    pub fn to_proto(&self) -> &CpModelProto {
        &self.proto
    }

    /// Get a mutable reference to the underlying proto (escape hatch).
    ///
    /// # Warning
    ///
    /// Modifying the proto directly (reordering, deleting variables or
    /// constraints) will invalidate all previously returned `IntVar`,
    /// `BoolVar`, and `IntervalVar` handles. Only use this for adding
    /// fields the builder doesn't yet support.
    pub fn raw_proto_mut(&mut self) -> &mut CpModelProto {
        &mut self.proto
    }

    /// Serialize the model to protobuf bytes.
    ///
    /// # Panics
    ///
    /// Cannot panic in practice — prost encode always succeeds on valid proto.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.proto.encoded_len());
        self.proto
            .encode(&mut buf)
            .expect("prost encode cannot fail on valid proto");
        buf
    }

    /// Deserialize a model from protobuf bytes.
    ///
    /// # Errors
    ///
    /// Returns `prost::DecodeError` if the bytes are not a valid `CpModelProto`.
    ///
    /// Note: variable and constraint handles (`IntVar`, `IntervalVar`) from the
    /// original model are valid on the deserialized copy (same indices).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, prost::DecodeError> {
        let proto = CpModelProto::decode(bytes)?;
        Ok(Self { proto })
    }
}

impl Default for CpModel {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CpModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CpModel({} vars, {} constraints{})",
            self.proto.variables.len(),
            self.proto.constraints.len(),
            if self.proto.objective.is_some() {
                ", has objective"
            } else {
                ""
            },
        )
    }
}

fn var_indices<V: Borrow<IntVar>>(vars: impl IntoIterator<Item = V>) -> Vec<i32> {
    vars.into_iter().map(|v| v.borrow().0).collect()
}

fn literal_indices<L: Borrow<BoolVar>>(literals: impl IntoIterator<Item = L>) -> Vec<i32> {
    literals.into_iter().map(|l| l.borrow().index()).collect()
}

fn interval_indices<I: Borrow<IntervalVar>>(intervals: impl IntoIterator<Item = I>) -> Vec<i32> {
    intervals.into_iter().map(|iv| iv.borrow().0).collect()
}
