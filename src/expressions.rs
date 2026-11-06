//! Linear expressions and bounded linear expressions for constraints.

use crate::vars::{BoolVar, IntVar};
use std::borrow::Borrow;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

/// A linear expression: sum(coeff_i * var_i) + constant.
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct LinearExpr {
    pub(crate) terms: Vec<(IntVar, i64)>,
    pub(crate) constant: i64,
}

impl LinearExpr {
    /// Create an expression equal to a constant.
    pub fn constant(val: i64) -> Self {
        Self {
            terms: vec![],
            constant: val,
        }
    }

    /// Create the sum of the given expressions (variables, literals,
    /// constants or other expressions).
    pub fn sum<E: Into<LinearExpr>>(exprs: impl IntoIterator<Item = E>) -> Self {
        exprs.into_iter().sum()
    }

    /// Create a weighted sum of variables.
    ///
    /// # Panics
    ///
    /// Panics if `vars` and `coeffs` have different lengths.
    pub fn weighted_sum<V: Borrow<IntVar>, C: Borrow<i64>>(
        vars: impl IntoIterator<Item = V>,
        coeffs: impl IntoIterator<Item = C>,
    ) -> Self {
        let vars: Vec<IntVar> = vars.into_iter().map(|v| *v.borrow()).collect();
        let coeffs: Vec<i64> = coeffs.into_iter().map(|c| *c.borrow()).collect();
        assert_eq!(
            vars.len(),
            coeffs.len(),
            "weighted_sum: vars ({}) and coeffs ({}) must have the same length",
            vars.len(),
            coeffs.len()
        );
        Self {
            terms: vars.into_iter().zip(coeffs).collect(),
            constant: 0,
        }
    }

    /// Add a term to this expression.
    pub fn add_term(&mut self, var: IntVar, coeff: i64) {
        self.terms.push((var, coeff));
    }

    /// Constrain this expression to be <= upper_bound.
    pub fn le(self, ub: i64) -> BoundedLinearExpr {
        BoundedLinearExpr {
            expr: self,
            lb: i64::MIN,
            ub,
        }
    }

    /// Constrain this expression to be >= lower_bound.
    pub fn ge(self, lb: i64) -> BoundedLinearExpr {
        BoundedLinearExpr {
            expr: self,
            lb,
            ub: i64::MAX,
        }
    }

    /// Constrain this expression to be == value.
    pub fn eq(self, val: i64) -> BoundedLinearExpr {
        BoundedLinearExpr {
            expr: self,
            lb: val,
            ub: val,
        }
    }

    /// Constrain this expression to be in [lb, ub].
    ///
    /// # Panics
    ///
    /// Panics if `lb > ub`.
    pub fn between(self, lb: i64, ub: i64) -> BoundedLinearExpr {
        assert!(lb <= ub, "between: lb ({lb}) must be <= ub ({ub})");
        BoundedLinearExpr { expr: self, lb, ub }
    }

    /// Convert to the proto representation.
    pub(crate) fn to_proto(&self) -> crate::proto::LinearExpressionProto {
        crate::proto::LinearExpressionProto {
            vars: self.terms.iter().map(|(v, _)| v.0).collect(),
            coeffs: self.terms.iter().map(|(_, c)| *c).collect(),
            offset: self.constant,
        }
    }
}

// --- From conversions ---

impl From<IntVar> for LinearExpr {
    fn from(v: IntVar) -> Self {
        Self {
            terms: vec![(v, 1)],
            constant: 0,
        }
    }
}

impl From<BoolVar> for LinearExpr {
    fn from(b: BoolVar) -> Self {
        Self::from(b.as_int_var())
    }
}

impl From<&IntVar> for LinearExpr {
    fn from(v: &IntVar) -> Self {
        Self::from(*v)
    }
}

impl From<&BoolVar> for LinearExpr {
    fn from(b: &BoolVar) -> Self {
        Self::from(*b)
    }
}

impl From<&LinearExpr> for LinearExpr {
    fn from(e: &LinearExpr) -> Self {
        e.clone()
    }
}

impl From<i64> for LinearExpr {
    fn from(val: i64) -> Self {
        Self::constant(val)
    }
}

impl From<i32> for LinearExpr {
    fn from(val: i32) -> Self {
        Self::constant(i64::from(val))
    }
}

// --- Arithmetic operators ---

impl<T: Into<LinearExpr>> AddAssign<T> for LinearExpr {
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.into();
        self.terms.extend(rhs.terms);
        self.constant = self
            .constant
            .checked_add(rhs.constant)
            .expect("LinearExpr constant overflow on addition");
    }
}

impl<T: Into<LinearExpr>> Sum<T> for LinearExpr {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut acc, e| {
            acc += e;
            acc
        })
    }
}

impl Add for LinearExpr {
    type Output = LinearExpr;
    fn add(mut self, rhs: LinearExpr) -> LinearExpr {
        self += rhs;
        self
    }
}

impl Add<IntVar> for LinearExpr {
    type Output = LinearExpr;
    fn add(mut self, rhs: IntVar) -> LinearExpr {
        self.terms.push((rhs, 1));
        self
    }
}

impl Add<BoolVar> for LinearExpr {
    type Output = LinearExpr;
    fn add(mut self, rhs: BoolVar) -> LinearExpr {
        self.terms.push((rhs.as_int_var(), 1));
        self
    }
}

impl Add<i64> for LinearExpr {
    type Output = LinearExpr;
    fn add(mut self, rhs: i64) -> LinearExpr {
        self.constant = self
            .constant
            .checked_add(rhs)
            .expect("LinearExpr constant overflow on addition");
        self
    }
}

impl Sub for LinearExpr {
    type Output = LinearExpr;
    fn sub(mut self, rhs: LinearExpr) -> LinearExpr {
        for (v, c) in rhs.terms {
            self.terms.push((
                v,
                c.checked_neg()
                    .expect("LinearExpr coefficient overflow on negation"),
            ));
        }
        self.constant = self
            .constant
            .checked_sub(rhs.constant)
            .expect("LinearExpr constant overflow on subtraction");
        self
    }
}

impl Sub<IntVar> for LinearExpr {
    type Output = LinearExpr;
    fn sub(mut self, rhs: IntVar) -> LinearExpr {
        self.terms.push((rhs, -1));
        self
    }
}

impl Sub<BoolVar> for LinearExpr {
    type Output = LinearExpr;
    fn sub(self, rhs: BoolVar) -> LinearExpr {
        self - rhs.as_int_var()
    }
}

impl Sub<i64> for LinearExpr {
    type Output = LinearExpr;
    fn sub(mut self, rhs: i64) -> LinearExpr {
        self.constant = self
            .constant
            .checked_sub(rhs)
            .expect("LinearExpr constant overflow on subtraction");
        self
    }
}

impl Neg for LinearExpr {
    type Output = LinearExpr;
    fn neg(mut self) -> LinearExpr {
        for term in &mut self.terms {
            term.1 = term
                .1
                .checked_neg()
                .expect("LinearExpr coefficient overflow on negation");
        }
        self.constant = self
            .constant
            .checked_neg()
            .expect("LinearExpr constant overflow on negation");
        self
    }
}

impl Mul<i64> for LinearExpr {
    type Output = LinearExpr;
    fn mul(mut self, rhs: i64) -> LinearExpr {
        for term in &mut self.terms {
            term.1 = term
                .1
                .checked_mul(rhs)
                .expect("LinearExpr coefficient overflow on multiplication");
        }
        self.constant = self
            .constant
            .checked_mul(rhs)
            .expect("LinearExpr constant overflow on multiplication");
        self
    }
}

// coeff * IntVar → LinearExpr
impl Mul<IntVar> for i64 {
    type Output = LinearExpr;
    fn mul(self, rhs: IntVar) -> LinearExpr {
        LinearExpr {
            terms: vec![(rhs, self)],
            constant: 0,
        }
    }
}

// IntVar + IntVar → LinearExpr
impl Add for IntVar {
    type Output = LinearExpr;
    fn add(self, rhs: IntVar) -> LinearExpr {
        LinearExpr {
            terms: vec![(self, 1), (rhs, 1)],
            constant: 0,
        }
    }
}

// IntVar + i64 → LinearExpr
impl Add<i64> for IntVar {
    type Output = LinearExpr;
    fn add(self, rhs: i64) -> LinearExpr {
        LinearExpr {
            terms: vec![(self, 1)],
            constant: rhs,
        }
    }
}

// IntVar - IntVar → LinearExpr
impl Sub for IntVar {
    type Output = LinearExpr;
    fn sub(self, rhs: IntVar) -> LinearExpr {
        LinearExpr {
            terms: vec![(self, 1), (rhs, -1)],
            constant: 0,
        }
    }
}

// IntVar * i64 → LinearExpr
impl Mul<i64> for IntVar {
    type Output = LinearExpr;
    fn mul(self, rhs: i64) -> LinearExpr {
        LinearExpr {
            terms: vec![(self, rhs)],
            constant: 0,
        }
    }
}

// BoolVar operators delegate to the IntVar ones. Like `From<BoolVar>`, they
// panic on negated literals.
macro_rules! bool_var_binop {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty) => {
        impl $trait<$rhs> for $lhs {
            type Output = LinearExpr;
            fn $method(self, rhs: $rhs) -> LinearExpr {
                $trait::$method(IntVar::from(self), IntVar::from(rhs))
            }
        }
    };
}

bool_var_binop!(Add, add, BoolVar, BoolVar);
bool_var_binop!(Add, add, BoolVar, IntVar);
bool_var_binop!(Add, add, IntVar, BoolVar);
bool_var_binop!(Sub, sub, BoolVar, BoolVar);
bool_var_binop!(Sub, sub, BoolVar, IntVar);
bool_var_binop!(Sub, sub, IntVar, BoolVar);

// BoolVar + i64 → LinearExpr
impl Add<i64> for BoolVar {
    type Output = LinearExpr;
    fn add(self, rhs: i64) -> LinearExpr {
        self.as_int_var() + rhs
    }
}

// BoolVar * i64 → LinearExpr
impl Mul<i64> for BoolVar {
    type Output = LinearExpr;
    fn mul(self, rhs: i64) -> LinearExpr {
        self.as_int_var() * rhs
    }
}

// coeff * BoolVar → LinearExpr
impl Mul<BoolVar> for i64 {
    type Output = LinearExpr;
    fn mul(self, rhs: BoolVar) -> LinearExpr {
        self * rhs.as_int_var()
    }
}

// Remaining var/constant/expression combinations, for both IntVar and BoolVar:
// var ± LinearExpr, var - i64, i64 ± var. They go through `LinearExpr` so
// the overflow checks (and BoolVar negation checks) are shared.
macro_rules! var_expr_ops {
    ($var:ty) => {
        impl Add<LinearExpr> for $var {
            type Output = LinearExpr;
            fn add(self, rhs: LinearExpr) -> LinearExpr {
                LinearExpr::from(self) + rhs
            }
        }

        impl Sub<LinearExpr> for $var {
            type Output = LinearExpr;
            fn sub(self, rhs: LinearExpr) -> LinearExpr {
                LinearExpr::from(self) - rhs
            }
        }

        impl Sub<i64> for $var {
            type Output = LinearExpr;
            fn sub(self, rhs: i64) -> LinearExpr {
                LinearExpr::from(self) - rhs
            }
        }

        impl Add<$var> for i64 {
            type Output = LinearExpr;
            fn add(self, rhs: $var) -> LinearExpr {
                LinearExpr::from(rhs) + self
            }
        }

        impl Sub<$var> for i64 {
            type Output = LinearExpr;
            fn sub(self, rhs: $var) -> LinearExpr {
                LinearExpr::constant(self) - LinearExpr::from(rhs)
            }
        }
    };
}

var_expr_ops!(IntVar);
var_expr_ops!(BoolVar);

/// A bounded linear expression: lb <= expr <= ub.
/// Created by calling `.le()`, `.ge()`, `.eq()`, or `.between()` on a `LinearExpr`.
#[must_use]
pub struct BoundedLinearExpr {
    pub(crate) expr: LinearExpr,
    pub(crate) lb: i64,
    pub(crate) ub: i64,
}

// Convenience: IntVar <= i64 → BoundedLinearExpr
// We can't impl PartialOrd for foreign types, so use free functions.

/// Create a bounded expression: var <= ub.
pub fn leq(var: impl Into<LinearExpr>, ub: i64) -> BoundedLinearExpr {
    var.into().le(ub)
}

/// Create a bounded expression: var >= lb.
pub fn geq(var: impl Into<LinearExpr>, lb: i64) -> BoundedLinearExpr {
    var.into().ge(lb)
}

/// Create a bounded expression: expr == val.
pub fn eq(var: impl Into<LinearExpr>, val: i64) -> BoundedLinearExpr {
    var.into().eq(val)
}
