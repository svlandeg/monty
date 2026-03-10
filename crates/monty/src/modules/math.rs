//! Implementation of Python's `math` module.
//!
//! Provides mathematical functions and constants matching CPython 3.14 behavior
//! and error messages. All functions are pure computations that don't require
//! host involvement, so they return `Value` directly rather than `AttrCallResult`.
//!
//! ## Implemented functions
//!
//! **Rounding**: `floor`, `ceil`, `trunc`
//! **Roots & powers**: `sqrt`, `isqrt`, `cbrt`, `pow`, `exp`, `exp2`, `expm1`
//! **Logarithms**: `log`, `log2`, `log10`, `log1p`
//! **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
//! **Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
//! **Angular**: `degrees`, `radians`
//! **Float properties**: `fabs`, `isnan`, `isinf`, `isfinite`, `copysign`, `isclose`,
//!   `nextafter`, `ulp`
//! **Integer math**: `factorial`, `gcd`, `lcm`, `comb`, `perm`
//! **Modular**: `fmod`, `remainder`, `modf`, `frexp`, `ldexp`
//! **Special**: `gamma`, `lgamma`, `erf`, `erfc`
//!
//! ## Constants
//!
//! `pi`, `e`, `tau`, `inf`, `nan`

use num_bigint::BigInt;
use smallvec::smallvec;

use crate::{
    args::ArgValues,
    bytecode::VM,
    defer_drop, defer_drop_mut,
    exception_private::{ExcType, RunResult, SimpleException},
    heap::{Heap, HeapData, HeapId},
    intern::{Interns, StaticStrings},
    modules::ModuleFunctions,
    resource::{ResourceError, ResourceTracker},
    types::{LongInt, Module, PyTrait, allocate_tuple},
    value::Value,
};

// ==========================
// Shared constants and error helpers
// ==========================

/// Lanczos approximation coefficients for g=7, n=9 (from Paul Godfrey's tables).
///
/// Used by both `gamma_impl` and `lgamma_impl` to compute the Lanczos series.
#[expect(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    reason = "Lanczos coefficients require full precision and exact values"
)]
const LANCZOS_P: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1259.139_216_722_402_8,
    771.323_428_777_653_08,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Lanczos `g` parameter — controls the trade-off between accuracy and convergence.
const LANCZOS_G: f64 = 7.0;

/// Precomputed `sqrt(2π)` for the Lanczos gamma computation.
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Precomputed `0.5 * ln(2π)` for the Lanczos lgamma computation.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

/// Returns a `ValueError` with the standard CPython "math domain error" message.
fn math_domain_error() -> crate::exception_private::RunError {
    SimpleException::new_msg(ExcType::ValueError, "math domain error").into()
}

/// Returns an `OverflowError` with the standard CPython "math range error" message.
fn math_range_error() -> crate::exception_private::RunError {
    SimpleException::new_msg(ExcType::OverflowError, "math range error").into()
}

/// Checks whether a computation overflowed (finite input produced infinite result).
///
/// Returns `Err(OverflowError("math range error"))` if `result` is infinite
/// but `input` was finite.
fn check_range_error(result: f64, input: f64) -> RunResult<()> {
    if result.is_infinite() && input.is_finite() {
        Err(math_range_error())
    } else {
        Ok(())
    }
}

/// Checks that a value is in the `[-1, 1]` range, raising `ValueError` if not.
///
/// NaN passes through (it will propagate through the subsequent math operation).
/// Used by `math.asin` and `math.acos`.
fn require_unit_range(f: f64) -> RunResult<()> {
    if !f.is_nan() && !(-1.0..=1.0).contains(&f) {
        Err(SimpleException::new_msg(
            ExcType::ValueError,
            format!("expected a number in range from -1 up to 1, got {f:?}"),
        )
        .into())
    } else {
        Ok(())
    }
}

/// Checks for non-positive integer arguments (poles of the Gamma function).
///
/// These are the finite non-positive integers where Gamma diverges to ±∞.
/// Does NOT reject `-inf` — callers that need to reject it (like `math.gamma`)
/// must do so separately, since `lgamma(-inf)` is valid and returns `inf`.
#[expect(
    clippy::float_cmp,
    reason = "exact comparison detects integer poles of gamma function"
)]
fn check_gamma_pole(f: f64) -> RunResult<()> {
    if f <= 0.0 && f == f.floor() && f.is_finite() {
        Err(SimpleException::new_msg(
            ExcType::ValueError,
            format!("expected a noninteger or positive integer, got {f:?}"),
        )
        .into())
    } else {
        Ok(())
    }
}

/// Computes the Lanczos series sum for the given `z = x - 1`.
///
/// This is the shared core of both `lanczos_gamma` and `lanczos_lgamma`.
/// Returns `(sum, t)` where `t = z + G + 0.5`.
fn lanczos_series(z: f64) -> (f64, f64) {
    let mut sum = LANCZOS_P[0];
    for (i, &coeff) in LANCZOS_P.iter().enumerate().skip(1) {
        sum += coeff / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    (sum, t)
}

/// Evaluates the erf/erfc rational polynomial for the `0.84375 ≤ |x| < 1.25` range.
///
/// Returns `P(s) / Q(s)` where `s = |x| - 1`.
fn erf_range2_poly(abs_x: f64) -> f64 {
    let s = abs_x - 1.0;
    let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
    let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
    p / q
}

/// Math module functions — each variant corresponds to a Python-visible function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::Display, serde::Serialize, serde::Deserialize)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum MathFunctions {
    // Rounding
    Floor,
    Ceil,
    Trunc,
    // Roots & powers
    Sqrt,
    Isqrt,
    Cbrt,
    Pow,
    Exp,
    Exp2,
    Expm1,
    // Logarithms
    Log,
    Log1p,
    Log2,
    Log10,
    // Float properties
    Fabs,
    Isnan,
    Isinf,
    Isfinite,
    Copysign,
    Isclose,
    Nextafter,
    Ulp,
    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    // Hyperbolic
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    // Angular conversion
    Degrees,
    Radians,
    // Integer math
    Factorial,
    Gcd,
    Lcm,
    Comb,
    Perm,
    // Modular / decomposition
    Fmod,
    Remainder,
    Modf,
    Frexp,
    Ldexp,
    // Special functions
    Gamma,
    Lgamma,
    Erf,
    Erfc,
}

/// Creates the `math` module and allocates it on the heap.
///
/// Registers all math functions and constants (`pi`, `e`, `tau`, `inf`, `nan`)
/// matching CPython's `math` module. Functions are registered as
/// `ModuleFunctions::Math` variants.
///
/// # Returns
/// A `HeapId` pointing to the newly allocated module.
///
/// # Panics
/// Panics if the required strings have not been pre-interned during prepare phase.
pub fn create_module(vm: &mut VM<'_, '_, impl ResourceTracker>) -> Result<HeapId, ResourceError> {
    let mut module = Module::new(StaticStrings::Math);

    // Register all math functions
    for (name, func) in MATH_FUNCTIONS {
        module.set_attr(*name, Value::ModuleFunction(ModuleFunctions::Math(*func)), vm);
    }

    // Constants
    module.set_attr(StaticStrings::Pi, Value::Float(std::f64::consts::PI), vm);
    module.set_attr(StaticStrings::MathE, Value::Float(std::f64::consts::E), vm);
    module.set_attr(StaticStrings::Tau, Value::Float(std::f64::consts::TAU), vm);
    module.set_attr(StaticStrings::MathInf, Value::Float(f64::INFINITY), vm);
    module.set_attr(StaticStrings::MathNan, Value::Float(f64::NAN), vm);

    vm.heap.allocate(HeapData::Module(module))
}

/// Static mapping of attribute names to math functions for module creation.
const MATH_FUNCTIONS: &[(StaticStrings, MathFunctions)] = &[
    // Rounding
    (StaticStrings::Floor, MathFunctions::Floor),
    (StaticStrings::Ceil, MathFunctions::Ceil),
    (StaticStrings::Trunc, MathFunctions::Trunc),
    // Roots & powers
    (StaticStrings::Sqrt, MathFunctions::Sqrt),
    (StaticStrings::Isqrt, MathFunctions::Isqrt),
    (StaticStrings::Cbrt, MathFunctions::Cbrt),
    (StaticStrings::Pow, MathFunctions::Pow),
    (StaticStrings::Exp, MathFunctions::Exp),
    (StaticStrings::Exp2, MathFunctions::Exp2),
    (StaticStrings::Expm1, MathFunctions::Expm1),
    // Logarithms
    (StaticStrings::Log, MathFunctions::Log),
    (StaticStrings::Log1p, MathFunctions::Log1p),
    (StaticStrings::Log2, MathFunctions::Log2),
    (StaticStrings::Log10, MathFunctions::Log10),
    // Float properties
    (StaticStrings::Fabs, MathFunctions::Fabs),
    (StaticStrings::Isnan, MathFunctions::Isnan),
    (StaticStrings::Isinf, MathFunctions::Isinf),
    (StaticStrings::Isfinite, MathFunctions::Isfinite),
    (StaticStrings::Copysign, MathFunctions::Copysign),
    (StaticStrings::Isclose, MathFunctions::Isclose),
    (StaticStrings::Nextafter, MathFunctions::Nextafter),
    (StaticStrings::Ulp, MathFunctions::Ulp),
    // Trigonometric
    (StaticStrings::Sin, MathFunctions::Sin),
    (StaticStrings::Cos, MathFunctions::Cos),
    (StaticStrings::Tan, MathFunctions::Tan),
    (StaticStrings::Asin, MathFunctions::Asin),
    (StaticStrings::Acos, MathFunctions::Acos),
    (StaticStrings::Atan, MathFunctions::Atan),
    (StaticStrings::Atan2, MathFunctions::Atan2),
    // Hyperbolic
    (StaticStrings::Sinh, MathFunctions::Sinh),
    (StaticStrings::Cosh, MathFunctions::Cosh),
    (StaticStrings::Tanh, MathFunctions::Tanh),
    (StaticStrings::Asinh, MathFunctions::Asinh),
    (StaticStrings::Acosh, MathFunctions::Acosh),
    (StaticStrings::Atanh, MathFunctions::Atanh),
    // Angular conversion
    (StaticStrings::Degrees, MathFunctions::Degrees),
    (StaticStrings::Radians, MathFunctions::Radians),
    // Integer math
    (StaticStrings::Factorial, MathFunctions::Factorial),
    (StaticStrings::Gcd, MathFunctions::Gcd),
    (StaticStrings::Lcm, MathFunctions::Lcm),
    (StaticStrings::Comb, MathFunctions::Comb),
    (StaticStrings::Perm, MathFunctions::Perm),
    // Modular / decomposition
    (StaticStrings::Fmod, MathFunctions::Fmod),
    (StaticStrings::Remainder, MathFunctions::Remainder),
    (StaticStrings::Modf, MathFunctions::Modf),
    (StaticStrings::Frexp, MathFunctions::Frexp),
    (StaticStrings::Ldexp, MathFunctions::Ldexp),
    // Special functions
    (StaticStrings::Gamma, MathFunctions::Gamma),
    (StaticStrings::Lgamma, MathFunctions::Lgamma),
    (StaticStrings::Erf, MathFunctions::Erf),
    (StaticStrings::Erfc, MathFunctions::Erfc),
];

/// Dispatches a call to a math module function.
///
/// All math functions are pure computations and return `Value` directly.
pub(super) fn call(
    vm: &mut VM<'_, '_, impl ResourceTracker>,
    function: MathFunctions,
    args: ArgValues,
) -> RunResult<Value> {
    match function {
        // Rounding
        MathFunctions::Floor => math_floor(vm.heap, args),
        MathFunctions::Ceil => math_ceil(vm.heap, args),
        MathFunctions::Trunc => math_trunc(vm.heap, args),
        // Roots & powers
        MathFunctions::Sqrt => math_sqrt(vm.heap, args),
        MathFunctions::Isqrt => math_isqrt(vm.heap, args),
        MathFunctions::Cbrt => math_cbrt(vm.heap, args),
        MathFunctions::Pow => math_pow(vm.heap, args),
        MathFunctions::Exp => math_exp(vm.heap, args),
        MathFunctions::Exp2 => math_exp2(vm.heap, args),
        MathFunctions::Expm1 => math_expm1(vm.heap, args),
        // Logarithms
        MathFunctions::Log => math_log(vm.heap, args),
        MathFunctions::Log1p => math_log1p(vm.heap, args),
        MathFunctions::Log2 => math_log2(vm.heap, args),
        MathFunctions::Log10 => math_log10(vm.heap, args),
        // Float properties
        MathFunctions::Fabs => math_fabs(vm.heap, args),
        MathFunctions::Isnan => math_isnan(vm.heap, args),
        MathFunctions::Isinf => math_isinf(vm.heap, args),
        MathFunctions::Isfinite => math_isfinite(vm.heap, args),
        MathFunctions::Copysign => math_copysign(vm.heap, args),
        MathFunctions::Isclose => math_isclose(vm.heap, args, vm.interns),
        MathFunctions::Nextafter => math_nextafter(vm.heap, args),
        MathFunctions::Ulp => math_ulp(vm.heap, args),
        // Trigonometric
        MathFunctions::Sin => math_sin(vm.heap, args),
        MathFunctions::Cos => math_cos(vm.heap, args),
        MathFunctions::Tan => math_tan(vm.heap, args),
        MathFunctions::Asin => math_asin(vm.heap, args),
        MathFunctions::Acos => math_acos(vm.heap, args),
        MathFunctions::Atan => math_atan(vm.heap, args),
        MathFunctions::Atan2 => math_atan2(vm.heap, args),
        // Hyperbolic
        MathFunctions::Sinh => math_sinh(vm.heap, args),
        MathFunctions::Cosh => math_cosh(vm.heap, args),
        MathFunctions::Tanh => math_tanh(vm.heap, args),
        MathFunctions::Asinh => math_asinh(vm.heap, args),
        MathFunctions::Acosh => math_acosh(vm.heap, args),
        MathFunctions::Atanh => math_atanh(vm.heap, args),
        // Angular conversion
        MathFunctions::Degrees => math_degrees(vm.heap, args),
        MathFunctions::Radians => math_radians(vm.heap, args),
        // Integer math
        MathFunctions::Factorial => math_factorial(vm.heap, args),
        MathFunctions::Gcd => math_gcd(vm.heap, args),
        MathFunctions::Lcm => math_lcm(vm.heap, args),
        MathFunctions::Comb => math_comb(vm.heap, args),
        MathFunctions::Perm => math_perm(vm.heap, args),
        // Modular / decomposition
        MathFunctions::Fmod => math_fmod(vm.heap, args),
        MathFunctions::Remainder => math_remainder(vm.heap, args),
        MathFunctions::Modf => math_modf(vm.heap, args),
        MathFunctions::Frexp => math_frexp(vm.heap, args),
        MathFunctions::Ldexp => math_ldexp(vm.heap, args),
        // Special functions
        MathFunctions::Gamma => math_gamma(vm.heap, args),
        MathFunctions::Lgamma => math_lgamma(vm.heap, args),
        MathFunctions::Erf => math_erf(vm.heap, args),
        MathFunctions::Erfc => math_erfc(vm.heap, args),
    }
}

// ==========================
// Rounding functions
// ==========================

/// `math.floor(x)` — returns the largest integer less than or equal to x.
///
/// Accepts int, float, or bool. Returns int.
/// Raises `OverflowError` for infinity, `ValueError` for NaN.
fn math_floor(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.floor", heap)?;
    defer_drop!(value, heap);

    match value {
        Value::Float(f) => float_to_int_checked(f.floor(), *f, heap),
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Bool(b) => Ok(Value::Int(i64::from(*b))),
        _ => Err(ExcType::type_error(format!(
            "must be real number, not {}",
            value.py_type(heap)
        ))),
    }
}

/// `math.ceil(x)` — returns the smallest integer greater than or equal to x.
///
/// Accepts int, float, or bool. Returns int.
/// Raises `OverflowError` for infinity, `ValueError` for NaN.
fn math_ceil(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.ceil", heap)?;
    defer_drop!(value, heap);

    match value {
        Value::Float(f) => float_to_int_checked(f.ceil(), *f, heap),
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Bool(b) => Ok(Value::Int(i64::from(*b))),
        _ => Err(ExcType::type_error(format!(
            "must be real number, not {}",
            value.py_type(heap)
        ))),
    }
}

/// `math.trunc(x)` — truncates x to the nearest integer toward zero.
///
/// Accepts int, float, or bool. Returns int.
fn math_trunc(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.trunc", heap)?;
    defer_drop!(value, heap);

    match value {
        Value::Float(f) => float_to_int_checked(f.trunc(), *f, heap),
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Bool(b) => Ok(Value::Int(i64::from(*b))),
        _ => Err(ExcType::type_error(format!(
            "type {} doesn't define __trunc__ method",
            value.py_type(heap)
        ))),
    }
}

// ==========================
// Roots & powers
// ==========================

/// `math.sqrt(x)` — returns the square root of x.
///
/// Always returns a float. Raises `ValueError` for negative inputs with a
/// descriptive message matching CPython 3.14: "expected a nonnegative input, got <x>".
fn math_sqrt(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.sqrt", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    if f < 0.0 {
        Err(SimpleException::new_msg(ExcType::ValueError, format!("expected a nonnegative input, got {f:?}")).into())
    } else {
        Ok(Value::Float(f.sqrt()))
    }
}

/// `math.isqrt(n)` — returns the integer square root of a non-negative integer.
///
/// Returns the largest integer `r` such that `r * r <= n`.
/// Only accepts non-negative integers (and bools).
fn math_isqrt(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.isqrt", heap)?;
    defer_drop!(value, heap);

    let n = value_to_int(value, heap)?;
    if n < 0 {
        return Err(SimpleException::new_msg(ExcType::ValueError, "isqrt() argument must be nonnegative").into());
    }
    if n == 0 {
        return Ok(Value::Int(0));
    }

    // Integer square root via f64 estimate + correction.
    // For i64 inputs, f64 sqrt is accurate to within ±1, so we need to
    // correct both overshoot and undershoot. The cast truncates toward zero,
    // so undershoot is possible for perfect squares near f64 precision limits.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "initial estimate doesn't need to be exact, correction refines it"
    )]
    let mut x = (n as f64).sqrt() as i64;
    // Correct overshoot: use `x > n / x` instead of `x * x > n` to avoid i64 overflow.
    while x > n / x {
        x -= 1;
    }
    // Correct undershoot: check if (x+1)² ≤ n using division to avoid overflow.
    while x < n / (x + 1) {
        x += 1;
    }
    Ok(Value::Int(x))
}

/// `math.cbrt(x)` — returns the cube root of x.
///
/// Always returns a float. Unlike `sqrt`, works for negative inputs.
fn math_cbrt(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.cbrt", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(f.cbrt()))
}

/// `math.pow(x, y)` — returns x raised to the power y.
///
/// Always returns a float. Unlike the builtin `pow()`, does not support
/// three-argument modular exponentiation. Raises `ValueError` for
/// negative base with non-integer exponent.
fn math_pow(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (x_val, y_val) = args.get_two_args("math.pow", heap)?;
    defer_drop!(x_val, heap);
    defer_drop!(y_val, heap);

    let x = value_to_float(x_val, heap)?;
    let y = value_to_float(y_val, heap)?;
    let result = x.powf(y);
    // CPython raises ValueError for domain errors: 0**negative, negative**non-integer
    if result.is_nan() && !x.is_nan() && !y.is_nan() {
        return Err(math_domain_error());
    }
    if result.is_infinite() && x.is_finite() && y.is_finite() {
        // 0**negative is a domain error (ValueError), not overflow
        if x == 0.0 && y < 0.0 {
            return Err(math_domain_error());
        }
        return Err(math_range_error());
    }
    Ok(Value::Float(result))
}

/// `math.exp(x)` — returns e raised to the power x.
fn math_exp(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.exp", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    let result = f.exp();
    check_range_error(result, f)?;
    Ok(Value::Float(result))
}

/// `math.exp2(x)` — returns 2 raised to the power x.
fn math_exp2(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.exp2", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    let result = f.exp2();
    check_range_error(result, f)?;
    Ok(Value::Float(result))
}

/// `math.expm1(x)` — returns e**x - 1.
///
/// More accurate than `exp(x) - 1` for small values of x.
fn math_expm1(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.expm1", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    let result = f.exp_m1();
    check_range_error(result, f)?;
    Ok(Value::Float(result))
}

// ==========================
// Logarithms
// ==========================

/// `math.log(x[, base])` — returns the logarithm of x.
///
/// With one argument, returns the natural logarithm (base e).
/// With two arguments, returns `log(x) / log(base)`.
/// Raises `ValueError` for non-positive inputs (CPython 3.14: "expected a positive input").
fn math_log(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (x_val, base_val) = args.get_one_two_args("math.log", heap)?;
    defer_drop!(x_val, heap);
    defer_drop!(base_val, heap);

    let x = value_to_float(x_val, heap)?;
    if x <= 0.0 {
        return Err(SimpleException::new_msg(ExcType::ValueError, "expected a positive input").into());
    }

    match base_val {
        Some(base_v) => {
            let base = value_to_float(base_v, heap)?;
            // base == 1.0 causes division by zero in log(x)/log(base), matching
            // CPython which raises ZeroDivisionError for this case.
            #[expect(
                clippy::float_cmp,
                reason = "exact comparison with 1.0 is intentional — log(1.0) is exactly 0.0"
            )]
            if base == 1.0 {
                return Err(SimpleException::new_msg(ExcType::ZeroDivisionError, "division by zero").into());
            }
            if base <= 0.0 {
                return Err(SimpleException::new_msg(ExcType::ValueError, "expected a positive input").into());
            }
            Ok(Value::Float(x.ln() / base.ln()))
        }
        None => Ok(Value::Float(x.ln())),
    }
}

/// `math.log1p(x)` — returns the natural logarithm of 1 + x.
///
/// More accurate than `log(1 + x)` for small values of x.
/// CPython 3.14 raises ValueError with "expected argument value > -1, got <x>".
fn math_log1p(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.log1p", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    if f <= -1.0 {
        return Err(
            SimpleException::new_msg(ExcType::ValueError, format!("expected argument value > -1, got {f:?}")).into(),
        );
    }
    Ok(Value::Float(f.ln_1p()))
}

/// `math.log2(x)` — returns the base-2 logarithm of x.
///
/// Returns `inf` for positive infinity, `nan` for NaN.
/// Raises `ValueError` for non-positive finite inputs.
fn math_log2(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.log2", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    if f <= 0.0 {
        Err(SimpleException::new_msg(ExcType::ValueError, "expected a positive input").into())
    } else {
        Ok(Value::Float(f.log2()))
    }
}

/// `math.log10(x)` — returns the base-10 logarithm of x.
///
/// Returns `inf` for positive infinity, `nan` for NaN.
/// Raises `ValueError` for non-positive finite inputs.
fn math_log10(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.log10", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    if f <= 0.0 {
        Err(SimpleException::new_msg(ExcType::ValueError, "expected a positive input").into())
    } else {
        Ok(Value::Float(f.log10()))
    }
}

// ==========================
// Float properties
// ==========================

/// `math.fabs(x)` — returns the absolute value as a float.
///
/// Unlike the builtin `abs()`, always returns a float.
fn math_fabs(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.fabs", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(f.abs()))
}

/// `math.isnan(x)` — returns True if x is NaN.
fn math_isnan(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.isnan", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Bool(f.is_nan()))
}

/// `math.isinf(x)` — returns True if x is positive or negative infinity.
fn math_isinf(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.isinf", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Bool(f.is_infinite()))
}

/// `math.isfinite(x)` — returns True if x is neither infinity nor NaN.
fn math_isfinite(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.isfinite", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Bool(f.is_finite()))
}

/// `math.copysign(x, y)` — returns x with the sign of y.
///
/// Always returns a float.
fn math_copysign(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (x_val, y_val) = args.get_two_args("math.copysign", heap)?;
    defer_drop!(x_val, heap);
    defer_drop!(y_val, heap);

    let x = value_to_float(x_val, heap)?;
    let y = value_to_float(y_val, heap)?;
    Ok(Value::Float(x.copysign(y)))
}

/// `math.isclose(a, b, *, rel_tol=1e-9, abs_tol=0.0)` — returns True if a and b are close.
///
/// Supports keyword-only `rel_tol` and `abs_tol` parameters matching CPython.
/// Raises `ValueError` if either tolerance is negative.
fn math_isclose(heap: &mut Heap<impl ResourceTracker>, args: ArgValues, interns: &Interns) -> RunResult<Value> {
    let (positional, kwargs) = args.into_parts();
    defer_drop_mut!(positional, heap);

    // Extract exactly two positional args
    let Some(a_val) = positional.next() else {
        return Err(ExcType::type_error_at_least("math.isclose", 2, 0));
    };
    defer_drop!(a_val, heap);
    let Some(b_val) = positional.next() else {
        return Err(ExcType::type_error_at_least("math.isclose", 2, 1));
    };
    defer_drop!(b_val, heap);
    if positional.len() > 0 {
        return Err(ExcType::type_error_at_most("math.isclose", 2, 2 + positional.len()));
    }

    let a = value_to_float(a_val, heap)?;
    let b = value_to_float(b_val, heap)?;

    // Parse optional keyword arguments rel_tol and abs_tol
    let (rel_tol, abs_tol) = extract_isclose_kwargs(kwargs, heap, interns)?;

    if rel_tol < 0.0 {
        return Err(SimpleException::new_msg(ExcType::ValueError, "tolerances must be non-negative").into());
    }
    if abs_tol < 0.0 {
        return Err(SimpleException::new_msg(ExcType::ValueError, "tolerances must be non-negative").into());
    }

    // Exact equality check matches CPython's isclose() behavior — two identical
    // values (including infinities) are always considered close.
    #[expect(
        clippy::float_cmp,
        reason = "exact equality check matches CPython's isclose() semantics"
    )]
    if a == b {
        return Ok(Value::Bool(true));
    }
    if a.is_infinite() || b.is_infinite() {
        return Ok(Value::Bool(false));
    }
    if a.is_nan() || b.is_nan() {
        return Ok(Value::Bool(false));
    }

    let diff = (a - b).abs();
    let result = diff <= (rel_tol * a.abs().max(b.abs())).max(abs_tol);
    Ok(Value::Bool(result))
}

/// Extracts `rel_tol` and `abs_tol` keyword arguments for `math.isclose`.
///
/// Returns `(rel_tol, abs_tol)` with defaults of `(1e-9, 0.0)`.
fn extract_isclose_kwargs(
    kwargs: crate::args::KwargsValues,
    heap: &mut Heap<impl ResourceTracker>,
    interns: &Interns,
) -> RunResult<(f64, f64)> {
    let mut rel_tol: f64 = 1e-9;
    let mut abs_tol: f64 = 0.0;

    if kwargs.is_empty() {
        return Ok((rel_tol, abs_tol));
    }

    for (key, value) in kwargs {
        defer_drop!(key, heap);
        defer_drop!(value, heap);

        let Some(keyword_name) = key.as_either_str(heap) else {
            return Err(ExcType::type_error("keywords must be strings"));
        };

        let key_str = keyword_name.as_str(interns);
        match key_str {
            "rel_tol" => {
                rel_tol = value_to_float(value, heap)?;
            }
            "abs_tol" => {
                abs_tol = value_to_float(value, heap)?;
            }
            other => {
                return Err(ExcType::type_error(format!(
                    "isclose() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    Ok((rel_tol, abs_tol))
}

/// `math.nextafter(x, y)` — returns the next float after x towards y.
fn math_nextafter(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (x_val, y_val) = args.get_two_args("math.nextafter", heap)?;
    defer_drop!(x_val, heap);
    defer_drop!(y_val, heap);

    let x = value_to_float(x_val, heap)?;
    let y = value_to_float(y_val, heap)?;

    // Use bit manipulation to compute nextafter, matching C's nextafter behavior:
    // - If x == y, return y
    // - If x or y is NaN, return NaN
    // - Otherwise, step x towards y by one ULP
    let result = nextafter_impl(x, y);
    Ok(Value::Float(result))
}

/// `math.ulp(x)` — returns the value of the least significant bit of x.
///
/// For finite non-zero x, returns the smallest float `u` such that `x + u != x`.
/// Special cases: `ulp(nan)` returns nan, `ulp(inf)` returns inf.
fn math_ulp(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.ulp", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    if f.is_nan() {
        return Ok(Value::Float(f64::NAN));
    }
    if f.is_infinite() {
        return Ok(Value::Float(f64::INFINITY));
    }
    let f = f.abs();
    if f == 0.0 {
        // CPython returns the smallest positive subnormal: 5e-324
        return Ok(Value::Float(f64::from_bits(1)));
    }
    // ULP = nextafter(f, inf) - f
    let next = nextafter_impl(f, f64::INFINITY);
    Ok(Value::Float(next - f))
}

// ==========================
// Trigonometric functions
// ==========================

/// `math.sin(x)` — returns the sine of x (in radians).
///
/// CPython 3.14 raises ValueError for infinity: "expected a finite input, got inf".
fn math_sin(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.sin", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    require_finite(f)?;
    Ok(Value::Float(f.sin()))
}

/// `math.cos(x)` — returns the cosine of x (in radians).
fn math_cos(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.cos", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    require_finite(f)?;
    Ok(Value::Float(f.cos()))
}

/// `math.tan(x)` — returns the tangent of x (in radians).
fn math_tan(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.tan", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    require_finite(f)?;
    Ok(Value::Float(f.tan()))
}

/// `math.asin(x)` — returns the arc sine of x (in radians).
///
/// CPython 3.14: "expected a number in range from -1 up to 1, got <x>".
fn math_asin(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.asin", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    require_unit_range(f)?;
    Ok(Value::Float(f.asin()))
}

/// `math.acos(x)` — returns the arc cosine of x (in radians).
///
/// CPython 3.14: "expected a number in range from -1 up to 1, got <x>".
fn math_acos(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.acos", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    require_unit_range(f)?;
    Ok(Value::Float(f.acos()))
}

/// `math.atan(x)` — returns the arc tangent of x (in radians).
fn math_atan(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.atan", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(f.atan()))
}

/// `math.atan2(y, x)` — returns atan(y/x) in radians, using the signs of both
/// to determine the correct quadrant.
fn math_atan2(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (y_val, x_val) = args.get_two_args("math.atan2", heap)?;
    defer_drop!(y_val, heap);
    defer_drop!(x_val, heap);

    let y = value_to_float(y_val, heap)?;
    let x = value_to_float(x_val, heap)?;
    Ok(Value::Float(y.atan2(x)))
}

// ==========================
// Hyperbolic functions
// ==========================

/// `math.sinh(x)` — returns the hyperbolic sine of x.
fn math_sinh(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.sinh", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    let result = f.sinh();
    check_range_error(result, f)?;
    Ok(Value::Float(result))
}

/// `math.cosh(x)` — returns the hyperbolic cosine of x.
fn math_cosh(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.cosh", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    let result = f.cosh();
    check_range_error(result, f)?;
    Ok(Value::Float(result))
}

/// `math.tanh(x)` — returns the hyperbolic tangent of x.
fn math_tanh(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.tanh", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(f.tanh()))
}

/// `math.asinh(x)` — returns the inverse hyperbolic sine of x.
fn math_asinh(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.asinh", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(f.asinh()))
}

/// `math.acosh(x)` — returns the inverse hyperbolic cosine of x.
///
/// CPython 3.14: "expected argument value not less than 1, got <x>".
fn math_acosh(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.acosh", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    if f < 1.0 {
        return Err(SimpleException::new_msg(
            ExcType::ValueError,
            format!("expected argument value not less than 1, got {f:?}"),
        )
        .into());
    }
    Ok(Value::Float(f.acosh()))
}

/// `math.atanh(x)` — returns the inverse hyperbolic tangent of x.
///
/// CPython 3.14: "expected a number between -1 and 1, got <x>".
fn math_atanh(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.atanh", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    if f <= -1.0 || f >= 1.0 {
        return Err(SimpleException::new_msg(
            ExcType::ValueError,
            format!("expected a number between -1 and 1, got {f:?}"),
        )
        .into());
    }
    Ok(Value::Float(f.atanh()))
}

// ==========================
// Angular conversion
// ==========================

/// `math.degrees(x)` — converts angle x from radians to degrees.
fn math_degrees(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.degrees", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(f.to_degrees()))
}

/// `math.radians(x)` — converts angle x from degrees to radians.
fn math_radians(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.radians", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(f.to_radians()))
}

// ==========================
// Integer math
// ==========================

/// `math.factorial(n)` — returns n factorial.
///
/// Only accepts non-negative integers (and bools). Raises `ValueError` for
/// negative values, `TypeError` for non-integer types.
fn math_factorial(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.factorial", heap)?;
    defer_drop!(value, heap);

    let n = match value {
        Value::Int(n) => *n,
        Value::Bool(b) => i64::from(*b),
        _ => {
            return Err(ExcType::type_error(format!(
                "'{}' object cannot be interpreted as an integer",
                value.py_type(heap)
            )));
        }
    };

    if n < 0 {
        return Err(
            SimpleException::new_msg(ExcType::ValueError, "factorial() not defined for negative values").into(),
        );
    }

    // Compute factorial iteratively
    let mut result: i64 = 1;
    for i in 2..=n {
        match result.checked_mul(i) {
            Some(v) => result = v,
            None => {
                // Overflow — for simplicity, return an error for very large factorials
                // since we don't have LongInt factorial support yet
                return Err(
                    SimpleException::new_msg(ExcType::OverflowError, "int too large to convert to factorial").into(),
                );
            }
        }
    }
    Ok(Value::Int(result))
}

/// `math.gcd(*integers)` — returns the greatest common divisor of the arguments.
///
/// Supports 0 or more arguments, matching CPython 3.9+. `gcd()` returns 0,
/// `gcd(n)` returns `abs(n)`, and for multiple args reduces pairwise.
/// The result is always non-negative.
fn math_gcd(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let positional = args.into_pos_only("math.gcd", heap)?;
    defer_drop_mut!(positional, heap);

    let mut result: u64 = 0;
    for arg in positional.by_ref() {
        defer_drop!(arg, heap);
        let n = value_to_int(arg, heap)?;
        result = gcd(result, n.unsigned_abs());
    }
    u64_to_value(result, heap)
}

/// `math.lcm(*integers)` — returns the least common multiple of the arguments.
///
/// Supports 0 or more arguments, matching CPython 3.9+. `lcm()` returns 1,
/// `lcm(n)` returns `abs(n)`, and for multiple args reduces pairwise.
/// The result is always non-negative. Returns 0 if any argument is 0.
fn math_lcm(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let positional = args.into_pos_only("math.lcm", heap)?;
    defer_drop_mut!(positional, heap);

    let mut result: u64 = 1;
    for arg in positional.by_ref() {
        defer_drop!(arg, heap);
        let n = value_to_int(arg, heap)?;
        let abs_n = n.unsigned_abs();
        if abs_n == 0 {
            return Ok(Value::Int(0));
        }
        let g = gcd(result, abs_n);
        // lcm(a, b) = |a| / gcd(a,b) * |b| — dividing first avoids intermediate overflow
        result = (result / g)
            .checked_mul(abs_n)
            .ok_or_else(|| SimpleException::new_msg(ExcType::OverflowError, "integer overflow in lcm"))?;
    }
    u64_to_value(result, heap)
}

/// `math.comb(n, k)` — returns the number of ways to choose k items from n.
///
/// Both arguments must be non-negative integers.
fn math_comb(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (n_val, k_val) = args.get_two_args("math.comb", heap)?;
    defer_drop!(n_val, heap);
    defer_drop!(k_val, heap);

    let n = value_to_int(n_val, heap)?;
    let k = value_to_int(k_val, heap)?;

    if n < 0 {
        return Err(SimpleException::new_msg(ExcType::ValueError, "n must be a non-negative integer").into());
    }
    if k < 0 {
        return Err(SimpleException::new_msg(ExcType::ValueError, "k must be a non-negative integer").into());
    }
    if k > n {
        return Ok(Value::Int(0));
    }

    // Use the smaller of k and n-k for efficiency: C(n, k) = C(n, n-k)
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        // Use GCD reduction to keep intermediates small:
        // result = result * (n - i) / (i + 1)
        // By dividing both numerator and denominator by their GCD first,
        // we reduce the chance of overflow in the multiplication step.
        let mut numerator = n - i;
        let mut denominator = i + 1;
        #[expect(clippy::cast_sign_loss, reason = "both values are known non-negative at this point")]
        let g = gcd(numerator as u64, denominator as u64).cast_signed();
        numerator /= g;
        denominator /= g;
        // Also reduce against the running result
        #[expect(clippy::cast_sign_loss, reason = "result and denominator are known non-negative")]
        let g2 = gcd(result as u64, denominator as u64).cast_signed();
        result /= g2;
        denominator /= g2;
        debug_assert!(denominator == 1, "denominator should be 1 after GCD reduction in comb");
        match result.checked_mul(numerator) {
            Some(v) => result = v,
            None => {
                return Err(SimpleException::new_msg(ExcType::OverflowError, "integer overflow in comb").into());
            }
        }
    }
    Ok(Value::Int(result))
}

/// `math.perm(n, k=None)` — returns the number of k-length permutations from n items.
///
/// Both arguments must be non-negative integers. When `k` is omitted, defaults to `n`
/// (i.e., `perm(n)` returns `n!`), matching CPython behavior.
fn math_perm(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (n_val, k_val) = args.get_one_two_args("math.perm", heap)?;
    defer_drop!(n_val, heap);

    let n = value_to_int(n_val, heap)?;
    let k_explicit = k_val.is_some();
    let k = match k_val {
        Some(kv) => {
            defer_drop!(kv, heap);
            value_to_int(kv, heap)?
        }
        None => n,
    };

    if n < 0 {
        // When called as perm(n) without k, CPython uses the factorial error message
        let msg = if k_explicit {
            "n must be a non-negative integer"
        } else {
            "factorial() not defined for negative values"
        };
        return Err(SimpleException::new_msg(ExcType::ValueError, msg).into());
    }
    if k < 0 {
        return Err(SimpleException::new_msg(ExcType::ValueError, "k must be a non-negative integer").into());
    }
    if k > n {
        return Ok(Value::Int(0));
    }

    let mut result: i64 = 1;
    for i in 0..k {
        match result.checked_mul(n - i) {
            Some(v) => result = v,
            None => {
                return Err(SimpleException::new_msg(ExcType::OverflowError, "integer overflow in perm").into());
            }
        }
    }
    Ok(Value::Int(result))
}

// ==========================
// Modular / decomposition
// ==========================

/// `math.fmod(x, y)` — returns x modulo y as a float.
///
/// Unlike `x % y`, the result has the same sign as x. Raises `ValueError`
/// when y is zero (CPython: "math domain error").
fn math_fmod(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (x_val, y_val) = args.get_two_args("math.fmod", heap)?;
    defer_drop!(x_val, heap);
    defer_drop!(y_val, heap);

    let x = value_to_float(x_val, heap)?;
    let y = value_to_float(y_val, heap)?;

    if y == 0.0 || x.is_infinite() {
        // CPython raises for both fmod(x, 0) and fmod(inf, y)
        // but NaN inputs propagate
        if !x.is_nan() && !y.is_nan() {
            return Err(math_domain_error());
        }
    }
    Ok(Value::Float(x % y))
}

/// `math.remainder(x, y)` — IEEE 754 remainder of x with respect to y.
///
/// The result is `x - n*y` where n is the closest integer to `x/y`.
fn math_remainder(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (x_val, y_val) = args.get_two_args("math.remainder", heap)?;
    defer_drop!(x_val, heap);
    defer_drop!(y_val, heap);

    let x = value_to_float(x_val, heap)?;
    let y = value_to_float(y_val, heap)?;

    // NaN propagates
    if x.is_nan() || y.is_nan() {
        return Ok(Value::Float(f64::NAN));
    }
    if y == 0.0 {
        return Err(math_domain_error());
    }
    if x.is_infinite() {
        return Err(math_domain_error());
    }
    if y.is_infinite() {
        return Ok(Value::Float(x));
    }

    // IEEE 754 remainder: result = x - round_half_even(x/y) * y
    let n = round_half_even(x / y);
    let result = x - n * y;
    Ok(Value::Float(result))
}

/// `math.modf(x)` — returns the fractional and integer parts of x as a tuple.
///
/// Both values carry the sign of x. Returns `(fractional, integer)`.
fn math_modf(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.modf", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;

    // Special cases: modf(inf) = (0.0, inf), modf(nan) = (nan, nan)
    if f.is_nan() {
        let tuple = allocate_tuple(smallvec![Value::Float(f64::NAN), Value::Float(f64::NAN)], heap)?;
        return Ok(tuple);
    }
    if f.is_infinite() {
        // The fractional part is ±0.0 (signed to match the input sign)
        let frac = if f > 0.0 { 0.0 } else { -0.0_f64 };
        let tuple = allocate_tuple(smallvec![Value::Float(frac), Value::Float(f)], heap)?;
        return Ok(tuple);
    }

    let integer = f.trunc();
    // Preserve the sign of the input on the fractional part — e.g. modf(-0.0)
    // should return (-0.0, -0.0), not (0.0, -0.0). Using copysign ensures the
    // fractional part carries the correct sign even when it's zero.
    let fractional = (f - integer).copysign(f);
    let tuple = allocate_tuple(smallvec![Value::Float(fractional), Value::Float(integer)], heap)?;
    Ok(tuple)
}

/// `math.frexp(x)` — returns (mantissa, exponent) such that `x == mantissa * 2**exponent`.
///
/// The mantissa is always in the range [0.5, 1.0) or zero.
/// Returns a tuple `(float, int)`.
#[expect(
    clippy::cast_possible_wrap,
    reason = "IEEE 754 bit manipulation requires u64-to-i64 casts for exponent values masked to 11 bits"
)]
fn math_frexp(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.frexp", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;

    if f == 0.0 || f.is_nan() || f.is_infinite() {
        // Special cases: frexp(0) = (0.0, 0), frexp(nan) = (nan, 0), frexp(inf) = (inf, 0)
        let tuple = allocate_tuple(smallvec![Value::Float(f), Value::Int(0)], heap)?;
        return Ok(tuple);
    }

    // Decompose using bit manipulation of IEEE 754 representation
    let bits = f.to_bits();
    let sign = bits & (1u64 << 63);
    let exponent_bits = ((bits >> 52) & 0x7ff) as i64;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;

    if exponent_bits == 0 {
        // Subnormal: multiply by 2^53 to normalize, then adjust
        let normalized = f * (1u64 << 53) as f64;
        let n_bits = normalized.to_bits();
        let n_exp = ((n_bits >> 52) & 0x7ff) as i64;
        let n_mant = n_bits & 0x000f_ffff_ffff_ffff;
        // Exponent: (biased_exp - 1022) gives the frexp exponent for normal numbers,
        // minus 53 to compensate for the 2^53 normalization factor
        let exp = n_exp - 1022 - 53;
        let m = f64::from_bits(sign | (0x3fe_u64 << 52) | n_mant);
        let tuple = allocate_tuple(smallvec![Value::Float(m), Value::Int(exp)], heap)?;
        return Ok(tuple);
    }

    // For normal numbers: frexp exponent = biased_exponent - 1022
    // (1022 = IEEE 754 bias 1023 minus 1, since mantissa is in [0.5, 1.0) not [1.0, 2.0))
    let exp = exponent_bits - 1022;
    let m = f64::from_bits(sign | (0x3fe_u64 << 52) | mantissa_bits);

    let tuple = allocate_tuple(smallvec![Value::Float(m), Value::Int(exp)], heap)?;
    Ok(tuple)
}

/// `math.ldexp(x, i)` — returns `x * 2**i`, the inverse of `frexp`.
#[expect(
    clippy::cast_sign_loss,
    reason = "exponent values are validated to be in range before casting"
)]
fn math_ldexp(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let (x_val, i_val) = args.get_two_args("math.ldexp", heap)?;
    defer_drop!(x_val, heap);
    defer_drop!(i_val, heap);

    let x = value_to_float(x_val, heap)?;
    let i = value_to_int(i_val, heap)?;

    // Special cases: inf/nan/zero pass through regardless of exponent
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return Ok(Value::Float(x));
    }

    // Clamp extreme exponents to bound the loop iterations. IEEE 754 double precision
    // has exponents from -1074 (smallest subnormal) to +1023 (largest finite). A finite
    // float `x` has exponent in [-1074, 1023], so the result exponent is `exp_x + i`.
    // If `i > 2100`, even the smallest subnormal (exp -1074) would overflow to infinity.
    // If `i < -2100`, even the largest finite (exp 1023) would underflow to zero.
    // Clamping to ±2100 is safe and limits the loop to at most ~3 iterations.
    let mut result = x;
    let mut exp = i.clamp(-2100, 2100);
    while exp > 0 {
        let step = exp.min(1023);
        result *= f64::from_bits(((1023 + step) as u64) << 52);
        exp -= step;
    }
    while exp < 0 {
        let step = (-exp).min(1022);
        result *= f64::from_bits(((1023 - step) as u64) << 52);
        exp += step;
    }

    // If the result overflowed to infinity, CPython raises OverflowError
    if result.is_infinite() {
        return Err(math_range_error());
    }

    Ok(Value::Float(result))
}

// ==========================
// Special functions
// ==========================

/// `math.gamma(x)` — returns the Gamma function at x.
///
/// CPython 3.14 raises ValueError for non-positive integers:
/// "expected a noninteger or positive integer, got <x>".
fn math_gamma(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.gamma", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    // CPython also rejects -inf for gamma (but not lgamma, where lgamma(-inf) = inf)
    if f == f64::NEG_INFINITY {
        return Err(SimpleException::new_msg(
            ExcType::ValueError,
            format!("expected a noninteger or positive integer, got {f:?}"),
        )
        .into());
    }
    check_gamma_pole(f)?;

    let result = gamma_impl(f);
    check_range_error(result, f)?;
    Ok(Value::Float(result))
}

/// `math.lgamma(x)` — returns the natural log of the absolute value of Gamma(x).
fn math_lgamma(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.lgamma", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    check_gamma_pole(f)?;

    let result = lgamma_impl(f);
    check_range_error(result, f)?;
    Ok(Value::Float(result))
}

/// `math.erf(x)` — returns the error function at x.
fn math_erf(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.erf", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(erf_impl(f)))
}

/// `math.erfc(x)` — returns the complementary error function at x (1 - erf(x)).
///
/// More accurate than `1 - erf(x)` for large x.
fn math_erfc(heap: &mut Heap<impl ResourceTracker>, args: ArgValues) -> RunResult<Value> {
    let value = args.get_one_arg("math.erfc", heap)?;
    defer_drop!(value, heap);

    let f = value_to_float(value, heap)?;
    Ok(Value::Float(erfc_impl(f)))
}

// ==========================
// Helper functions
// ==========================

/// Converts a rounded float to an integer `Value`, checking for infinity/NaN.
///
/// `rounded` is the already-rounded float value (e.g., from `floor()`, `ceil()`, `trunc()`).
/// `original` is the original input float, used only to determine the error type:
/// infinity produces `OverflowError`, NaN produces `ValueError`.
///
/// For finite values outside the i64 range, promotes to `LongInt` to match CPython's
/// behavior of returning arbitrary-precision integers from `math.floor`/`ceil`/`trunc`.
fn float_to_int_checked(rounded: f64, original: f64, heap: &mut Heap<impl ResourceTracker>) -> RunResult<Value> {
    if original.is_infinite() {
        Err(SimpleException::new_msg(ExcType::OverflowError, "cannot convert float infinity to integer").into())
    } else if original.is_nan() {
        Err(SimpleException::new_msg(ExcType::ValueError, "cannot convert float NaN to integer").into())
    } else if rounded >= i64::MIN as f64 && rounded < i64::MAX as f64 {
        // Note: `i64::MAX as f64` rounds up to 2^63 (9223372036854775808.0), so we use
        // strict less-than to exclude that value. `i64::MIN as f64` is exact (-2^63).
        #[expect(
            clippy::cast_possible_truncation,
            reason = "intentional: value is within i64 range after bounds check"
        )]
        let result = rounded as i64;
        Ok(Value::Int(result))
    } else {
        // Value exceeds i64 range — promote to LongInt.
        // Format with no decimal places and parse as BigInt. This is correct because
        // `rounded` is already an integer-valued float from floor/ceil/trunc.
        let s = format!("{rounded:.0}");
        let bi = s
            .parse::<BigInt>()
            .map_err(|_| SimpleException::new_msg(ExcType::ValueError, "float too large to convert to integer"))?;
        Ok(LongInt::new(bi).into_value(heap)?)
    }
}

/// Converts a `Value` to `f64`, raising `TypeError` if the value is not numeric.
///
/// Accepts `Float`, `Int`, and `Bool` values. For other types, raises a `TypeError`
/// with a message matching CPython's format: "must be real number, not <type>".
#[expect(
    clippy::cast_precision_loss,
    reason = "i64-to-f64 can lose precision for large integers (beyond 2^53), but this matches CPython's conversion semantics"
)]
fn value_to_float(value: &Value, heap: &Heap<impl ResourceTracker>) -> RunResult<f64> {
    match value {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => Err(ExcType::type_error(format!(
            "must be real number, not {}",
            value.py_type(heap)
        ))),
    }
}

/// Converts a `Value` to `i64`, raising `TypeError` if the value is not an integer.
///
/// Accepts `Int` and `Bool` values. For other types, raises a `TypeError`
/// with a message matching CPython's format.
fn value_to_int(value: &Value, heap: &Heap<impl ResourceTracker>) -> RunResult<i64> {
    match value {
        Value::Int(n) => Ok(*n),
        Value::Bool(b) => Ok(i64::from(*b)),
        _ => Err(ExcType::type_error(format!(
            "'{}' object cannot be interpreted as an integer",
            value.py_type(heap)
        ))),
    }
}

/// Requires that a float is finite, raising ValueError if it's inf or nan.
///
/// CPython 3.14 uses "expected a finite input, got inf" for trig functions.
fn require_finite(f: f64) -> RunResult<()> {
    if f.is_infinite() {
        Err(SimpleException::new_msg(ExcType::ValueError, format!("expected a finite input, got {f:?}")).into())
    } else {
        Ok(())
    }
}

/// Euclidean GCD algorithm for unsigned 64-bit integers.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Converts a `u64` result to a `Value`, promoting to `LongInt` if it exceeds `i64::MAX`.
///
/// This is needed for operations like `gcd(i64::MIN, 0)` where the unsigned result
/// (`2^63`) doesn't fit in a signed `i64`.
fn u64_to_value(n: u64, heap: &mut Heap<impl ResourceTracker>) -> RunResult<Value> {
    if let Ok(signed) = i64::try_from(n) {
        Ok(Value::Int(signed))
    } else {
        Ok(LongInt::new(BigInt::from(n)).into_value(heap)?)
    }
}

/// Rounds a float using "round half to even" (banker's rounding).
///
/// When the fractional part is exactly 0.5, rounds to the nearest even integer.
/// This matches IEEE 754 rounding behavior used by `math.remainder`.
#[expect(clippy::float_cmp, reason = "exact comparison needed for halfway detection")]
fn round_half_even(x: f64) -> f64 {
    let rounded = x.round();
    // Check if we're exactly at a halfway point
    if (x - rounded).abs() == 0.5 {
        // Round to even: if rounded is odd, go the other way
        let truncated = x.trunc();
        if truncated % 2.0 == 0.0 { truncated } else { rounded }
    } else {
        rounded
    }
}

/// Computes `nextafter(x, y)` — the next representable float from x towards y.
///
/// Matches C's `nextafter` behavior: if x == y returns y, NaN propagates.
#[expect(clippy::float_cmp, reason = "exact comparison is correct for nextafter semantics")]
fn nextafter_impl(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        // Step from zero towards y: smallest subnormal with sign of y
        return if y > 0.0 {
            f64::from_bits(1)
        } else {
            f64::from_bits(1 | (1u64 << 63))
        };
    }
    let bits = x.to_bits();
    let result_bits = if (x < y) == (x > 0.0) { bits + 1 } else { bits - 1 };
    f64::from_bits(result_bits)
}

/// Lanczos approximation of the Gamma function for positive arguments.
///
/// Uses a 7-term Lanczos series with g=7, which provides ~15 digits of
/// precision for positive real arguments. For negative non-integer arguments,
/// uses the reflection formula: Γ(x) = π / (sin(πx) · Γ(1-x)).
#[expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "mathematical function needs exact comparisons and integer factorial computation"
)]
fn gamma_impl(x: f64) -> f64 {
    // Note: NEG_INFINITY and non-positive integers are handled by `math_gamma` before
    // calling this function, so we don't need to check for them here.
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    // For positive integers, return exact factorial
    if x > 0.0 && x == x.floor() && x <= 21.0 {
        let n = x as u64;
        let mut result: u64 = 1;
        for i in 2..n {
            result *= i;
        }
        return result as f64;
    }
    if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) · Γ(1-x))
        let sin_px = (std::f64::consts::PI * x).sin();
        return std::f64::consts::PI / (sin_px * gamma_impl(1.0 - x));
    }
    lanczos_gamma(x)
}

/// Lanczos series computation for Γ(x) where x >= 0.5.
fn lanczos_gamma(x: f64) -> f64 {
    let z = x - 1.0;
    let (sum, t) = lanczos_series(z);
    SQRT_2PI * t.powf(z + 0.5) * (-t).exp() * sum
}

/// Computes ln(|Γ(x)|) using the Lanczos approximation.
#[expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "mathematical function needs exact comparisons and integer factorial computation"
)]
fn lgamma_impl(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        if x == f64::NEG_INFINITY {
            return f64::INFINITY;
        }
        return x.abs();
    }
    // Exact results for small positive integers: lgamma(n) = ln((n-1)!)
    if x > 0.0 && x == x.floor() && x <= 23.0 {
        let n = x as u64;
        let mut fact: u64 = 1;
        for i in 2..n {
            fact *= i;
        }
        return (fact as f64).ln();
    }
    if x < 0.5 {
        // Reflection: ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|
        // Note: non-positive integers (where sin(πx) == 0) are handled by `math_lgamma`
        // before calling this function.
        let sin_px = (std::f64::consts::PI * x).sin().abs();
        return std::f64::consts::PI.ln() - sin_px.ln() - lgamma_impl(1.0 - x);
    }
    lanczos_lgamma(x)
}

/// Lanczos series computation for ln(Γ(x)) where x >= 0.5.
fn lanczos_lgamma(x: f64) -> f64 {
    let z = x - 1.0;
    let (sum, t) = lanczos_series(z);
    HALF_LN_2PI + (z + 0.5) * t.ln() - t + sum.ln()
}

/// Error function with full double precision (~15 significant digits).
///
/// Uses a piecewise rational polynomial approximation derived from the
/// Sun Microsystems / FreeBSD implementation (also used by musl and glibc).
/// The algorithm splits the domain into ranges with tailored rational
/// approximations for each:
///
/// - `|x| < 0.84375`: Direct rational approximation `x + x * P(x²)/Q(x²)`
/// - `0.84375 ≤ |x| < 1.25`: Approximation around `erf(1) ≈ 0.8450629`
/// - `1.25 ≤ |x| < 28`: Computed via `erfc` with rational approximations
/// - `|x| ≥ 28`: Returns ±1 (erfc is below f64 precision)
fn erf_impl(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let abs_x = x.abs();
    let sign = x.signum();

    if abs_x < 0.843_75 {
        // |x| < 0.84375: erf(x) = x + x * P(x²) / Q(x²)
        if abs_x < f64::from_bits(0x3E30_0000_0000_0000) {
            // |x| < 2^-28: erf(x) ≈ x * (2/√π) to avoid underflow
            return x + EFX * x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        x + x * (r / s)
    } else if abs_x < 1.25 {
        // 0.84375 ≤ |x| < 1.25: erf(x) = erx + P1(|x|-1) / Q1(|x|-1)
        sign * (ERX + erf_range2_poly(abs_x))
    } else if abs_x >= 28.0 {
        sign
    } else {
        // 1.25 ≤ |x| < 28: compute via erfc
        sign * (1.0 - erfc_inner(abs_x))
    }
}

/// Complementary error function with full double precision (~15 significant digits).
///
/// Uses the same piecewise rational polynomial as `erf_impl`. Avoids catastrophic
/// cancellation for large `|x|` by computing `erfc` directly rather than `1 - erf(x)`.
///
/// The algorithm uses range-specific rational approximations:
/// - `|x| < 0.84375`: Computed via `1 - erf(x)` (safe, no cancellation)
/// - `0.84375 ≤ |x| < 1.25`: Direct approximation around `erfc(1) ≈ 0.1549370`
/// - `1.25 ≤ |x| < 28`: `erfc(x) = exp(-x²) * R(1/x²) / S(1/x²)` with two sub-ranges
/// - `|x| ≥ 28`: Returns 0 (positive x) or 2 (negative x)
fn erfc_impl(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let abs_x = x.abs();

    if abs_x < 0.843_75 {
        // Small x: no cancellation risk, safe to compute 1 - erf(x) directly
        return 1.0 - erf_impl(x);
    }
    if abs_x >= 28.0 {
        return if x < 0.0 { 2.0 } else { 0.0 };
    }

    let result = if abs_x < 1.25 {
        // 0.84375 ≤ |x| < 1.25
        let pq = erf_range2_poly(abs_x);
        if x < 0.0 { 1.0 + ERX + pq } else { 1.0 - ERX - pq }
    } else {
        erfc_inner(abs_x)
    };

    if x < 0.0 { 2.0 - result } else { result }
}

/// Inner erfc computation for `1.25 ≤ |x| < 28` using rational polynomial
/// approximation of exp(x²) · erfc(x).
///
/// Uses two sub-ranges with different coefficient sets for optimal accuracy:
/// - `1.25 ≤ |x| < 1/0.35 ≈ 2.857`: Coefficients RA0-RA7 / SA1-SA8
/// - `2.857 ≤ |x| < 28`: Coefficients RB0-RB6 / SB1-SB7
///
/// The result is computed as `exp(-x² - 0.5625) * exp(correction) * R/S`
/// where the exp is split to preserve precision.
fn erfc_inner(abs_x: f64) -> f64 {
    let s = 1.0 / (abs_x * abs_x);
    let (r, sv) = if abs_x < 1.0 / 0.35 {
        // 1.25 ≤ |x| < ~2.857
        let r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        let sv = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
        (r, sv)
    } else {
        // 2.857 ≤ |x| < 28
        let r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        let sv = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
        (r, sv)
    };
    // Split exp(-x²) into exp(-z²) * exp(z²-x²) for precision.
    // Zero the low 32 bits of abs_x to get z (the "high word" trick).
    let z = f64::from_bits(abs_x.to_bits() & 0xFFFF_FFFF_0000_0000);
    (-z * z - 0.5625).exp() * ((z - abs_x) * (z + abs_x) + r / sv).exp() / abs_x
}

// =============================
// erf/erfc rational polynomial coefficients
// =============================
// From Sun Microsystems / FreeBSD libm (s_erf.c), also used by musl and glibc.
// These provide full double-precision accuracy (~15 significant digits).

// Coefficients from Sun Microsystems / FreeBSD libm (s_erf.c), also used by musl and glibc.
// Trailing digits are kept exactly as the reference to guarantee bit-identical constants.
// Some constants have trailing zeros that trigger clippy::excessive_precision but are
// preserved for traceability back to the reference implementation.

/// erf(0.84375) — the precomputed value at the first range boundary.
const ERX: f64 = 8.450_629_115_104_675e-01;

/// Coefficient for tiny-x approximation: 2/√π - 1.
const EFX: f64 = 1.283_791_670_955_125_7e-01;

// --- Range 1: |x| < 0.84375 ---
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const PP0: f64 = 1.283_791_670_955_125_59e-01;
const PP1: f64 = -3.250_421_072_470_015e-01;
const PP2: f64 = -2.848_174_957_559_851e-02;
const PP3: f64 = -5.770_270_296_489_442e-03;
const PP4: f64 = -2.376_301_665_665_016_3e-05;
const QQ1: f64 = 3.979_172_239_591_554e-01;
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const QQ2: f64 = 6.502_224_998_876_729e-02;
const QQ3: f64 = 5.081_306_281_875_766e-03;
const QQ4: f64 = 1.324_947_380_043_216e-04;
const QQ5: f64 = -3.960_228_278_775_368e-06;

// --- Range 2: 0.84375 ≤ |x| < 1.25 ---
const PA0: f64 = -2.362_118_560_752_659e-03;
const PA1: f64 = 4.148_561_186_837_483e-01;
const PA2: f64 = -3.722_078_760_357_013e-01;
const PA3: f64 = 3.183_466_199_011_618e-01;
const PA4: f64 = -1.108_946_942_823_967e-01;
const PA5: f64 = 3.547_830_432_561_824e-02;
const PA6: f64 = -2.166_375_594_868_791e-03;
const QA1: f64 = 1.064_208_804_008_442e-01;
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const QA2: f64 = 5.403_979_177_021_710e-01;
const QA3: f64 = 7.182_865_441_419_627e-02;
const QA4: f64 = 1.261_712_198_087_616e-01;
const QA5: f64 = 1.363_708_391_202_905e-02;
const QA6: f64 = 1.198_449_984_679_911e-02;

// --- Range 3: 1.25 ≤ |x| < ~2.857 ---
const RA0: f64 = -9.864_944_034_847_148e-03;
const RA1: f64 = -6.938_585_727_071_818e-01;
const RA2: f64 = -1.055_862_622_532_329_1e+01;
const RA3: f64 = -6.237_533_245_032_601e+01;
const RA4: f64 = -1.623_966_694_625_735e+02;
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const RA5: f64 = -1.846_050_929_067_110e+02;
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const RA6: f64 = -8.128_743_550_630_659e+01;
const RA7: f64 = -9.814_329_344_169_145e+00;
const SA1: f64 = 1.965_127_166_743_926e+01;
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const SA2: f64 = 1.376_577_541_435_190e+02;
const SA3: f64 = 4.345_658_774_752_292e+02;
const SA4: f64 = 6.453_872_717_332_679e+02;
const SA5: f64 = 4.290_081_400_275_678e+02;
const SA6: f64 = 1.086_350_055_417_794e+02;
const SA7: f64 = 6.570_249_770_319_282e+00;
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const SA8: f64 = -6.042_441_521_485_810e-02;

// --- Range 4: ~2.857 ≤ |x| < 28 ---
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const RB0: f64 = -9.864_942_924_700_099e-03;
#[expect(clippy::excessive_precision, reason = "FreeBSD libm reference constant")]
const RB1: f64 = -7.992_832_376_805_230e-01;
const RB2: f64 = -1.775_795_491_775_475_2e+01;
const RB3: f64 = -1.606_363_848_558_219_2e+02;
const RB4: f64 = -6.375_664_433_683_896e+02;
const RB5: f64 = -1.025_095_131_611_077e+03;
const RB6: f64 = -4.835_191_916_086_514e+02;
const SB1: f64 = 3.033_806_074_348_246e+01;
const SB2: f64 = 3.257_925_129_965_739e+02;
const SB3: f64 = 1.536_729_586_084_437e+03;
const SB4: f64 = 3.199_858_219_508_596e+03;
const SB5: f64 = 2.553_050_406_433_164e+03;
const SB6: f64 = 4.745_285_412_069_554e+02;
const SB7: f64 = -2.244_095_244_658_582e+01;
