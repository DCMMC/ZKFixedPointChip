use std::{ops::Sub, fmt::Debug};
use halo2_base::{
    utils::{ScalarField, BigPrimeField, biguint_to_fe, fe_to_biguint}, gates::{GateChip, GateInstructions, RangeChip, range::RangeStrategy, RangeInstructions},
    QuantumCell, Context, AssignedValue
};
use halo2_base::QuantumCell::{Constant, Existing, Witness};
use num_bigint::{BigUint};
use num_integer::Integer;


#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FixedPointStrategy {
    /// Vanilla implementation using vertical basic gate(s).
    Vertical,
}

/// A ZK-friendly fixed-point arithmetic chip over the BN254 scalar field.
///
/// `PRECISION_BITS` (P) controls both integer and fractional width: P.P format.
/// For example, `PRECISION_BITS = 32` gives 32.32 fixed-point (64-bit total),
/// and `PRECISION_BITS = 63` gives 63.63 fixed-point (126-bit total).
///
/// # Representation
/// - Real value `x` is stored as `x_q = round(x * 2^P)` in the field.
/// - Negative values use modular representation: `-x` is stored as `p - x_q` where `p` is the BN254 prime.
/// - Valid range: `(-2^{2P}, 2^{2P})`.
///
/// # Precision
/// The `qexp2` and `qlog2` functions use Higher-Precision (HP) polynomial evaluation
/// with `EXTRA_PRECISION_BITS` (K=20) extra internal bits to achieve >= P bits of
/// precision (max 1 ULP error) for both P=32 and P=63 configurations.
/// Coefficients are pre-computed with 100 decimal digits of precision using Chebyshev
/// approximation, avoiding the ~52-bit limitation of f64.
#[derive(Clone, Debug)]
pub struct FixedPointChip<F: BigPrimeField, const PRECISION_BITS: u32> {
    strategy: FixedPointStrategy,
    pub gate: RangeChip<F>,
    pub quantization_scale: F,
    pub max_value: BigUint,
    pub bn254_max: F,
    pub negative_point: F,
    pub lookup_bits: usize,
    pub pow_of_two: Vec<F>
}

impl<F: BigPrimeField, const PRECISION_BITS: u32> FixedPointChip<F, PRECISION_BITS> {
    pub fn new(strategy: FixedPointStrategy, lookup_bits: usize) -> Self {
        assert!(PRECISION_BITS <= 63, "support only precision bits <= 63");
        assert!(PRECISION_BITS >= 32, "support only precision bits >= 32");
        let gate = RangeChip::new(
            match strategy {
                FixedPointStrategy::Vertical => RangeStrategy::Vertical,
            },
            lookup_bits
        );
        // Simple uniform symmetric quantization scheme which enforces zero point to be exactly 0
        // to reduce lots of computations.
        // Quantization: x_q = xS where S is `quantization_scale`
        // De-quantization: x = x_q / S
        let quantization_scale = F::from_u128(2u128.pow(PRECISION_BITS as u32));
        // Becuase BN254 is cyclic, negative number will be denoted as (-x) % m = m - x where m = 2^254,
        // in this chip, we treat all x > negative_point as a negative numbers.
        let bn254_max = biguint_to_fe(&BigUint::parse_bytes(
            &F::MODULUS[2..].bytes().collect::<Vec<u8>>(), 16).unwrap().sub(1u32));
        // -max_value % m = negative_point
        let negative_point = bn254_max - F::from_u128(2u128.pow(PRECISION_BITS * 2 + 1)) + F::one();
        // min_value < x < max_value
        let max_value = BigUint::from(2u32).pow(PRECISION_BITS * 2);

        let mut pow_of_two = Vec::with_capacity(F::NUM_BITS as usize);
        let two = F::from(2);
        pow_of_two.push(F::one());
        pow_of_two.push(two);
        for _ in 2..F::NUM_BITS {
            pow_of_two.push(two * pow_of_two.last().unwrap());
        }

        Self { strategy, gate, quantization_scale, max_value, bn254_max, negative_point, lookup_bits, pow_of_two }
    }

    pub fn default(lookup_bits: usize) -> Self {
        Self::new(FixedPointStrategy::Vertical, lookup_bits)
    }

    pub fn quantization(&self, x: f64) -> F {
        let sign = x.signum();
        let x = x.abs();
        let x_q = (x * self.quantization_scale.get_lower_128() as f64).round() as u128;
        let x_q_biguint = BigUint::from(x_q).to_bytes_le();
        let mut x_q_bytes_le = [0u8; 64];
        for (idx, val) in x_q_biguint.iter().enumerate() {
            x_q_bytes_le[idx] = *val;
        }
        let mut x_q_f = F::from_bytes_wide(&x_q_bytes_le);
        if sign < 0.0 {
            x_q_f = self.bn254_max - x_q_f + F::one();
        }

        x_q_f
    }

    pub fn dequantization(&self, x: F) -> f64 {
        let mut x_mut = x;
        let negative = if x > self.negative_point {
            x_mut = self.bn254_max - x - F::one();
            -1f64
        } else {
            1f64
        };
        let x_u128: u128 = x_mut.get_lower_128();
        let quantization_scale = self.quantization_scale.get_lower_128();
        let x_int = (x_u128 / quantization_scale) as f64;
        let x_frac = (x_u128 % quantization_scale) as f64 / quantization_scale as f64;
        let x_deq = negative * (x_int + x_frac);

        x_deq
    }

    /// Extra bits used for Higher-Precision (HP) polynomial evaluation.
    /// During Horner evaluation, each multiply-then-truncate step loses ~1 ULP.
    /// With degree-d polynomial, this accumulates to ~d ULPs at P-bit precision.
    /// By working at (P+K)-bit precision internally and truncating once at the end,
    /// the per-step error becomes negligible at P-bit scale.
    /// K=20 keeps intermediates within BN254: 2*(P+K) = 2*(63+20) = 166 < 254.
    const EXTRA_PRECISION_BITS: u32 = 20;

    /// Convert a signed i128 coefficient to a field element.
    /// Negative values are represented as `p - |val|` in the BN254 field.
    fn quantize_i128(&self, val: i128) -> F {
        if val >= 0 {
            biguint_to_fe(&BigUint::from(val as u128))
        } else {
            self.bn254_max - biguint_to_fe::<F>(&BigUint::from((-val) as u128)) + F::one()
        }
    }

    /// Chebyshev polynomial coefficients for exp2(t) on [0, 1] at HP scale (2^{P+K}).
    /// Pre-quantized as integers with mpmath (100 decimal digits precision).
    /// - P<=32: degree 12 (13 terms), HP_SCALE=2^52, verified >= 32 bits precision
    /// - P>32:  degree 16 (17 terms), HP_SCALE=2^83, verified >= 63 bits precision
    fn generate_exp2_poly_hp(&self) -> Vec<QuantumCell<F>> {
        let coef_i128: Vec<i128> = if PRECISION_BITS <= 32 {
            // degree 12, HP_SCALE=2^52
            vec![
                163898, 1855089, 32025673, 458102292, 5951930881,
                68692103124, 693713372880, 6004900741814, 43316202666321,
                249968283100316, 1081884007225527, 3121657384082680,
                4503599627370496,
            ]
        } else {
            // degree 16, HP_SCALE=2^83
            vec![
                1859160554, 28050891277, 660997987636, 13232945686538,
                248356396920884, 4299452683312243, 68230923496384621,
                984364075377840504, 12781234560285917319,
                147515389723298731308, 1489738048462252400249,
                12895426168440252968521, 93020836916792629615618,
                536802800476790956263267, 2323328214549521983009611,
                6703708186976009930559257, 9671406556917033397649408,
            ]
        };
        coef_i128.iter().map(|&c| Constant(self.quantize_i128(c))).collect()
    }

    /// Chebyshev polynomial coefficients for log2(1+t) on [0, 1] at HP scale (2^{P+K}).
    /// Used in qlog2: after normalizing input to [2, 4), substitute t = x/2 - 1 so t in [0, 1),
    /// then log2(x) = 1 + poly(t).
    /// Pre-quantized as integers with mpmath (100 decimal digits precision).
    /// - P<=32: degree 22 (23 terms), HP_SCALE=2^52, verified >= 32 bits precision
    /// - P>32:  degree 29 (30 terms), HP_SCALE=2^83, verified >= 63 bits precision
    fn generate_log_poly_hp(&self) -> Vec<QuantumCell<F>> {
        let coef_i128: Vec<i128> = if PRECISION_BITS <= 32 {
            // degree 22, HP_SCALE=2^52
            vec![
                -72714530283, 913853260841, -5470741880415, 20845750791055,
                -57111260767282, 120828996766082, -207841871597824,
                303624723506539, -392412810390578, 466627212833677,
                -528752577204712, 586692679652559, -648726562848844,
                721721666458897, -812133047794269, 928184813884460,
                -1082886459384203, 1299464147350276, -1624330211174166,
                2165773616159927, -3248660424278035, 6497320848556796, 0,
            ]
        } else {
            // degree 29, HP_SCALE=2^83
            vec![
                8489630466307435614, -136264558136253235350,
                1050526704828427793816, -5189834027395357318753,
                18506692453020330767193, -50926565676127844703261,
                113060535939451201505032, -209440667972691351779979,
                333122647536372942841436, -467190200783459387677135,
                593095012913227390548973, -699534786007381648647137,
                785655430331027083568081, -858016718324957451357692,
                925384172747064501184695, -995233525936403540442198,
                1072954147630102434451654, -1162669663058552030975272,
                1268432389421105617469566, -1395287320139384762195722,
                1550320948804107476170624, -1744111267449001152614350,
                1993270038527229370667010, -2325481712951963447868217,
                2790578055614722117309453, -3488222569521243587856957,
                4650963426028401854480036, -6976445139042604142949689,
                13952890278085208301077931, 14,
            ]
        };
        coef_i128.iter().map(|&c| Constant(self.quantize_i128(c))).collect()
    }

    fn generate_sin_poly(&self) -> Vec<QuantumCell<F>> {
        // generated by lolremez -d 14  -r "0:pi" "sin(x)"
        // Estimated max error: 1.9323057584419826e-15
        let coef: Vec<F> = [
            -1.1008071636607462e-11, 2.4208013888629323e-10,
            -3.8584805817996712e-10, -2.3786993104309845e-08,
            -2.9795813710683115e-09, 2.7608543130047009e-06,
            -6.4467066994122565e-09, -0.00019840680551418068,
            -3.839555844512214e-09, 0.0083333350601673614,
            -5.0943769725466814e-10, -0.16666666657583049,
            -8.5029878414113731e-12, 1.0000000000003146,
            -1.9323057584419828e-15
        ].into_iter().map(|c| self.quantization(c)).collect();

        coef.iter().map(|x| Constant(*x)).collect()
    }
}

/// Trait defining fixed-point decimal arithmetic operations in ZK circuits.
///
/// All operations work on values in P.P fixed-point format where P = `PRECISION_BITS`.
/// Values are quantized as `x_q = round(x * 2^P)` and stored as BN254 field elements.
///
/// # References
/// - [FixPointCS](https://github.com/XMunkki/FixPointCS/blob/c701f57c3cfe6478d1f6fd7578ae040c59386b3d/Cpp/Fixed64.h)
/// - [ABDKMath64x64](https://github.com/abdk-consulting/abdk-libraries-solidity/blob/master/ABDKMath64x64.sol)
pub trait FixedPointInstructions<F: ScalarField, const PRECISION_BITS: u32> {
    type Gate: GateInstructions<F>;
    type RangeGate: RangeInstructions<F>;

    fn gate(&self) -> &Self::Gate;
    fn range_gate(&self) -> &Self::RangeGate;
    fn strategy(&self) -> FixedPointStrategy;

    /// Returns the absolute value of `a`. Expensive: calls `is_neg` internally.
    fn qabs(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where
        F: BigPrimeField;

    /// Returns 1 if `a` represents a negative fixed-point value, 0 otherwise.
    /// Expensive: uses a 254-bit `div_mod` to check the sign bit.
    fn is_neg(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where
        F: BigPrimeField;

    /// Returns +1 or -1 (as field elements) depending on the sign of `a`.
    fn sign(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where
        F: BigPrimeField;

    /// Conditionally negates `a` based on the boolean `is_neg` flag.
    fn cond_neg(
        &self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>, is_neg: AssignedValue<F> 
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Clip the value to ensure it's in the valid range: (-2^{2P}, 2^{2P}).
    /// Warning: assumes a < 2^{2P+1}. May fail silently for larger values.
    fn clip(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Evaluate a polynomial using Horner's method at P-bit fixed-point precision.
    /// `coef` is in Horner order: `[c_d, c_{d-1}, ..., c_1, c_0]` for `c_d*x^d + ... + c_0`.
    /// Each step does `y = y*x + c` via `qmul` (which truncates by 2^P per multiply).
    /// Accumulated truncation error: ~d ULPs. Use `polynomial_hp` for >= P-bit accuracy.
    fn polynomial<QA>(
        &self,
        ctx: &mut Context<F>,
        x: impl Into<QuantumCell<F>>,
        coef: impl IntoIterator<Item = QA>
    ) -> AssignedValue<F>
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Debug + Copy;

    /// XOR of two single-bit values. Both inputs must be 0 or 1.
    fn bit_xor(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = self.gate().add(ctx, Constant(F::zero()), a.into());
        let b = self.gate().add(ctx, Constant(F::zero()), b.into());
        self.gate().assert_bit(ctx, a);
        self.gate().assert_bit(ctx, b);
        let ab = self.gate().add(ctx, a, b);
        let one = self.gate().add(ctx, Constant(F::one()), Constant(F::zero()));
        let xor = self.gate().is_equal(ctx, ab, one);

        xor
    }

    fn qsum<Q>(&self, ctx: &mut Context<F>, a: impl IntoIterator<Item = Q>) -> AssignedValue<F>
    where
        Q: Into<QuantumCell<F>>,
    {
        self.gate().sum(ctx, a)
    }

    fn neg(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        self.gate().neg(ctx, a)
    }

    /// Fixed-point addition. Direct field addition (no overflow check).
    fn qadd(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Fixed-point subtraction. Direct field subtraction (no overflow check).
    fn qsub(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;
    
    /// Fixed-point multiplication: computes `a * b / 2^P` via `signed_div_scale`.
    fn qmul(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;
    
    /// Fixed-point division: computes `(a * 2^P) / b`.
    fn qdiv(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;
    
    /// Fixed-point inner product: computes `sum(a_i * b_i) / 2^P`.
    /// Batches all multiplications before a single `signed_div_scale`, reducing
    /// circuit cost from ~90*N to ~N+90 cells compared to N separate `qmul` calls.
    /// Warning: the accumulated sum must fit in the BN254 field (~254 bits).
    fn inner_product<QA>(
        &self,
        ctx: &mut Context<F>,
        a: impl IntoIterator<Item = QA>,
        b: impl IntoIterator<Item = QA>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Copy;

    /// Fixed-point modulo: `a mod b` where `b > 0`. Result has same sign convention as remainder.
    fn qmod(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Computes 2^a using HP polynomial evaluation. Precision: >= P bits (max 1 ULP).
    /// Splits input into integer and fractional parts: 2^a = 2^int * poly(frac).
    /// For negative inputs: 2^(-|a|) = 1 / 2^|a|.
    fn qexp2(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Computes log2(a) for a > 0 using HP polynomial evaluation. Precision: >= P bits (max 1 ULP).
    /// Normalizes input to [2, 4) at HP scale (avoiding right-shift truncation),
    /// then evaluates log2(1+t) Chebyshev polynomial where t = normalized/2 - 1.
    fn qlog2(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;
 
    /// Computes sin(a) using polynomial approximation on [0, pi].
    /// Reduces input modulo 2*pi, then uses symmetry for [pi, 2*pi).
    fn qsin(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Computes cos(a) = sin(a + pi/2).
    fn qcos(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Constrains that `pow2_exponent == 2^exponent` by checking the bit decomposition
    /// has exactly one set bit at position `exponent`.
    fn check_power_of_two(&self, ctx: &mut Context<F>, pow2_exponent: AssignedValue<F>, exponent: AssignedValue<F>)
    where
        F: BigPrimeField;

    /// Computes tan(a) = sin(a) / cos(a).
    fn qtan(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let sin_a = self.qsin(ctx, a);
        let cos_a = self.qcos(ctx, a);
        let y = self.qdiv(ctx, sin_a, cos_a);

        y
    }

    /// Computes e^a = 2^(a / ln(2)).
    fn qexp(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;


    /// Computes sinh(a) = (e^a - e^{-a}) / 2.
    fn qsinh(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;
    
    /// Computes cosh(a) = (e^a + e^{-a}) / 2.
    fn qcosh(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;
    
    /// Computes tanh(a) = sinh(a) / cosh(a).
    fn qtanh(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let sinh = self.qsinh(ctx, a);
        let cosh = self.qcosh(ctx, a);
        let y = self.qdiv(ctx, sinh, cosh);

        y
    }

    /// Returns max(a, b).
    fn qmax(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Returns min(a, b).
    fn qmin(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Computes ln(a) = log2(a) / log2(e).
    fn qlog(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Computes x^exponent = exp(exponent * ln(x)).
    fn qpow(
        &self,
        ctx: &mut Context<F>,
        x: impl Into<QuantumCell<F>>,
        exponent: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        // x^a = exp(a * log(x))
        let logx = self.qlog(ctx, x);
        let alogx = self.qmul(ctx, exponent, logx);
        let y = self.qexp(ctx, alogx);

        y
    }

    /// Computes sqrt(x) = x^0.5 via qpow.
    fn qsqrt(
        &self,
        ctx: &mut Context<F>,
        x: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField;

    /// Divide `a` by the quantization scale `2^P`, returning `(quotient, remainder)`.
    /// Constrains: `a = 2^P * q + r` with `0 <= r < 2^P` and `|q| < 2^{3P}`.
    /// Uses an offset trick to avoid expensive `qabs`: translates `q` by `2^{3P}-1`
    /// and does a single range check, reducing cells from ~250 to ~90.
    fn signed_div_scale(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> (AssignedValue<F>, AssignedValue<F>);

    /// Generalized signed division by `2^{pow_bits}`, returning `(quotient, remainder)`.
    /// Constrains: `a = 2^{pow_bits} * q + r` with `0 <= r < 2^{pow_bits}` and `|q| < 2^{max_quotient_bits}`.
    /// Used by HP polynomial evaluation for intermediate divisions at (P+K)-bit scale.
    fn signed_div_by_pow2(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        pow_bits: u32,
        max_quotient_bits: u32,
    ) -> (AssignedValue<F>, AssignedValue<F>)
    where
        F: BigPrimeField;

    /// Higher-Precision polynomial evaluation using Horner's method.
    /// Scales `x` up by `2^K` internally, evaluates at (P+K)-bit precision, then truncates back.
    /// Coefficients must be pre-quantized at `2^{P+K}` scale (use `generate_*_poly_hp`).
    /// Achieves <= 1 ULP error at P-bit scale (vs ~d ULPs for standard `polynomial`).
    fn polynomial_hp<QA>(
        &self,
        ctx: &mut Context<F>,
        x: impl Into<QuantumCell<F>>,
        coef_hp: impl IntoIterator<Item = QA>,
    ) -> AssignedValue<F>
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Debug + Copy;
}

impl<F: BigPrimeField, const PRECISION_BITS: u32> FixedPointInstructions<F, PRECISION_BITS> for FixedPointChip<F, PRECISION_BITS> {
    type Gate = GateChip<F>;
    type RangeGate = RangeChip<F>;

    fn range_gate(&self) -> &Self::RangeGate {
        &self.gate
    }

    fn gate(&self) -> &Self::Gate {
        &self.gate.gate()
    }

    fn strategy(&self) -> FixedPointStrategy {
        self.strategy
    }

    /// Fixed-point addition. Direct field addition (no overflow check).
    fn qadd(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        self.gate().add(ctx, a, b)
    }

    /// Fixed-point subtraction. Direct field subtraction (no overflow check).
    fn qsub(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        self.gate().sub(ctx, a, b)
    }

    fn qabs(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let a_reverse = self.gate().neg(ctx, a);
        let is_neg = self.is_neg(ctx, a);
        let a_abs = self.gate().select(ctx, a_reverse, a, is_neg);

        a_abs
    }

    fn is_neg(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let a_num_bits = 254;
        let (a_shift, _) = self.range_gate().div_mod(
            ctx, a, BigUint::from(2u32).pow((PRECISION_BITS * 2 + 1)as u32), a_num_bits);
        let is_pos = self.gate().is_zero(ctx, a_shift);
        let is_neg = self.gate().not(ctx, is_pos);

        is_neg
    }

    fn cond_neg(
        &self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>, is_neg: AssignedValue<F> 
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let neg_a = self.gate().neg(ctx, a);
        // self.gate().assert_bit(ctx, is_neg_assigned);
        let res = self.gate().select(ctx, neg_a, a, is_neg);

        res
    }

    fn sign(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let pos_one = Constant(F::one());
        // (-1) % m where m = 2^254
        let neg_one = self.gate().neg(ctx, pos_one);
        let is_neg = self.is_neg(ctx, a);
        let res = self.gate().select(ctx, neg_one, pos_one, is_neg);

        res
    }

    fn clip(&self, ctx: &mut Context<F>, a: impl Into<QuantumCell<F>>) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let sign = self.is_neg(ctx, a);
        let a_abs = self.qabs(ctx, a);
        let a_num_bits = 254;
        let m = self.max_value.clone();
        // clipped = a % m
        // TODO (Wentao XIAO) should we just throw panic when overflow?
        let (_, unsigned_cliped) = self.range_gate().div_mod(ctx, a_abs, m, a_num_bits);
        let clipped = self.cond_neg(ctx, unsigned_cliped, sign);

        clipped
    }

    /// Fixed-point multiplication: computes `a * b / 2^P` via `signed_div_scale`.
    fn qmul(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let b = b.into();

        let ab = self.gate().mul(ctx, a, b);
        let (res, _) = self.signed_div_scale(ctx, ab);

        res
    }

    /// Fixed-point modulo: `a mod b` where `b > 0`. Result has same sign convention as remainder.
    fn qmod(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        // b must be positive
        let a = a.into();
        let b = b.into();
        let a_sign = self.is_neg(ctx, a);
        let b_sign = self.is_neg(ctx, b);
        self.gate().assert_is_const(ctx, &b_sign, &F::zero());
        let a_abs = self.qabs(ctx, a);
        let a_num_bits = PRECISION_BITS as usize * 4;
        let b_num_bits = PRECISION_BITS as usize * 2;
        let (_, res_abs) = self.range_gate().div_mod_var(
            ctx, a_abs, b, a_num_bits, b_num_bits
        );
        let res_abs_comp = self.gate().sub(ctx, b, res_abs);
        let res = self.gate().select(ctx, res_abs_comp, res_abs, a_sign);

        res
    }

    /// Fixed-point division: computes `(a * 2^P) / b`.
    fn qdiv(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let b = b.into();
        let a_sign = self.is_neg(ctx, a);
        let b_sign = self.is_neg(ctx, b);
        let a_abs = self.qabs(ctx, a);
        let b_abs = self.qabs(ctx, b);
        // Because a_rescale \in [0, 2^{4p}) and b \in [0, 2^p)
        let a_num_bits = PRECISION_BITS as usize * 4;
        let b_num_bits = PRECISION_BITS as usize * 2;
        let a_rescale = self.gate().mul(ctx, a_abs, Constant(self.quantization_scale));
        let (res_abs, _) = self.range_gate().div_mod_var(
            ctx, a_rescale, b_abs, a_num_bits, b_num_bits
        );
        let ab_sign = self.bit_xor(ctx, a_sign, b_sign);
        let res = self.cond_neg(ctx, res_abs, ab_sign);

        res
    }

    fn polynomial<QA>(
        &self,
        ctx: &mut Context<F>,
        x: impl Into<QuantumCell<F>>,
        coef: impl IntoIterator<Item = QA>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Debug + Copy
    {
        let x = x.into();
        let mut intermediates = vec![Constant(F::zero())];
        let coef_iter: Vec<QA> = coef.into_iter().collect();
        let last_idx_coef = coef_iter.len() - 1;
        let mut result: AssignedValue<F> = self.qadd(ctx, x, Constant(F::zero()));
        for (idx, c) in coef_iter.into_iter().enumerate() {
            let last_y = *intermediates.get(intermediates.len() - 1).unwrap();
            let y_add = self.qadd(ctx, last_y, c);
            intermediates.push(Existing(y_add));
            if idx < last_idx_coef {
                let y = self.qmul(ctx, x, Existing(y_add));
                intermediates.push(Existing(y));
            } else {
                result = y_add;
            }
        }

        result
    }

    /// Constrains that `pow2_exponent == 2^exponent` by checking the bit decomposition
    /// has exactly one set bit at position `exponent`.
    fn check_power_of_two(&self, ctx: &mut Context<F>, pow2_exponent: AssignedValue<F>, exponent: AssignedValue<F>)
    where
        F: BigPrimeField,
    {
        let range_bits = PRECISION_BITS as usize * 2;
        let bits = self.gate().num_to_bits(ctx, pow2_exponent, range_bits);
        let sum_of_bits = self.gate().sum(ctx, bits.clone());
        let sum_of_bits_m1 = self.gate().sub(ctx, sum_of_bits, Constant(F::one()));
        let is_zero = self.gate().is_zero(ctx, sum_of_bits_m1);
        // ensure the bits of pow2_exponent has only one of bit one.
        self.gate().assert_is_const(ctx, &is_zero, &F::one());
        let bit = self.gate().select_from_idx(
            ctx, 
            bits.into_iter().map(|x| Existing(x)), 
            exponent
        );
        let bit_m1 = self.gate().sub(ctx, bit, Constant(F::one()));
        let is_zero_bit_m1 = self.gate().is_zero(ctx, bit_m1);
        // ensures bits[expnent] is exact bit one
        self.gate().assert_is_const(ctx, &is_zero_bit_m1, &F::one());
    }

    fn qexp2(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let a_abs = self.qabs(ctx, a);
        let num_bits = PRECISION_BITS as usize * 2;
        let shift = 2u128.pow(PRECISION_BITS);
        let (int_part, frac_part) = self.range_gate().div_mod(
            ctx, Existing(a_abs), shift, num_bits);
        // int_part must be small as large number leads to overflow.
        let pow_of_two: Vec<QuantumCell<F>> = self.pow_of_two.iter().map(|x| Constant(*x)).collect();
        let int_part_pow2 = self.gate().select_from_idx(
            ctx, pow_of_two, int_part);
        let coef = self.generate_exp2_poly_hp();
        let y_frac = self.polynomial_hp(ctx, frac_part, coef);
        let res_pos = self.gate().mul(ctx, Existing(int_part_pow2), Existing(y_frac));

        let one = Constant(F::from_u128(shift));
        let res_neg = self.qdiv(ctx, one, res_pos);
        let is_neg = self.is_neg(ctx, a);
        let res = self.gate().select(ctx, res_neg, res_pos, is_neg);

        res
    }

    fn qlog2(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where
        F: BigPrimeField
    {
        let a = a.into();
        let a_assigned = self.gate().add(ctx, a, Constant(F::zero()));
        let is_neg = self.is_neg(ctx, a);
        let is_zero = self.gate().is_zero(ctx, a_assigned);
        let is_invalid = self.gate().or(ctx, is_neg, is_zero);
        self.gate().assert_is_const(ctx, &is_invalid, &F::zero());
        let num_bits = (PRECISION_BITS * 2) as usize;
        let num_digits = a_assigned.value()
            .to_repr()
            .as_ref()
            .iter()
            .flat_map(|byte| (0..8u32).map(|i| (*byte as u64 >> i) & 1))
            .enumerate()
            .fold(1u64, |acc, (idx, val)| {
                if val == 1u64 {
                    idx as u64
                } else {
                    acc
                }
            });
        let pow1 = self.gate().pow_of_two()[num_digits as usize];
        let pow1_witness = self.gate().add(ctx, Witness(pow1), Constant(F::zero()));
        let exp1 = self.gate().add(ctx, Witness(F::from(num_digits)), Constant(F::zero()));
        self.check_power_of_two(ctx, pow1_witness, exp1);
        let pow2_witness = self.gate().mul(ctx, pow1_witness, Constant(F::from(2)));
        let exp2 = self.gate().add(ctx, exp1, Constant(F::one()));
        self.check_power_of_two(ctx, pow2_witness, exp2);
        // pow1 <= a < pow2, pow1 = 2^n, pow2 = 2^{n+1}
        let a_lt_pow2 = self.range_gate().is_less_than(ctx, a, pow2_witness, num_bits);
        let a_gt_pow1 = self.range_gate().is_less_than(ctx, pow1_witness, a, num_bits);
        let a_eq_pow1 = self.gate().is_equal(ctx, a, pow1_witness);
        let a_ge_pow1 = self.gate().or(ctx, a_eq_pow1, a_gt_pow1);
        let a_bound = self.gate().and(ctx, a_lt_pow2, a_ge_pow1);
        self.gate().assert_is_const(ctx, &a_bound, &F::one());

        // Scale up to HP to avoid truncation error during normalization right-shift
        let extra_bits = Self::EXTRA_PRECISION_BITS;
        let hp_bits = PRECISION_BITS + extra_bits;
        let extra_scale = F::from_u128(2u128.pow(extra_bits));
        let hp_quantization_scale = F::from_u128(2u128.pow(hp_bits));
        let a_hp = self.gate().mul(ctx, a, Constant(extra_scale));
        let hp_num_bits = (hp_bits * 2 + 2) as usize;

        // shift a_hp to ensure a_hp_norm in [2^{hp_bits+1}, 2^{hp_bits+2})
        // shift = (P+2) - exp2 (same as original; a_hp already has K extra bits)
        let shift = self.gate().sub(
            ctx, Constant(F::from(PRECISION_BITS as u64 + 2)), exp2);
        let is_shift_neg = self.is_neg(ctx, shift);
        let shift_abs = self.qabs(ctx, shift);
        let shift_pow2 = self.gate().pow_of_two()[shift_abs.value().get_lower_32() as usize];
        let shift_pow2_witness = self.gate().add(ctx, Witness(shift_pow2), Constant(F::zero()));
        self.check_power_of_two(ctx, shift_pow2_witness, shift_abs);
        let a_hp_ls = self.gate().mul(ctx, a_hp, shift_pow2_witness);
        let (a_hp_rs, _) = self.range_gate().div_mod_var(
            ctx, a_hp, shift_pow2_witness, hp_num_bits, hp_bits as usize + 1);
        let a_hp_norm = self.gate().select(ctx, a_hp_rs, a_hp_ls, is_shift_neg);

        // t_hp = a_hp_norm/2 - 2^{P+K}, in HP fixed-point [0, 2^{P+K})
        let (a_hp_norm_half, _) = self.range_gate().div_mod(
            ctx, a_hp_norm, 2u128, hp_num_bits);
        let t_hp = self.gate().sub(ctx, a_hp_norm_half, Constant(hp_quantization_scale));

        // Evaluate polynomial directly at HP scale using Horner
        let coef = self.generate_log_poly_hp();
        let coef_iter: Vec<QuantumCell<F>> = coef.into_iter().collect();
        let last_idx = coef_iter.len() - 1;
        let mut y: AssignedValue<F> = ctx.load_zero();

        for (idx, c) in coef_iter.into_iter().enumerate() {
            let y_add = self.gate().add(ctx, y, c);
            if idx < last_idx {
                let prod = self.gate().mul(ctx, y_add, Existing(t_hp));
                let (div, _) = self.signed_div_by_pow2(ctx, prod, hp_bits, hp_bits + 2);
                y = div;
            } else {
                y = y_add;
            }
        }
        let log_1_plus_t_hp = y;

        // log2(a_hp_norm_real) = 1 + poly(t) at HP scale
        let log_a_norm_hp = self.gate().add(ctx, log_1_plus_t_hp, Constant(hp_quantization_scale));

        // log2(a_real) = log2(a_hp_norm_real) - shift
        let log_shift = self.gate().neg(ctx, shift);
        let log_shift_q = self.gate().mul(ctx, log_shift, Constant(hp_quantization_scale));
        let result_hp = self.gate().add(ctx, log_a_norm_hp, log_shift_q);

        // Convert from HP scale to P-bit scale
        let (result, _) = self.signed_div_by_pow2(ctx, result_hp, extra_bits, hp_bits);

        result
    }

    fn bit_xor(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = self.gate().add(ctx, Constant(<F>::zero()), a.into());
        let b = self.gate().add(ctx, Constant(<F>::zero()), b.into());
        self.gate().assert_bit(ctx, a);
        self.gate().assert_bit(ctx, b);
        let ab = self.gate().add(ctx, a, b);
        let one = self.gate().add(ctx, Constant(<F>::one()), Constant(<F>::zero()));
        let xor = self.gate().is_equal(ctx, ab, one);

        xor
    }

    fn qsin(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let a_abs = self.qabs(ctx, a);
        let a_sign = self.is_neg(ctx, a);
        let pi_2 = Constant(self.quantization(std::f64::consts::PI * 2.0));
        // |a| % 2pi
        let a_mod = self.qmod(ctx, a_abs, pi_2);
        let pi = Constant(self.quantization(std::f64::consts::PI));
        // (|a| % 2pi) - pi
        let a_mpi = self.qsub(ctx, a_mod, pi);
        let is_neg_a_mpi = self.is_neg(ctx, a_mpi);
        let coef1 = self.generate_sin_poly();
        let sin_a_mod = self.polynomial(ctx, a_mod, coef1);
        let coef2 = self.generate_sin_poly();
        // -sin(a-pi) for pi <= a < 2pi
        let sin_a_mpi_rev = self.polynomial(ctx, a_mpi, coef2);
        let sin_a_mpi = self.neg(ctx, sin_a_mpi_rev);
        let sin_a_abs = self.gate().select(ctx, sin_a_mod, sin_a_mpi, is_neg_a_mpi);
        let sin_a = self.cond_neg(ctx, sin_a_abs, a_sign);

        sin_a
    }

    /// Computes cos(a) = sin(a + pi/2).
    fn qcos(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let half_pi = ctx.load_constant(self.quantization(std::f64::consts::FRAC_PI_2));
        let a_plus_half_pi = self.qadd(ctx, a, half_pi);
        let y = self.qsin(ctx, a_plus_half_pi);

        y
    }

    /// Fixed-point inner product: computes `sum(a_i * b_i) / 2^P`.
    /// Batches all multiplications before a single `signed_div_scale`, reducing
    /// circuit cost from ~90*N to ~N+90 cells compared to N separate `qmul` calls.
    /// Warning: the accumulated sum must fit in the BN254 field (~254 bits).
    fn inner_product<QA>(
        &self,
        ctx: &mut Context<F>,
        a: impl IntoIterator<Item = QA>,
        b: impl IntoIterator<Item = QA>
    ) -> AssignedValue<F>
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Copy
    {
        let a: Vec<QA> = a.into_iter().collect();
        let b: Vec<QA> = b.into_iter().collect();
        assert!(a.len() == b.len());
        let mut res_s = ctx.load_witness(F::zero());
        self.gate().assert_is_const(ctx, &res_s, &F::zero());
        for (ai, bi) in a.iter().zip(b.iter()).into_iter() {
            res_s = self.gate().mul_add(ctx, *ai, *bi, res_s);
        }
        let (res, _) = self.signed_div_scale(ctx, res_s);

        res
    }

    /// Computes e^a = 2^(a / ln(2)).
    fn qexp(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        // e^x == 2^(x / ln(2))
        let ln2 = ctx.load_constant(self.quantization(2.0f64.ln()));
        let x1 = self.qdiv(ctx, a, ln2);
        let y = self.qexp2(ctx, x1);

        y
    }

    /// Computes sinh(a) = (e^a - e^{-a}) / 2.
    fn qsinh(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let ea = self.qexp(ctx, a);
        let na = self.neg(ctx, a);
        let ena = self.qexp(ctx, na);
        let nume = self.qsub(ctx, ea, ena);
        let two = ctx.load_constant(self.quantization(2.0));
        let y = self.qdiv(ctx, nume, two);

        y
    }

    /// Computes cosh(a) = (e^a + e^{-a}) / 2.
    fn qcosh(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let ea = self.qexp(ctx, a);
        let na = self.neg(ctx, a);
        let ena = self.qexp(ctx, na);
        let nume = self.qadd(ctx, ea, ena);
        let two = ctx.load_constant(self.quantization(2.0));
        let y = self.qdiv(ctx, nume, two);

        y
    }

    /// Returns max(a, b).
    fn qmax(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let b = b.into();
        let amb = self.qsub(ctx, a, b);
        let sign_amb = self.is_neg(ctx, amb);
        let y = self.gate().select(ctx, b, a, sign_amb);

        y
    }

    /// Returns min(a, b).
    fn qmin(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        b: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let a = a.into();
        let b = b.into();
        let amb = self.qsub(ctx, a, b);
        let sign_amb = self.is_neg(ctx, amb);
        let y = self.gate().select(ctx, a, b, sign_amb);

        y
    }

    /// Computes ln(a) = log2(a) / log2(e).
    fn qlog(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        // log(x) = log2(x) / log2(e)
        let log2e = ctx.load_constant(self.quantization(std::f64::consts::LOG2_E));
        let log2a = self.qlog2(ctx, a);
        let y = self.qdiv(ctx, log2a, log2e);

        y
    }

    /// Computes sqrt(x) = x^0.5 via qpow.
    fn qsqrt(
        &self,
        ctx: &mut Context<F>,
        x: impl Into<QuantumCell<F>>
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        let half = ctx.load_constant(self.quantization(0.5));
        self.qpow(ctx, x, half)
    }

    fn signed_div_scale(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>
    ) -> (AssignedValue<F>, AssignedValue<F>)
    {
        // a = b * q + r, r in [0, b), q in [-2^n, 2^n]
        let a = a.into();
        // b = 2^p
        let b = fe_to_biguint(&self.quantization_scale);
        // 2^254-2^252 > 2^252
        let a_is_neg = fe_to_biguint(a.value()) > BigUint::from(2u32).pow(252u32);
        let (q, r) = if a_is_neg {
            let a_abs = fe_to_biguint(&(self.bn254_max - a.value() + F::one()));
            let q = fe_to_biguint(&self.bn254_max) - a_abs.div_ceil(&b) + BigUint::from(1u32);
            let r = fe_to_biguint::<F>(a.value()) - fe_to_biguint::<F>(
                &(biguint_to_fe::<F>(&b.clone()) * biguint_to_fe::<F>(&q.clone())));
            // assert!(*a.value() == biguint_to_fe::<F>(&b) * biguint_to_fe::<F>(&q) + biguint_to_fe::<F>(&r));
            (q, r)
        } else {
            fe_to_biguint(a.value()).div_mod_floor(&b)
        };
        ctx.assign_region(
            [Witness(biguint_to_fe(&r)), Constant(biguint_to_fe(&b)), Witness(biguint_to_fe(&q)), a],
            [0]
        );
        let rem = ctx.get(-4);
        let div = ctx.get(-2);

        self.range_gate().check_big_less_than_safe(ctx, rem, b);
        // a < 2^{4p}, b = 2^p, so |q| < 2^{3p}
        // Use offset trick to avoid expensive qabs: translate q by 2^{3p}-1
        // so that [-(2^{3p}-1), 2^{3p}-1] maps to [0, 2^{3p+1}-2]
        let abs_bound = BigUint::from(2u32).pow(PRECISION_BITS * 3 as u32);
        let abs_bound_minus1 = abs_bound.clone() - BigUint::from(1u32);
        let new_bound = abs_bound * BigUint::from(2u32) - BigUint::from(1u32);
        let div_plus_offset =
            self.gate().add(ctx, div, Constant(biguint_to_fe(&abs_bound_minus1)));
        self.range_gate().check_big_less_than_safe(ctx, div_plus_offset, new_bound);

        (div, rem)
    }

    fn signed_div_by_pow2(
        &self,
        ctx: &mut Context<F>,
        a: impl Into<QuantumCell<F>>,
        pow_bits: u32,
        max_quotient_bits: u32,
    ) -> (AssignedValue<F>, AssignedValue<F>)
    where
        F: BigPrimeField
    {
        let a = a.into();
        let b = BigUint::from(2u32).pow(pow_bits);
        let a_is_neg = fe_to_biguint(a.value()) > BigUint::from(2u32).pow(252u32);
        let (q, r) = if a_is_neg {
            let a_abs = fe_to_biguint(&(self.bn254_max - a.value() + F::one()));
            let q = fe_to_biguint(&self.bn254_max) - a_abs.div_ceil(&b) + BigUint::from(1u32);
            let r = fe_to_biguint::<F>(a.value()) - fe_to_biguint::<F>(
                &(biguint_to_fe::<F>(&b.clone()) * biguint_to_fe::<F>(&q.clone())));
            (q, r)
        } else {
            fe_to_biguint(a.value()).div_mod_floor(&b)
        };
        ctx.assign_region(
            [Witness(biguint_to_fe(&r)), Constant(biguint_to_fe(&b)), Witness(biguint_to_fe(&q)), a],
            [0]
        );
        let rem = ctx.get(-4);
        let div = ctx.get(-2);
        self.range_gate().check_big_less_than_safe(ctx, rem, b);
        let bound = BigUint::from(2u32).pow(max_quotient_bits);
        let div_abs = self.qabs(ctx, div);
        self.range_gate().check_big_less_than_safe(ctx, div_abs, bound);
        (div, rem)
    }

    fn polynomial_hp<QA>(
        &self,
        ctx: &mut Context<F>,
        x: impl Into<QuantumCell<F>>,
        coef_hp: impl IntoIterator<Item = QA>,
    ) -> AssignedValue<F>
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Debug + Copy
    {
        let x = x.into();
        let extra_bits = Self::EXTRA_PRECISION_BITS;
        let hp_bits = PRECISION_BITS + extra_bits;
        let extra_scale = F::from_u128(2u128.pow(extra_bits));
        let x_hp = self.gate().mul(ctx, x, Constant(extra_scale));

        let coef_iter: Vec<QA> = coef_hp.into_iter().collect();
        let last_idx = coef_iter.len() - 1;
        let mut y: AssignedValue<F> = ctx.load_zero();

        for (idx, c) in coef_iter.into_iter().enumerate() {
            let y_add = self.gate().add(ctx, y, c);
            if idx < last_idx {
                let prod = self.gate().mul(ctx, y_add, Existing(x_hp));
                let (div, _) = self.signed_div_by_pow2(ctx, prod, hp_bits, hp_bits + 2);
                y = div;
            } else {
                let (result, _) = self.signed_div_by_pow2(ctx, y_add, extra_bits, hp_bits);
                return result;
            }
        }
        unreachable!()
    }
}
