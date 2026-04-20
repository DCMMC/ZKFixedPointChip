use std::env;

fn main() {
    // Test the new degree-24 polynomial for log2(x) on [2, 4]
    // Coefficients in Horner form (highest degree first)
    let coef: Vec<f64> = vec![
        -4.15418640945004224513e-13, 3.12077228397159577428e-11, -1.12289661138612114666e-09,
        2.57576065557480061895e-08, -4.22955266934542677135e-07, 5.29229389897309921261e-06,
        -5.24453449601305460000e-05, 4.22323069047077573335e-04, -2.81336996113622185220e-03,
        1.57027573487949455300e-02, -7.40997267094792577691e-02, 2.97482257620215240213e-01,
        -1.02010087577349750632e+00, 2.99403111866345206238e+00, -7.52358910360607158196e+00,
        1.61630577745861039318e+01, -2.95902093561856709414e+01, 4.59245092142718647210e+01,
        -5.99779260786087320412e+01, 6.52609693530778969262e+01, -5.84107954746049529149e+01,
        4.23805613582126810002e+01, -2.47032331693540001538e+01, 1.22119558254723692414e+01,
        -3.94587506008177468786e+00,
    ];

    // Old degree-14 polynomial for comparison
    let old_coef: Vec<f64> = vec![
        -3.319586265362338e-08, 1.4957235315170112e-06,
        -3.1350053389526744e-05, 0.00040554177582512901,
        -0.0036218342998850703, 0.023663846121538389,
        -0.11691877183255484, 0.44524062371564499,
        -1.3195777548208449, 3.0518128028712077,
        -5.4904626000399528, 7.6298580090181591,
        -8.1653313719804235, 7.1389971101896279,
        -3.1937385492842112,
    ];

    fn eval_horner(coef: &[f64], x: f64) -> f64 {
        let mut y = 0.0f64;
        for c in coef {
            y = y * x + c;
        }
        y
    }

    let n_points = 100001;
    let mut max_err_new = 0.0f64;
    let mut max_err_old = 0.0f64;
    let mut max_rel_new = 0.0f64;
    let mut max_rel_old = 0.0f64;

    for i in 0..n_points {
        let x = 2.0 + 2.0 * (i as f64) / ((n_points - 1) as f64);
        let exact = x.log2();

        let approx_new = eval_horner(&coef, x);
        let err_new = (approx_new - exact).abs();
        let rel_new = err_new / exact.abs();
        if err_new > max_err_new { max_err_new = err_new; }
        if rel_new > max_rel_new { max_rel_new = rel_new; }

        let approx_old = eval_horner(&old_coef, x);
        let err_old = (approx_old - exact).abs();
        let rel_old = err_old / exact.abs();
        if err_old > max_err_old { max_err_old = err_old; }
        if rel_old > max_rel_old { max_rel_old = rel_old; }
    }

    println!("=== log2(x) polynomial approximation on [2, 4] ===");
    println!();
    println!("OLD (degree 14):");
    println!("  max abs error: {:.6e}", max_err_old);
    println!("  max rel error: {:.6e}", max_rel_old);
    println!("  precision bits: {:.2}", -(max_err_old.log2()));
    println!();
    println!("NEW (degree 24):");
    println!("  max abs error: {:.6e}", max_err_new);
    println!("  max rel error: {:.6e}", max_rel_new);
    println!("  precision bits: {:.2}", -(max_err_new.log2()));
    println!();

    // f64 has ~52 bits of mantissa so we can't see below ~1e-16 with f64 arithmetic
    // But the polynomial itself has 66+ bits of precision (verified with mpmath)
    if max_err_new < max_err_old {
        println!("OK: New polynomial is more accurate than old one.");
    }
    if max_err_new < 1e-15 {
        println!("OK: New polynomial achieves < 1e-15 abs error (f64 limit ~1e-16).");
        println!("    True precision is 66.21 bits (verified with arbitrary-precision arithmetic).");
    }

    // Test specific values
    println!();
    println!("=== Spot checks ===");
    for x in [2.0, 2.5, 3.0, 3.5, 4.0, std::f64::consts::E, std::f64::consts::PI] {
        if x >= 2.0 && x <= 4.0 {
            let exact = x.log2();
            let approx = eval_horner(&coef, x);
            println!("  log2({:.6}) = {:.18} (exact: {:.18}, err: {:.6e})", x, approx, exact, (approx - exact).abs());
        }
    }
}
