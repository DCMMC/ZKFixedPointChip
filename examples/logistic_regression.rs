use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::AssignedValue;
use halo2_base::Context;
use halo2_scaffold::gadget::logistic_regression::LogisticRegressionChip;
use halo2_scaffold::scaffold::{gen_key, prove_private};
#[allow(unused_imports)]
use halo2_scaffold::scaffold::{mock, prove};
use log::warn;
use std::cmp::min;
use std::env::{var, set_var};
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array, Axis};

pub fn train_native(
    train_x: Vec<Vec<f64>>, train_y: Vec<f64>, lr: f64, epoch: i32, batch_size: usize
) {
    let dim = train_x[0].len();
    let mut w = vec![0.; dim];
    let mut b = 0.;
    let n_batch = (train_x.len() as f64 / batch_size as f64).ceil() as i64;
    for idx_epoch in 0..epoch {
        println!("Epoch {:?}", idx_epoch + 1);
        for idx_batch in 0..n_batch {
            if (idx_batch as usize + 1) * batch_size > train_x.len() {
                continue;
            }
            let batch_x = (&train_x[idx_batch as usize * batch_size..(idx_batch as usize + 1) * batch_size]).to_vec();
            let batch_y = (&train_y[idx_batch as usize * batch_size..min(train_y.len(), (idx_batch as usize + 1) * batch_size)]).to_vec();
            let n_sample = batch_x.len();
            let batch_lr = lr / n_sample as f64;

            let y_pred: Vec<f64> = batch_x.iter().map(|xi| {
                let mut yi = b;
                for j in 0..xi.len() {
                    yi += xi[j] * w[j];
                }
                yi = 1.0 / (1.0 + (-yi).exp());

                yi
            }).collect();
            let diff_y: Vec<f64> = y_pred.iter().zip(batch_y.clone()).map(|(yi, ti)| yi - ti).collect();
            let loss: f64 = y_pred.iter().zip(batch_y).map(|(yi, ti)| {
                if ti == 1.0 {
                    -yi.ln()
                } else {
                    -(1.0 - yi).ln()
                }
            }).sum::<f64>() / y_pred.len() as f64;
            println!("loss: {:?}", loss);
            b = b - batch_lr * diff_y.iter().sum::<f64>();
            for j in 0..w.len() {
                w[j] = w[j] - batch_lr * diff_y.iter().zip(batch_x.iter()).map(|(diff_yi, batch_xi)| diff_yi * batch_xi[j]).sum::<f64>();
            }
            println!("w: {:?}, b: {:?}", w, b);
        }
    }
}

pub fn train<F: ScalarField>(
    ctx: &mut Context<F>,
    input: (Vec<F>, F, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, f64),
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let lookup_bits =
        var("LOOKUP_BITS").unwrap_or_else(|_| panic!("LOOKUP_BITS not set")).parse().unwrap();
    let chip = LogisticRegressionChip::<F>::new(lookup_bits);

    let (w, b, train_x, train_y, learning_rate) = input;
    let mut w = w.iter().map(
        |wi| ctx.load_witness(*wi)).collect();
    let mut b = ctx.load_witness(b);
    let mut train_x_witness: Vec<Vec<Vec<AssignedValue<F>>>> = vec![];
    for x_batch in train_x {
        let mut batch: Vec<Vec<AssignedValue<F>>> = vec![];
        for xi in x_batch {
            batch.push(xi.iter().map(|xij| ctx.load_witness(chip.chip.quantization(*xij))).collect::<Vec<AssignedValue<F>>>());
        }
        train_x_witness.push(batch);
    }
    
    let mut train_y_witness: Vec<Vec<AssignedValue<F>>> = vec![];
    for y_batch in train_y {
        train_y_witness.push(y_batch.iter().map(|yi| ctx.load_witness(chip.chip.quantization(*yi))).collect::<Vec<AssignedValue<F>>>());
    }

    (w, b) = chip.train_multi_batch(ctx, w, b, train_x_witness, train_y_witness, learning_rate);
    for wi in w {
        make_public.push(wi);
    }
    make_public.push(b);
    let param: Vec<f64> = make_public.iter().map(|x| chip.chip.dequantization(*x.value())).collect();
    println!("params: {:?}", param);
}

pub fn inference<F: ScalarField>(
    ctx: &mut Context<F>,
    x: Vec<f64>,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    // `Context` can roughly be thought of as a single-threaded execution trace of a program we want to ZK prove. We do some post-processing on `Context` to optimally divide the execution trace into multiple columns in a PLONKish arithmetization
    // More advanced usage with multi-threaded witness generation is possible, but we do not explain it here

    // lookup bits must agree with the size of the lookup table, which is specified by an environmental variable
    let lookup_bits =
        var("LOOKUP_BITS").unwrap_or_else(|_| panic!("LOOKUP_BITS not set")).parse().unwrap();
    let chip = LogisticRegressionChip::<F>::new(lookup_bits);

    let x_deq: Vec<F> = x.iter().map(|xi| {
        chip.chip.quantization(*xi)
    }).collect();
    let mut x_witness = vec![];
    for idx in 0..x_deq.len() {
        let xi = ctx.load_witness(x_deq[idx]);
        x_witness.push(xi);
        make_public.push(xi);
    }

    let (train, _) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { 1 } else { 0 })
        .split_with_ratio(0.9);
    let model = LogisticRegression::default()
        .max_iterations(150)
        .fit(&train)
        .unwrap();
    println!("intercept:  {}", model.intercept());
    println!("parameters: {}", model.params());

    let sample_x = Array::from_vec(x);
    let ypred: f64 = model.predict_probabilities(&sample_x.insert_axis(Axis(0)))[0] as f64;

    let mut w = vec![];
    for wi in model.params().iter() {
        w.push(ctx.load_witness(chip.chip.quantization(*wi)));
    }
    let b = ctx.load_witness(chip.chip.quantization(model.intercept()));
    let y_zk_raw = chip.inference(ctx, w, x_witness, b);
    let y_zk = chip.chip.dequantization(*y_zk_raw.value());
    println!(
        "###### zk-linear-regression(x) = {}, native-linear-regression(x) = {:.6}, error = {:.6} ({:.6}%)",
        y_zk, ypred,
        (y_zk - ypred).abs(), (y_zk - ypred).abs() / ypred.abs() * 100.0
    );
    make_public.push(y_zk_raw);
}

fn main() {
    set_var("RUST_LOG", "warn");
    env_logger::init();
    // genrally lookup_bits is degree - 1
    set_var("LOOKUP_BITS", 18.to_string());
    set_var("DEGREE", 19.to_string());

    // let (train, valid) = linfa_datasets::winequality()
    //     .map_targets(|x| if *x > 6 { 1 } else { 0 })
    //     .split_with_ratio(0.9);
    // let model = LogisticRegression::default()
    //     .max_iterations(150)
    //     .fit(&train)
    //     .unwrap();
    // let pred = model.predict(&valid);
    // let cm = pred.confusion_matrix(&valid).unwrap();
    // println!("{:?}", cm);
    // println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
    // println!("intercept:  {}", model.intercept());
    // println!("parameters: {}", model.params());

    let x0 = vec![
        -0.00188201652779104, -0.044641636506989, -0.0514740612388061, -0.0263278347173518,
        -0.00844872411121698, -0.019163339748222, 0.0744115640787594, -0.0394933828740919,
        -0.0683297436244215, -0.09220404962683, 0.23];
    let x1 = vec![
        -0.123134, -0.129089, -0.0514740612388061, -0.21,
        -0.4353, -0.019163339748222, 0.4242, -0.44,
        -0.2341, -0.231, 0.32];
    prove(
        inference,
        x0.clone(),
        x1.clone()
    );

    let dataset = linfa_datasets::winequality()
    .map_targets(|x| if *x > 6 { 1 } else { 0 });
    let mut train_x: Vec<Vec<f64>> = vec![];
    let mut train_y: Vec<f64> = vec![];
    for (sample_x, sample_y) in dataset.sample_iter() {
        train_x.push(sample_x.iter().map(|xi| *xi).collect::<Vec<f64>>());
        train_y.push(*sample_y.iter().peekable().next().unwrap() as f64);
    }
    let dim = train_x[0].len();
    let mut w = vec![Fr::from(0); dim];
    let mut b = Fr::from(0);
    let epoch = 5;
    let learning_rate = 0.001;
    let batch_size: usize = 64;
    let learning_rate_batch = learning_rate / batch_size as f64;
    let multi_batch = 12;

    train_native(train_x.clone(), train_y.clone(), learning_rate, epoch, batch_size);

    let n_batch = (train_x.len() as f64 / batch_size as f64).ceil() as i64;
    println!("n_batch: {}", n_batch);
    let dummy_inputs = (w.clone(), b.clone(), vec![vec![vec![0.; dim]; batch_size as usize]; multi_batch], vec![vec![0.; batch_size as usize]; multi_batch], learning_rate_batch);
    let (pk, break_points) = gen_key(train, dummy_inputs);
    for idx_epoch in 0..epoch {
        warn!("Epoch {:?}", idx_epoch + 1);
        let mut buffer_x = vec![];
        let mut buffer_y = vec![];
        for idx_batch in 0..n_batch {
            if (idx_batch as usize + 1) * batch_size > train_x.len() {
                continue;
            }
            let batch_x = (&train_x[idx_batch as usize * batch_size..(idx_batch as usize + 1) * batch_size]).to_vec();
            let batch_y = (&train_y[idx_batch as usize * batch_size..min(train_y.len(), (idx_batch as usize + 1) * batch_size)]).to_vec();
            buffer_x.push(batch_x);
            buffer_y.push(batch_y);
            if buffer_x.len() == multi_batch {
                let private_inputs: (Vec<Fr>, Fr, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, f64) = (w, b, buffer_x, buffer_y, learning_rate_batch);
                let out = prove_private(train, private_inputs, &pk, break_points.clone());
                // out = mock(train, private_inputs);
                w = (&out[..dim]).iter().map(|wi| (*wi).clone()).collect();
                b = out[dim];
                buffer_x = vec![];
                buffer_y = vec![];
            }
        }
    }
    println!("w: {:?}, b: {:?}", w, b);

    // mock(train, (w, b, train_x, train_y));
    // prove(train, x0.clone(), x1.clone());
}
