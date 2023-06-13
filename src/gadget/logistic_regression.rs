use halo2_base::{
    utils::BigPrimeField,
    QuantumCell, Context, AssignedValue
};
use itertools::Itertools;
use log::warn;
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use std::convert::From;
use halo2_base::QuantumCell::{Constant, Existing};

#[derive(Clone, Debug)]
pub struct LogisticRegressionChip<F: BigPrimeField> {
    pub chip: FixedPointChip<F, 63>,
    pub lookup_bits: usize,
}

impl<F: BigPrimeField> LogisticRegressionChip<F> {
    pub fn new(lookup_bits: usize) -> Self {
        let chip = FixedPointChip::<F, 63>::default(lookup_bits);

        Self { chip, lookup_bits }
    }

    pub fn inference<QA>(
        &self,
        ctx: &mut Context<F>,
        w: impl IntoIterator<Item = QA>,
        x: impl IntoIterator<Item = QA>,
        b: QA
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField, QA: Into<QuantumCell<F>> + Copy
    {
        let wx = self.chip.inner_product(ctx, w, x);
        let logit = self.chip.qadd(ctx, wx, b);
        let neg_logit = self.chip.neg(ctx, logit);
        let exp_logit = self.chip.qexp(ctx, neg_logit);
        let one = Constant(self.chip.quantization(1.0));
        let exp_logit_p1 = self.chip.qadd(ctx, exp_logit, one);
        let y = self.chip.qdiv(ctx, one, exp_logit_p1);

        y
    }

    pub fn train_multi_batch<QA>(
        &self,
        ctx: &mut Context<F>,
        w: impl IntoIterator<Item = AssignedValue<F>>,
        b: AssignedValue<F>,
        x: impl IntoIterator<Item = impl IntoIterator<Item = impl IntoIterator<Item = QA>>>,
        y_truth: impl IntoIterator<Item = impl IntoIterator<Item = QA>>,
        learning_rate_batch: f64
    ) -> (Vec<AssignedValue<F>>, AssignedValue<F>)
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + From<AssignedValue<F>> + Copy
    {
        let x_multi_batch = x.into_iter();
        let y_truth_multi_batch = y_truth.into_iter();
        let mut w = w.into_iter().collect_vec();
        let mut b = b;
        for (cur_x, cur_y) in x_multi_batch.zip(y_truth_multi_batch) {
            (w, b) = self.train_one_batch(ctx, w, b, cur_x, cur_y, learning_rate_batch);
        }

        (w, b)
    }

    /// Mini-batch gradient descent for training linear regression
    pub fn train_one_batch<QA>(
        &self,
        ctx: &mut Context<F>,
        w: impl IntoIterator<Item = AssignedValue<F>>,
        b: AssignedValue<F>,
        x: impl IntoIterator<Item = impl IntoIterator<Item = QA>>,
        y_truth: impl IntoIterator<Item = QA>,
        learning_rate_batch: f64
    ) -> (Vec<AssignedValue<F>>, AssignedValue<F>)
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + From<AssignedValue<F>> + Copy
    {
        let y_truth: Vec<QA> = y_truth.into_iter().collect();
        let x: Vec<Vec<QA>> = x.into_iter().map(|xi| xi.into_iter().collect()).collect();
        let n_sample = x.len() as f64;

        let mut w: Vec<AssignedValue<F>> = w.into_iter().collect();
        let mut b = b;
        let dim = x[0].len();
        assert!(dim == w.len());

        let learning_rate = ctx.load_constant(self.chip.quantization(learning_rate_batch));
        let one = ctx.load_constant(self.chip.quantization_scale);

        // h_{\theta}(x) = \frac{1}{1+\exp(-\theta x)}
        let y: Vec<QA> = x.iter().map(|xi| {
            let wx = self.chip.inner_product(ctx, w.iter().map(|wi| QA::from(*wi)), (*xi).clone());
            let logit = self.chip.qadd(ctx, wx, b);
            let neg_logit = self.chip.neg(ctx, logit);
            let exp_logit = self.chip.qexp(ctx, neg_logit);
            let exp_logit_p1 = self.chip.qadd(ctx, exp_logit, one);
            let yi = self.chip.qdiv(ctx, one, exp_logit_p1);

            QA::from(yi)
        }).collect();
        // binary cross-entropy (aka., NLL) for sample i
        let mut bce_i = vec![];
        let mut diff_y = vec![];
        for (yi, ti) in y.iter().zip(y_truth.iter()) {
            diff_y.push(self.chip.qsub(ctx, *yi, *ti));
            let ti_deq = match (*ti).into() {
                Existing(AssignedValue { value, .. }) => {
                    self.chip.dequantization(value.evaluate())
                },
                _ => panic!()
            };
            let yi_deq = match (*yi).into() {
                Existing(AssignedValue { value, .. }) => {
                    self.chip.dequantization(value.evaluate())
                },
                _ => panic!()
            };
            if ti_deq == 1.0 {
                bce_i.push(-yi_deq.ln());
            } else {
                bce_i.push(-(1.0 - yi_deq).ln());
            }
        }
        // loss = BCE(y, t)
        let loss: f64 = bce_i.iter().sum::<f64>() / n_sample;
        warn!("loss: {:?}", loss);

        for j in 0..w.len() {
            let mut partial_wj = vec![];
            for i in 0..diff_y.len() {
                partial_wj.push(self.chip.qmul(ctx, diff_y[i], x[i][j]));
            }
            let partial_wj_sum = self.chip.qsum(ctx, partial_wj);
            let diff_wj = self.chip.qmul(ctx, learning_rate, partial_wj_sum);
            w[j] = self.chip.qsub(ctx, w[j], diff_wj);
        }

        let partial_b = self.chip.qsum(ctx, diff_y);
        let diff_b = self.chip.qmul(ctx, learning_rate, partial_b);
        b = self.chip.qsub(ctx, b, diff_b);

        // (w.iter().map(|wi| QA::from(*wi)).collect(), QA::from(b))
        (w, b)
    }
}