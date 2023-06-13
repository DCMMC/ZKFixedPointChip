use halo2_base::{
    utils::BigPrimeField,
    QuantumCell, Context, AssignedValue
};
use log::warn;
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use std::convert::From;

#[derive(Clone, Debug)]
pub struct LinearRegressionChip<F: BigPrimeField> {
    pub chip: FixedPointChip<F, 32>,
    pub lookup_bits: usize,
}

impl<F: BigPrimeField> LinearRegressionChip<F> {
    pub fn new(lookup_bits: usize) -> Self {
        let chip = FixedPointChip::<F, 32>::default(lookup_bits);

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
        let y = self.chip.qadd(ctx, wx, b);

        y
    }

    /// Mini-batch gradient descent for training linear regression
    pub fn train_one_batch<QA>(
        &self,
        ctx: &mut Context<F>,
        w: impl IntoIterator<Item = AssignedValue<F>>,
        b: AssignedValue<F>,
        x: impl IntoIterator<Item = impl IntoIterator<Item = QA>>,
        y_truth: impl IntoIterator<Item = QA>,
        learning_rate: f64
    ) -> (Vec<QA>, QA)
    where
        F: BigPrimeField, QA: Into<QuantumCell<F>> + From<AssignedValue<F>> + Copy
    {
        let y_truth: Vec<QA> = y_truth.into_iter().collect();
        let x: Vec<Vec<QA>> = x.into_iter().map(|xi| xi.into_iter().collect()).collect();
        let n_sample = x.len() as f64;
        // debug!("n_sample: {:?}", n_sample);

        let mut w: Vec<AssignedValue<F>> = w.into_iter().collect();
        let mut b = b;
        let dim = x[0].len();
        assert!(dim == w.len());

        let learning_rate = ctx.load_witness(self.chip.quantization(learning_rate / n_sample));

        let y: Vec<QA> = x.iter().map(|xi| {
            let xw = self.chip.inner_product(ctx, (*xi).clone(), w.iter().map(|wi| QA::from(*wi)));
            let yi = self.chip.qadd(ctx, xw, b);

            QA::from(yi)
        }).collect();
        let mut diff_y = vec![];
        for (yi, ti) in y.iter().zip(y_truth.iter()) {
            diff_y.push(self.chip.qsub(ctx, *yi, *ti));
        }
        let mut loss = 0.;
        for i in 0..diff_y.len() {
            loss += self.chip.dequantization(*diff_y[i].value()).powi(2);
        }
        // loss = 0.5 * MSE(y, t)
        loss /= n_sample * 2.0;
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

        (w.iter().map(|wi| QA::from(*wi)).collect(), QA::from(b))
    }
}