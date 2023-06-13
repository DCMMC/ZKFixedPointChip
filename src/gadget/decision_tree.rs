use std::{iter, vec};
// use log::{debug, warn};
use halo2_base::{
    utils::{BigPrimeField, fe_to_biguint},
    Context, AssignedValue, gates::{GateInstructions, RangeInstructions}, QuantumCell
};
use super::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::QuantumCell::Constant;
use itertools::{izip, Itertools};

const PRECISION_BITS: u32 = 63;
const MASK_CLS: u64 = 255;

#[derive(Clone, Debug)]
pub struct DecisionTreeChip<F: BigPrimeField> {
    pub chip: FixedPointChip<F, 63>,
    pub lookup_bits: usize,
}

impl<F: BigPrimeField> DecisionTreeChip<F> {
    pub fn new(lookup_bits: usize) -> Self {
        let chip = FixedPointChip::<F, PRECISION_BITS>::default(lookup_bits);

        Self { chip, lookup_bits }
    }

    pub fn inference(
        &self,
        ctx: &mut Context<F>,
        tree: &[F],
        x: &[F],
        max_path_len: usize
    ) -> AssignedValue<F>
    where 
        F: BigPrimeField
    {
        // format of tree: [[data_slot, split, left_index, right_index, cls], ...]
        let zero = Constant(F::zero());
        let mut node_idx = self.chip.gate().add(ctx, zero, Constant(F::zero()));
        let mut x_advice = vec![];
        for xi in x {
            x_advice.push(ctx.load_witness(*xi));
        }
        for _ in 0..max_path_len {
            let node_offset = self.chip.gate().mul(ctx, node_idx, Constant(F::from(5)));
            let data_slot_idx = self.chip.gate().add(ctx, node_offset, Constant(F::from(0)));
            let data_slot = self.chip.gate().select_from_idx(
                ctx, tree.iter().cloned().map(Constant), data_slot_idx
            );
            let mut x_copy = vec![];
            for i in 0..x.len() {
                x_copy.push(ctx.load_witness(x[i]));
                // ensure copy works
                ctx.constrain_equal(x_copy.last().unwrap(), &x_advice[i]);
            }
            let data = self.chip.gate().select_from_idx(
                ctx, x_copy, data_slot);
            let split_idx = self.chip.gate().add(ctx, node_offset, Constant(F::from(1)));
            let split = self.chip.gate().select_from_idx(
                ctx, tree.iter().cloned().map(Constant), split_idx);
            let diff = self.chip.qsub(ctx, data, split);
            let is_less = self.chip.is_neg(ctx, diff);
            let left_idx_idx = self.chip.gate().add(ctx, node_offset, Constant(F::from(2)));
            let left_idx = self.chip.gate().select_from_idx(
                ctx, tree.iter().cloned().map(Constant), left_idx_idx);
            let right_idx_idx = self.chip.gate().add(ctx, node_offset, Constant(F::from(3)));
            let right_idx = self.chip.gate().select_from_idx(
                ctx, tree.iter().cloned().map(Constant), right_idx_idx);
            let next_idx = self.chip.gate().select(ctx, left_idx, right_idx, is_less);
            node_idx = next_idx;
        }
        let offset = self.chip.gate().mul(ctx, node_idx, Constant(F::from(5)));
        let cls_idx = self.chip.gate().add(ctx, offset, Constant(F::from(4)));
        let cls = self.chip.gate().select_from_idx(
            ctx, tree.iter().cloned().map(Constant), cls_idx);
        
        cls
    }

    fn copy_elem(
        &self,
        ctx: &mut Context<F>,
        elem: &AssignedValue<F>
    ) -> AssignedValue<F>
    {
        let x_copy = ctx.load_witness(*elem.value());
        ctx.constrain_equal(&x_copy, elem);

        x_copy
    }

    fn square(
        &self,
        ctx: &mut Context<F>,
        x: AssignedValue<F>
    ) -> AssignedValue<F>
    {
        // only support positive x (quantized)
        let x_square = self.chip.gate().mul(ctx, x, x);
        let scale = self.chip.quantization_scale;
        let num_bits = (2 * PRECISION_BITS + 1) as usize;
        let (res, _) = self.chip.range_gate().div_mod(ctx, x_square, fe_to_biguint(&scale), num_bits * 2);

        res
    }

    pub fn gini(
        &self,
        ctx: &mut Context<F>,
        dataset_x: impl IntoIterator<Item = AssignedValue<F>>,
        dataset_y: impl IntoIterator<Item = AssignedValue<F>>,
        masks: impl IntoIterator<Item = AssignedValue<F>>,
        data_slot: usize,
        num_feature: usize,
        num_class: usize,
        split: AssignedValue<F>
    ) -> AssignedValue<F>
    where
        F: BigPrimeField
    {
        let mut num_group1: AssignedValue<F> = ctx.load_zero();
        let mut num_group2: AssignedValue<F> = ctx.load_zero();
        // ((x_ij, y_i), mask_i) where j is the data slot
        let targets = dataset_x.into_iter().skip(data_slot).step_by(num_feature).zip(
            dataset_y.into_iter()).zip(masks.into_iter());
        let one = Constant(F::one());
        let mut proportion_cls_grp1 = vec![ctx.load_zero(); num_class];
        let mut proportion_cls_grp2 = vec![ctx.load_zero(); num_class];
        let mut cls_adv = vec![];
        for cls_i in 0..num_class {
            cls_adv.push(ctx.load_constant(F::from(cls_i as u64)));
        }
        let mut cnt = ctx.load_zero();
        for ((x_ij, y_i), mask_i) in targets {
            // warn!("x_ij: {:?}, y_i: {:?}, mask_i: {:?}", self.chip.dequantization(*x_ij.value()), y_i.value(), mask_i.value());
            cnt = self.chip.gate().add(ctx, cnt, mask_i);
            let diff = self.chip.qsub(ctx, x_ij, split);
            let is_less = self.chip.is_neg(ctx, diff);
            let mask_i_copy = self.copy_elem(ctx, &mask_i);
            let is_less_mask = self.chip.gate().mul(ctx, is_less, mask_i);
            num_group1 = self.chip.gate().add(ctx, num_group1, is_less_mask);
            let is_greater = self.chip.gate().sub(ctx, one, is_less);
            let is_geater_mask = self.chip.gate().mul(ctx, is_greater, mask_i_copy);
            num_group2 = self.chip.gate().add(ctx, num_group2, is_geater_mask);
            // warn!("is_less_mask: {:?}, is_greater_mask: {:?}", is_less_mask.value(), is_geater_mask.value());
            for cls_i in 0..num_class {
                let y_i_copy = self.copy_elem(ctx, &y_i);
                let is_cls_i = self.chip.gate().is_equal(ctx, y_i_copy, cls_adv[cls_i]);
                let is_cls_i_grp1 = self.chip.gate().mul(ctx, is_cls_i, is_less_mask);
                let is_cls_i_grp2 = self.chip.gate().mul(ctx, is_cls_i, is_geater_mask);
                proportion_cls_grp1[cls_i] = self.chip.gate().add(ctx, proportion_cls_grp1[cls_i], is_cls_i_grp1);
                proportion_cls_grp2[cls_i] = self.chip.gate().add(ctx, proportion_cls_grp2[cls_i], is_cls_i_grp2);
            }
        }
        // warn!(
        //     "prop_cls_grp1: {:?}, prop_cls_grp2: {:?}",
        //     proportion_cls_grp1.iter().map(|x| x.value()).collect_vec(),
        //     proportion_cls_grp2.iter().map(|x| x.value()).collect_vec()
        // );
        // if cnt is zero, change it to one to avoid dividing zero
        let cnt_zero = self.chip.gate().is_zero(ctx, cnt);
        cnt = self.chip.gate().select(ctx, one, cnt, cnt_zero);
        let num_samples = self.copy_elem(ctx, &cnt);
        let num_bits = (2 * PRECISION_BITS + 1) as usize;
        let scale = QuantumCell::Constant(self.chip.quantization_scale);
        num_group1 = self.chip.gate().mul(ctx, num_group1, scale);
        // add a small epsilon, avoid for dividing zero
        num_group1 = self.chip.gate().add(ctx, num_group1, one);
        num_group2 = self.chip.gate().mul(ctx, num_group2, scale);
        num_group2 = self.chip.gate().add(ctx, num_group2, one);
        let mut gini_grp1 = ctx.load_constant(self.chip.quantization_scale);
        for pi in proportion_cls_grp1 {
            // mul twice
            let pi_q = self.chip.gate().mul(ctx, pi, scale);
            let pi_q = self.chip.gate().mul(ctx, pi_q, scale);
            let (pi_cls, _) = self.chip.range_gate().div_mod_var(ctx, pi_q, num_group1, num_bits * 2, num_bits);
            let pi_cls_square = self.square(ctx, pi_cls);
            gini_grp1 = self.chip.gate().sub(ctx, gini_grp1, pi_cls_square);
        }
        // println!("gini_grp1: {:?}", self.chip.dequantization(*gini_grp1.value()));
        let (weight_grp1, _) = self.chip.range_gate().div_mod_var(ctx, num_group1, cnt, num_bits * 2, num_bits);
        gini_grp1 = self.chip.gate().mul(ctx, gini_grp1, weight_grp1);
        // rescale
        gini_grp1 = self.chip.range_gate().div_mod(ctx, gini_grp1, fe_to_biguint(scale.value()), num_bits).0;
        let mut gini_grp2 = ctx.load_constant(self.chip.quantization_scale);
        for pi in proportion_cls_grp2 {
            let pi_q = self.chip.gate().mul(ctx, pi, scale);
            let pi_q = self.chip.gate().mul(ctx, pi_q, scale);
            let (pi_cls, _) = self.chip.range_gate().div_mod_var(ctx, pi_q, num_group2, num_bits * 2, num_bits);
            let pi_cls_square = self.square(ctx, pi_cls);
            gini_grp2 = self.chip.gate().sub(ctx, gini_grp2, pi_cls_square);
        }
        // println!("gini_grp2: {:?}", self.chip.dequantization(*gini_grp2.value()));
        let (weight_grp2, _) = self.chip.range_gate().div_mod_var(ctx, num_group2, num_samples, num_bits * 2, num_bits);
        gini_grp2 = self.chip.gate().mul(ctx, gini_grp2, weight_grp2);
        gini_grp2 = self.chip.range_gate().div_mod(ctx, gini_grp2, fe_to_biguint(scale.value()), num_bits).0;
        let gini = self.chip.gate().add(ctx, gini_grp1, gini_grp2);
        // fix for leaf node
        let fix_zero = self.chip.range_gate().is_less_than_safe(ctx, num_samples, 2);
        let gini = self.chip.gate().select(ctx, scale, gini, fix_zero);

        gini
    }

    fn copy_list(
        &self,
        ctx: &mut Context<F>,
        x: &[AssignedValue<F>]
    ) -> Vec<AssignedValue<F>>
    where
        F: BigPrimeField
    {
        let mut copy = vec![];
        for x_i in x.into_iter() {
            copy.push(self.copy_elem(ctx, &x_i));
        }

        copy
    }

    pub fn get_split(
        &self,
        ctx: &mut Context<F>,
        dataset_x: impl IntoIterator<Item = AssignedValue<F>>,
        dataset_y: impl IntoIterator<Item = AssignedValue<F>>,
        masks: impl IntoIterator<Item = AssignedValue<F>>,
        num_feature: usize,
        num_class: usize,
    ) -> (AssignedValue<F>, AssignedValue<F>)
    where
        F: BigPrimeField
    {
        let num_bits = (2 * PRECISION_BITS + 1) as usize;
        let dataset_x: Vec<AssignedValue<F>> = dataset_x.into_iter().collect();
        let dataset_x_copy = self.copy_list(ctx, &dataset_x);
        let dataset_y: Vec<AssignedValue<F>> = dataset_y.into_iter().collect();
        let masks: Vec<AssignedValue<F>> = masks.into_iter().collect();
        let mut masks_feature = vec![];
        for _ in 0..num_feature {
            masks_feature.push(self.copy_list(ctx, &masks));
        }
        let mut masks_copy = vec![];
        for i in 0..dataset_y.len() {
            for j in 0..num_feature {
                masks_copy.push(masks_feature[j][i]);
            }
        }
        let slots = iter::repeat(0..num_feature).take(dataset_y.len()).flatten();
        let mut best_slot = ctx.load_zero();
        let mut best_split = self.copy_elem(ctx, &dataset_x[0]);
        // maximum of gini impurity is 0.5, so we set 1.0 as the initial value
        let mut best_gini = ctx.load_constant(self.chip.quantization_scale);
        for (slot, split, mask) in izip!(slots.into_iter(), dataset_x_copy.into_iter(), masks_copy.into_iter()) {
            let mut masks_tmp = self.copy_list(ctx, &masks);
            for i in 0..masks_tmp.len() {
                masks_tmp[i] = self.chip.gate().mul(ctx, masks_tmp[i], mask);
            }
            let dataset_x_tmp = self.copy_list(ctx, &dataset_x);
            let dataset_y_tmp = self.copy_list(ctx, &dataset_y);
            let gini = self.gini(ctx, dataset_x_tmp, dataset_y_tmp, masks_tmp, slot, num_feature, num_class, split);
            // warn!("gini: {:?}, split: {:?}, slot: {:?}", self.chip.dequantization(*gini.value()), self.chip.dequantization(*split.value()), slot);
            let is_better = self.chip.range_gate().is_less_than(ctx, gini, best_gini, num_bits);
            best_gini = self.chip.gate().select(ctx, gini, best_gini, is_better);
            let slot_adv = ctx.load_constant(F::from(slot as u64));
            best_slot = self.chip.gate().select(ctx, slot_adv, best_slot, is_better);
            best_split = self.chip.gate().select(ctx, split, best_split, is_better);
        }

        (best_slot, best_split)
    }

    /// get the most common class of a given group of samples
    fn common_cls(
        &self,
        ctx: &mut Context<F>,
        dataset_y: impl IntoIterator<Item = AssignedValue<F>>,
        masks: impl IntoIterator<Item = AssignedValue<F>>,
        num_class: usize,
    ) -> AssignedValue<F>
    where
        F: BigPrimeField
    {
        let num_bits = (2 * PRECISION_BITS + 1) as usize;
        let mut cls = vec![];
        let mut cnt_cls = vec![];
        for i in 0..num_class {
            cls.push(ctx.load_constant(F::from(i as u64)));
            cnt_cls.push(ctx.load_zero());
        }
        for (yi, mask) in dataset_y.into_iter().zip(masks.into_iter()) {
            for i in 0..num_class {
                let is_cls = self.chip.gate().is_equal(ctx, yi, cls[i]);
                let is_cls_mask = self.chip.gate().mul(ctx, is_cls, mask);
                cnt_cls[i] = self.chip.gate().add(ctx, cnt_cls[i], is_cls_mask);
            }
        }
        let mut out_cls = ctx.load_zero();
        let mut max_cnt = ctx.load_zero();
        for (cls_idx, cnt) in cnt_cls.iter().enumerate() {
            let is_better = self.chip.range_gate().is_less_than(ctx, max_cnt, *cnt, num_bits);
            max_cnt = self.chip.gate().select(ctx, *cnt, max_cnt, is_better);
            out_cls = self.chip.gate().select(ctx, cls[cls_idx], out_cls, is_better);
        }

        // println!("common_cls: {:?}", out_cls.value());
        out_cls
    }

    pub fn train(
        &self,
        ctx: &mut Context<F>,
        dataset_x: impl IntoIterator<Item = AssignedValue<F>>,
        dataset_y: impl IntoIterator<Item = AssignedValue<F>>,
        num_feature: usize,
        num_class: usize,
        max_depth: usize,
        min_size: usize
    ) -> Vec<AssignedValue<F>>
    where
        F: BigPrimeField
    {
        // construct a complete binary tree with depth = max_depth
        let dataset_x = dataset_x.into_iter().collect_vec();
        let dataset_y = dataset_y.into_iter().collect_vec();
        assert!(dataset_x.len() == dataset_y.len() * num_feature);
        assert!(min_size >= 1);
        assert!(max_depth >= 1);
        assert!(num_class >= 2);
        let mask_cls = ctx.load_constant(F::from(MASK_CLS));
        let min_size = ctx.load_constant(F::from((min_size + 1) as u64));
        let num_bits = (2 * PRECISION_BITS + 1) as usize;
        let init_mask = vec![ctx.load_constant(F::one()); dataset_y.len()];
        let mut queue = vec![init_mask];
        let one = ctx.load_constant(F::one());
        let two = ctx.load_constant(F::from(2u64));
        let zero = ctx.load_zero();
        let mut tree = vec![];
        let mut idx_list = vec![];
        for i in 0..dataset_y.len() {
            idx_list.push(ctx.load_constant(F::from((i * num_feature) as u64)));
        }
        for layer in 0..max_depth {
            let size = queue.len();
            for node_idx in 0..size {
                let masks = queue.remove(0);
                let masks_copy = self.copy_list(ctx, &masks);
                let cnt_sample = self.chip.qsum(ctx, masks_copy);
                let leaf = self.chip.range_gate().is_less_than(ctx, cnt_sample, min_size, num_bits);
                let mut masks_left = self.copy_list(ctx, &masks);
                let mut masks_right = self.copy_list(ctx, &masks);
                let dataset_x_tmp = self.copy_list(ctx, &dataset_x);
                let dataset_y_tmp = self.copy_list(ctx, &dataset_y);
                let masks_tmp = self.copy_list(ctx, &masks);
                let (best_slot, best_split) = self.get_split(ctx, dataset_x_tmp, dataset_y_tmp, masks_tmp, num_feature, num_class);
                for idx in 0..dataset_y.len() {
                    let value_idx = self.chip.gate().add(ctx, best_slot, idx_list[idx]);
                    let dataset_x_copy = self.copy_list(ctx, &dataset_x);
                    let value = self.chip.gate().select_from_idx(ctx, dataset_x_copy, value_idx);
                    let diff = self.chip.qsub(ctx, value, best_split);
                    let is_left = self.chip.is_neg(ctx, diff);
                    masks_left[idx] = self.chip.gate().mul(ctx, masks_left[idx], is_left);
                    let is_right = self.chip.gate().not(ctx, is_left);
                    masks_right[idx] = self.chip.gate().mul(ctx, masks_right[idx], is_right);
                }

                let dataset_y = self.copy_list(ctx, &dataset_y);
                let cls = self.common_cls(ctx, dataset_y, masks, num_class);
                let cur_idx = ctx.load_constant(F::from(2u64.pow(layer as u32) - 1 + node_idx as u64));
                // warn!("cur_idx: {:?}, best_slot: {:?}, best_split: {:?}", cur_idx.value(), best_slot.value(), self.chip.dequantization(*best_split.value()));
                // warn!("masks_left: {:?}, masks_right: {:?}", masks_left.iter().map(|x| x.value()).collect_vec(), masks_right.iter().map(|x| x.value()).collect_vec());
                if layer < max_depth - 1 {
                    queue.push(masks_left);
                    queue.push(masks_right);
                    let cls = self.chip.gate().select(ctx, cls, mask_cls, leaf);
                    let left_child = self.chip.gate().mul_add(ctx, cur_idx, two, one);
                    let right_child = self.chip.gate().mul_add(ctx, cur_idx, two, two);
                    let left_child = self.chip.gate().select(ctx, cur_idx, left_child, leaf);
                    let right_child = self.chip.gate().select(ctx, cur_idx, right_child, leaf);
                    let best_slot = self.chip.gate().select(ctx, zero, best_slot, leaf);
                    let best_split = self.chip.gate().select(ctx, zero, best_split, leaf);

                    let node = vec![best_slot, best_split, left_child, right_child, cls];
                    tree.push(node);
                } else {
                    // termin all nodes to be leaf nodes in last layer of the decision tree
                    let left_child = self.copy_elem(ctx, &cur_idx);
                    let right_child = cur_idx;
                    let node = vec![zero, zero, left_child, right_child, cls];
                    tree.push(node);
                }
            }
        }
        tree.iter().flatten().copied().collect_vec()
    }
}