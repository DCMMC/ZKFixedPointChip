use halo2_base::utils::{ScalarField, BigPrimeField, fe_to_biguint};
use halo2_base::AssignedValue;
use halo2_base::Context;
use halo2_scaffold::gadget::decision_tree::DecisionTreeChip;
#[allow(unused_imports)]
use halo2_scaffold::scaffold::{mock, prove};
use itertools::Itertools;
use num_traits::ToPrimitive;
use std::collections::HashSet;
use std::env::{var, set_var};
use std::process::{Command, Stdio};

pub fn dot2svg(dot: &str) {
    let dot = Command::new("echo").arg(dot).stdout(Stdio::piped()).spawn().unwrap();
    let svg = Command::new("dot")
            .arg("-Tsvg")
            .stdin(Stdio::from(dot.stdout.unwrap()))
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
    let output = svg.wait_with_output().unwrap();
    let result = std::str::from_utf8(&output.stdout).unwrap();
    std::fs::write("./figure/dt.svg", result).expect("Unable to write file");
}

pub fn train<F: ScalarField>(
    ctx: &mut Context<F>,
    dataset: Vec<Vec<f64>>,
    make_public: &mut Vec<AssignedValue<F>>
) where F: BigPrimeField {
    let lookup_bits =
        var("LOOKUP_BITS").unwrap_or_else(|_| panic!("LOOKUP_BITS not set")).parse().unwrap();
    let chip = DecisionTreeChip::<F>::new(lookup_bits);

    let x = &dataset[0];
    let y = &dataset[1];
    let x_deq: Vec<AssignedValue<F>> = x.iter().map(|xi| {
        ctx.load_witness(chip.chip.quantization(*xi))
    }).collect();
    let y_adv = y.iter().map(|yi| {
        ctx.load_witness(F::from(*yi as u64))
    }).collect_vec();

    let num_feature = 2;
    let num_class = 2;
    let max_depth = 4;
    let min_size = 2;

    let tree = chip.train(ctx, x_deq, y_adv, num_feature, num_class, max_depth, min_size);
    println!("decision tree:");
    let mut dt_dot = "digraph G {\n  graph [ordering=\"out\"];\n".to_string();
    let mut leaf_nodes = HashSet::new();
    let cnt = tree.iter().copied().map(
        |x| fe_to_biguint(x.value()).to_u128().unwrap()
    ).collect::<Vec<u128>>().chunks(5).into_iter().enumerate().inspect(
        |(node_idx, x)| {
            let mut node = x.clone().iter().copied().map(|x| x as f64).collect_vec();
            let parent_idx = (*node_idx as i32 + 1) / 2 - 1;
            if node[node.len() - 1] == 255.0 {
                node[1] = chip.chip.dequantization(F::from_u128(node[1] as u128));
                dt_dot.push_str(&format!(
                    "  node{} [label=<X<SUB>{}</SUB> &lt; {:.2}>,style=\"filled,rounded\",shape=\"box\",fillcolor=\"palegreen\"]\n",
                    node_idx, node[0], node[1]
                ));
                if node_idx > &0 {
                    dt_dot.push_str(&format!("  node{} -> node{}\n", parent_idx, node_idx));
                }
            } else {
                if leaf_nodes.contains(&(parent_idx as usize)) {
                    dt_dot.push_str(&format!(
                        "  node{} [label=\"{}\",style=\"dashed,filled\",shape=\"circle\",fillcolor=\"cadetblue1\"]\n",
                        node_idx, node[node.len() - 1]
                    ));
                    dt_dot.push_str(&format!("  node{} -> node{} [style=\"dashed\"]\n", parent_idx, node_idx));
                } else {
                    dt_dot.push_str(&format!(
                        "  node{} [label=\"{}\",style=\"filled\",shape=\"circle\",fillcolor=\"yellow\"]\n",
                        node_idx, node[node.len() - 1]
                    ));
                    dt_dot.push_str(&format!("  node{} -> node{}\n", parent_idx, node_idx));
                }
                leaf_nodes.insert(*node_idx);
            }
            println!("{:?}", node);
        }).count();
    dt_dot.push_str("}\n");
    println!("#nodes: {:?}", cnt);
    // println!("dot: {}", dt_dot);
    dot2svg(&dt_dot);

    make_public.extend(tree);
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
    let chip = DecisionTreeChip::<F>::new(lookup_bits);

    let x_deq: Vec<F> = x.iter().map(|xi| {
        chip.chip.quantization(*xi)
    }).collect();

    let tree = [
        // node 0
        F::from(0),
        chip.chip.quantization(1.3),
        F::from(1),
        F::from(2),
        F::from(255), // padding cls
        // node 1
        F::from(1),
        chip.chip.quantization(-3.5),
        F::from(3),
        F::from(4),
        F::from(255),
        // node 2
        F::from(0),
        F::from(0),
        F::from(2),
        F::from(2),
        F::from(1),
        // node 3
        F::from(0),
        F::from(0),
        F::from(3),
        F::from(3),
        F::from(1),
        // node 4
        F::from(0),
        F::from(0),
        F::from(4),
        F::from(4),
        F::from(0),
        // node 5 (fake node)
        F::from(0),
        F::from(0),
        F::from(5),
        F::from(5),
        F::from(0),
        // node 6 (fake node)
        F::from(0),
        F::from(0),
        F::from(6),
        F::from(6),
        F::from(1)
    ];

    let y = chip.inference(ctx, &tree, &x_deq, 3);
    println!("y: {:?}", y);
    make_public.push(y);
}

fn main() {
    set_var("RUST_LOG", "warn");
    env_logger::init();
    // genrally lookup_bits is degree - 1
    set_var("LOOKUP_BITS", 15.to_string());
    set_var("DEGREE", 16.to_string());

    mock(inference, vec![-1.2, 0.1]);
    prove(inference, vec![-1.2, 0.1], vec![2.1, 3.2]);

    let x =  vec![[2.771244718,1.784783929],
    [1.,1.],
    [3.,2.],
    [3.,2.],
    [2.,2.],
    [7.,3.],
    [9.00220326,3.339047188],
    [7.444542326,0.476683375],
    [10.12493903,3.234550982],
    [6.642287351,3.319983761]
].iter().flatten().copied().collect_vec();
    let dataset = vec![
        x,
        // y
        vec![1., 0., 0., 1., 0., 1., 1., 0., 1., 1.]
    ];
    mock(train, dataset.clone());
    let x_dummy = vec![[2.771244718,1.784783929],
    [1.728571309,1.169761413],
    [3.678319846,2.81281357],
    [3.961043357,2.61995032],
    [2.999208922,2.209014212],
    [7.497545867,3.162953546],
    [9.00220326,3.339047188],
    [7.444542326,0.476683375],
    [10.12493903,3.234550982],
    [6.642287351,3.319983761]
].iter().flatten().copied().collect_vec();
    let dataset_dummy = vec![
        x_dummy, vec![0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
    ];
    mock(train, dataset_dummy.clone());
    prove(train, dataset, dataset_dummy);
}
