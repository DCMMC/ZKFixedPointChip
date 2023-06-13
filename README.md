# ZKFixedPointChip

ZK Fixed Point Arithmetic with its Application in Machine Learning based on [Halo2](https://github.com/privacy-scaling-explorations/halo2.git) & Axoim's [Halo2-base](https://github.com/axiom-crypto/halo2-lib).

## Features

* FixedPointChip: Fixed point arithmetic and math library
    * Support different kinds of precisions (from `32.32` to `63.63`) with automatically generated polynomial using Remez algorithm
    * Support negative number arithmetics with quantization
    * Support functions: `add`, `sub`, `mul`, `div`, `mod`, `sign`, `clip`, `polynomial`, `bit_xor`, `sum`, `neg`, `exp`, `log`, `pow`, `sqrt`, `max`, `sin`, `cos`, `tan`, `sinh`, `cosh`, etc.
* ZK-LR: LinearRegressionChip/LogisticRegressionChip
    * Support inference with vector multiplication and `sigmoid` (based on `exp`) using FixedPointChip
    * Support training with gradient descent algorithm
* ZK-DT: DecisionTreeChip
    * Support inference with tree traversal over the decision tree
    * Support training by building the decision tree recursively with the calculated Gini Impurity in each node


## Setup

Install rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone this repo:

```bash
git clone https://github.com/DCMMC/ZKFixedPointChip.git
cd ZKFixedPointChip
```

## Build & Run Examples

Fixed Point Arithmetic (exp2, log2, sin):

```bash
cargo run --example fixed_point
```

Linear Regression (Inference & Training):

```bash
cargo run --example linear_regression
```

Logistic Regression (Inference & Training):

```bash
cargo run --example logistic_regression
```

Decision Tree (Inference & Training):

```bash
cargo run --example decision_tree
```
