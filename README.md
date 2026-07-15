# Born Effective Charge Evaluation with Equivar

This repository evaluates a fine-tuned Equivar model against reference Born
effective charge (BEC) tensors stored in an extended XYZ test dataset.

The evaluation compares the diagonal BEC components (`xx`, `yy`, and `zz`) for
every atom in every test structure and reports overall and element-resolved
errors.

## Repository contents

The repository is expected to contain the following files:

```text
.
├── README.md
├── evaluate_bec.py
├── test_data.xyz
└── ScAlN_BEC_model.pth
```

- `evaluate_bec.py`: evaluation script.
- `test_split.xyz`: reference structures and per-atom BEC tensors.
- `finetuned_bec_model.pth`: fine-tuned TorchScript Equivar model.

The model filename may be different. Pass its actual path with `--model`.

## Requirements

- Python 3.7 or later
- PyTorch
- PyTorch Geometric
- ASE
- NumPy
- Matplotlib
- Equivar Eval

A CUDA-capable GPU is recommended for a large test dataset, but CPU inference
is also supported.

## Installation

Creating an isolated Python environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

Clone and install
[Equivar Eval](https://github.com/equivar/equivar_eval):

```bash
git clone https://github.com/equivar/equivar_eval.git
cd equivar_eval
python -m pip install -r requirements.txt
python -m pip install .
cd ..
```

The Equivar Eval requirements include PyTorch, PyTorch Geometric,
`torch_scatter`, `torch_sparse`, ASE, NumPy, and e3nn. If automatic
installation of the PyTorch packages fails, install PyTorch and PyTorch
Geometric for your operating system and CUDA version first, then install
Equivar Eval.

Installation can be checked with:

```bash
python -c "from equivar_eval.scripts.calculate import Calculate; print('Equivar Eval is available')"
```

## Test dataset format

`test_split.xyz` must be a multi-frame extended XYZ file. Each frame must
contain:

- atomic species;
- Cartesian atomic positions;
- a periodic simulation cell;
- periodic boundary conditions; and
- one 3 x 3 BEC tensor per atom under the `becs` property.

The expected header format is:

```text
Lattice="..." Properties=species:S:1:pos:R:3:becs:R:9 pbc="T T T"
```

Each atomic line has the following form:

```text
Element x y z Zxx Zxy Zxz Zyx Zyy Zyz Zzx Zzy Zzz
```

## Usage

If the model and test dataset use the filenames shown above, run:

```bash
python evaluate_bec.py \
  --test_xyz test_data.xyz \
  --model ScAlN_BEC_model.pth \
  --output_dir bec_evaluation
```

The script automatically uses CUDA when it is available. To select a device
explicitly, use either:

```bash
python evaluate_bec.py \
  --test_xyz test_data.xyz \
  --model ScAlN_BEC_model.pth \
  --output_dir bec_evaluation \
  --device cuda
```

or:

```bash
python evaluate_bec.py \
  --test_xyz test_data.xyz \
  --model ScAlN_BEC_model.pth \
  --output_dir bec_evaluation \
  --device cpu
```

All command-line options can be displayed with:

```bash
python evaluate_bec.py --help
```

## Outputs

The selected output directory contains:

```text
bec_evaluation/
├── bec_pairs.csv
├── metrics_by_element.csv
├── metrics_overall.csv
└── bec_parity.png
```

- `bec_pairs.csv`: reference value, prediction, and absolute error for every
  atom and each diagonal component.
- `metrics_by_element.csv`: count, MAE, RMSE, and mean signed error for each
  element.
- `metrics_overall.csv`: corresponding metrics over the complete test set.
- `bec_parity.png`: reference-versus-prediction parity plot with a `y = x`
  guide line.

Errors are defined from `reference - prediction`. The reported mean signed
error therefore indicates whether predictions are systematically above or
below the reference values.

## Evaluation scope

The script evaluates only the diagonal components of the BEC tensor:

```text
Zxx, Zyy, Zzz
```

The extended XYZ dataset and model contain all nine tensor components, but the
off-diagonal components are not included in the reported metrics or parity
plot.

The model is evaluated as provided. This script does not perform training or
modify the model file.
