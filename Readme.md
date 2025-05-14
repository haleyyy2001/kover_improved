Below is a sample **README** that you can include alongside your new, extended `kover.py`. It covers installation, usage examples, new features, and references, and is written in a typical Markdown style.

---

# Kover: Extended Version

**Kover** is a command-line tool to learn interpretable computational phenotyping models from **k-merized** genomic data. This extended version keeps the original structure and commands of Kover while adding new features for improved debugging, parallelization, and applying predictions to new datasets.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Dataset Commands](#dataset-commands)

   * [Create](#create)
   * [Info](#info)
   * [Split](#split)
5. [Learning Commands](#learning-commands)

   * [SCM (Set Covering Machine)](#scm-set-covering-machine)
   * [Tree (Decision Trees)](#tree-decision-trees)
   * [Predict](#predict-new-feature)
6. [Parallelization](#parallelization)
7. [Citing Kover](#citing-kover)
8. [License](#license)

---

## Features

* **Interpretable Models**: Learns conjunction/disjunction (“SCM”) or decision tree models for phenotypic classification from k-merized data.
* **Extended Debugging**: The extended version provides clearer error messages and more detailed logs when creation or learning fails.
* **Parallelization**: Dataset creation steps (`from-tsv`, `from-contigs`, `from-reads`) and learning steps (`scm`, `tree`) can now use multiple threads or CPU cores.
* **Prediction Command**: Train once and apply models to new datasets without re-training, via the new `predict` subcommand.
* **Backward Compatibility**: All original commands, arguments, and usage are preserved.

---

## Installation

1. **Clone or Download** the repository containing the extended `kover.py` script.
2. **Install Dependencies**:

   * Python 3+
   * Various Python modules:

     * `numpy`
     * `progressbar`
     * `pkg_resources` (usually comes with `setuptools`)
     * `kover` (core library)
     * `argparse` (included in Python 3 standard library)
     * `logging` (also standard library)
3. **Add `kover.py` to PATH** (optional):

   ```bash
   chmod +x kover.py
   # Optionally, move kover.py to a location in your $PATH
   ```
4. **Test the installation**:

   ```bash
   ./kover.py --version
   # or
   python kover.py --version
   ```

---

## Basic Usage

The general command-line pattern for Kover is:

```bash
kover.py <main-command> <subcommand> [options]
```

* **Main Commands**: `dataset` and `learn`
* **Subcommands** under `dataset`: `create`, `info`, and `split`
* **Subcommands** under `learn`: `scm`, `tree`, and **new** `predict`

For detailed help:

```bash
kover.py --help
kover.py dataset --help
kover.py dataset create --help
kover.py learn --help
kover.py learn scm --help
...
```

---

## Dataset Commands

### Create

Create a Kover dataset from genomic data in three different ways:

1. **From TSV**

   ```bash
   kover.py dataset create from-tsv \
     --genomic-data <kmer_matrix.tsv> \
     --phenotype-description "my phenotype" \
     --phenotype-metadata <phenotype_metadata.tsv> \
     --output <output_dataset.kover> \
     --compression 4 \
     --parallel 4 \
     --verbose
   ```

   * `--parallel` (optional) specifies the number of threads (if supported).

2. **From Contigs**

   ```bash
   kover.py dataset create from-contigs \
     --genomic-data <genome_list.tsv> \
     --phenotype-description "my phenotype" \
     --phenotype-metadata <phenotype_metadata.tsv> \
     --output <output_dataset.kover> \
     --kmer-size 31 \
     --singleton-kmers \
     --n-cpu 8 \
     --compression 4 \
     --temp-dir <some_tmp_dir> \
     --parallel 4 \
     --verbose
   ```

3. **From Reads**

   ```bash
   kover.py dataset create from-reads \
     --genomic-data <read_folders_list.tsv> \
     --phenotype-description "my phenotype" \
     --phenotype-metadata <phenotype_metadata.tsv> \
     --output <output_dataset.kover> \
     --kmer-size 31 \
     --kmer-min-abundance 10 \
     --singleton-kmers \
     --n-cpu 8 \
     --compression 4 \
     --parallel 4 \
     --verbose
   ```

### Info

Prints information about a dataset file:

```bash
kover.py dataset info \
  --dataset <dataset_file.kover> \
  --all
```

Use flags like `--genome-type`, `--kmer-count`, `--phenotype-tags` individually to query specific info.

### Split

Split a dataset into training/testing sets and optionally create k-fold splits:

```bash
kover.py dataset split \
  --dataset <dataset_file.kover> \
  --id <split_identifier> \
  --train-size 0.7 \
  --folds 5 \
  --random-seed 12345 \
  --verbose
```

Alternatively, specify exact training and testing genome IDs with `--train-ids` and `--test-ids`.

---

## Learning Commands

### SCM (Set Covering Machine)

Train a conjunction (AND) or disjunction (OR) model:

```bash
kover.py learn scm \
  --dataset <dataset_file.kover> \
  --split <split_identifier> \
  --model-type conjunction disjunction \
  --p 0.1 0.316 0.562 1.0 1.778 3.162 10.0 999999.0 \
  --hp-choice cv \
  --bound-max-genome-size 1000000 \
  --random-seed 123 \
  --n-cpu 4 \
  --output-dir results_scm \
  --progress \
  --verbose
```

### Tree (Decision Trees)

Train a decision tree model (CART-like) with various hyperparameters:

```bash
kover.py learn tree \
  --dataset <dataset_file.kover> \
  --split <split_identifier> \
  --criterion gini crossentropy \
  --max-depth 10 15 \
  --min-samples-split 2 5 \
  --class-importance 1.0 2.0 \
  --hp-choice cv \
  --n-cpu 4 \
  --output-dir results_tree \
  --progress \
  --verbose
```

### Predict (New Feature)

Use an **existing, trained model** (from SCM or Tree) to classify data from a **different** dataset/split:

```bash
kover.py learn predict \
  --model-dir <trained_model_dir> \
  --dataset <new_dataset_file.kover> \
  --split <split_identifier> \
  --output-file predictions.tsv \
  --verbose
```

This command loads the previously trained model (e.g., from `results_scm` or `results_tree`) and generates predictions on a new dataset without re-training.

---

## Parallelization

* **Dataset Creation**:
  Use `--parallel` or `--threads` to specify the number of threads to use. The underlying code attempts to split tasks (like reading files or performing k-mer extraction) across multiple threads or processes.

* **Learning**:
  Already supported by `--n-cpu` (alias `--n-cores`). The tool will run hyperparameter selection (e.g., cross-validation) in parallel processes if possible. Make sure your system has enough resources to run multiple CPU-bound processes.

---

## Citing Kover

If you use **Kover** in your work, please cite the following papers:

```
@article{Drouin2019,
  title={Interpretable genotype-to-phenotype classifiers with performance guarantees},
  author={Drouin, Alexandre and Letarte, Ga{\"e}l and Raymond, Fr{\'e}d{\'e}ric and Marchand, Mario and Corbeil, Jacques and Laviolette, Fran{\c{c}}ois},
  journal={Scientific reports},
  volume={9},
  number={1},
  pages={4071},
  year={2019},
  publisher={Nature Publishing Group}
}

@article{Drouin2016,
  author="Drouin, Alexandre
          and Gigu{\`e}re, S{\'e}bastien
          and D{\'e}raspe, Maxime
          and Marchand, Mario
          and Tyers, Michael
          and Loo, Vivian G.
          and Bourgault, Anne-Marie
          and Laviolette, Fran{\c{c}}ois
          and Corbeil, Jacques",
  title="Predictive computational phenotyping and biomarker discovery using reference-free genome comparisons",
  journal="BMC Genomics",
  year="2016",
  volume="17",
  number="1",
  pages="754",
  issn="1471-2164",
  doi="10.1186/s12864-016-2889-6",
  url="http://dx.doi.org/10.1186/s12864-016-2889-6"
}
```

---

## License

Kover is licensed under the [GNU General Public License (GPLv3)](http://www.gnu.org/licenses/). See the top of `kover.py` for full license text.
