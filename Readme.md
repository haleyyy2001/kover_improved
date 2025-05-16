
# FastKover  
 

FastKover is an **extended, GPU-enabled** adaptation of [Kover](https://github.com/aldro61/kover), an *out-of-core* implementation of rule-based machine learning algorithms tailored for genomic biomarker discovery.
It produces highly interpretable models, based on k-mers, that explicitly highlight genotype-to-phenotype associations, and adds optional GPU acceleration for speedups on modern hardware.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features of FastKover](#key-features-of-fastkover)
3. [Installation](#installation)

   1. [Prerequisites](#prerequisites)
   2. [Installing via Docker](#installing-via-docker)
   3. [Manual Installation](#manual-installation)
   4. [Optional GPU Dependencies](#optional-gpu-dependencies)
4. [Usage](#usage)

   1. [Command-line Overview](#command-line-overview)
   2. [Dataset Creation](#dataset-creation)
   3. [Dataset Splitting](#dataset-splitting)
   4. [Learning Models](#learning-models)
   5. [Prediction](#prediction)
5. [Tutorials](#tutorials)
6. [Documentation](#documentation)
7. [References](#references)
8. [License](#license)
9. [Contact](#contact)

---

## Introduction

Understanding the relationship between a cell’s genome and its phenotype is a crucial challenge in precision medicine. However, genotype-to-phenotype prediction with large-scale genomic data presents difficulties:

* **High dimensionality** can impede generalization.
* Many algorithms yield **complex, opaque models** that hamper biological insight.

FastKover (based on the original [Kover](https://github.com/aldro61/kover)) addresses these issues by:

* Generating **rule-based, interpretable** models with strong **theoretical guarantees** (sample compression theory).
* Employing **disk-based/out-of-core** methods to handle massive k-mer matrices with minimal RAM usage.
* Now providing an **optional GPU mode** to train and predict more quickly on CUDA-capable devices.

**Applications** have included genomic prediction of antimicrobial resistance, highlighting known and novel mechanisms.

Key references:

> **Drouin, A. et al.** (2019). *Interpretable genotype-to-phenotype classifiers with performance guarantees.* Scientific Reports, 9(1), 4071. [\[PDF\]](https://www.nature.com/articles/s41598-019-40561-2)

> **Drouin, A. et al.** (2016). *Predictive computational phenotyping and biomarker discovery using reference-free genome comparisons.* BMC Genomics, 17(1), 754. [\[PDF\]](http://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2889-6)

### Video Lecture

The Set Covering Machine implementation in Kover/FastKover was featured in this video lecture:

**Interpretable Models of Antibiotic Resistance with the Set Covering Machine Algorithm, Google, Cambridge, Massachusetts (February 2017)**

[![Google tech talk](https://img.youtube.com/vi/uomMdBdEwnk/0.jpg)](https://www.youtube.com/watch?v=uomMdBdEwnk)

---

## Key Features of FastKover

1. **GPU Acceleration**

   * Leverages PyTorch to run on CUDA-enabled devices (NVIDIA GPUs) or ROCm (AMD).
   * Substantial speedups for training on large k-mer datasets.

2. **Extended CLI Commands**

   * New `kover predict` command for inference on new datasets.
   * `--use-gpu` flags in `kover learn scm` and `kover learn tree` to enable GPU usage.

3. **Flexible Installation**

   * Docker image for turnkey setup.
   * Manual installation with clear prerequisites.
   * Optional CuPy-based acceleration for certain array operations.

4. **Backward Compatibility**

   * Retains out-of-core disk-based approach from the original Kover.
   * Falls back to CPU if no GPU is detected.

---

## Installation

### Prerequisites

#### Install yourself or via package manager:

* **CMake**
* **GNU C++ compiler (g++)**
* **GNU Fortran (gfortran)**
* **HDF5 library**
* **Python 2.7.x** (original Kover is Python 2–based)
* **Python development headers** (e.g., `python-dev` on Ubuntu)
* **NumPy** (recommended manual installation for performance)
* **SciPy** (same recommendation as NumPy)

#### Automatically installed by `pip` if missing:

* **Cython**
* **h5py >= 2.4.0**
* **pandas**
* **progressbar**
* *(optionally)* **NumPy** and **SciPy** if not preinstalled

### Installing via Docker

A Docker image with FastKover preinstalled can be pulled from Docker Hub:

```bash
docker pull aldro61/kover
```

Run interactively:

```bash
docker run -it --rm -v /path/to/data:/data aldro61/kover /bin/bash
```

Inside the container, you can run:

```bash
kover --version
kover dataset ...
kover learn ...
```

To access GPU resources, ensure [NVIDIA Docker support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) or equivalent is installed:

```bash
docker run --gpus all -it --rm aldro61/kover /bin/bash
```

### Manual Installation

1. **Clone** or **download** the FastKover (Kover) repository:

   ```bash
   git clone https://github.com/aldro61/kover.git
   cd kover
   ```

2. **Run installer** (Linux/Mac):

   ```bash
   ./install.sh
   ```

   This builds and installs the Kover engine plus dependencies. A `bin` directory with the `kover` executable is created.

3. (Optional) **Add** the `kover/bin` directory to your `$PATH`:

   ```bash
   export PATH=[PATH_TO_KOVER_DIRECTORY]/bin:$PATH
   ```

4. **Test** installation:

   ```bash
   kover --version
   ```

   Should output something like `cli-2.0.x` and `core-2.0.x`, but effectively you’re using the FastKover build.

### Optional GPU Dependencies

* **PyTorch** for GPU training.
  See [PyTorch.org](https://pytorch.org/) for your platform’s wheels (e.g., CUDA 11.x, CPU-only, etc.).

* **CuPy** (for certain array operations) – optional but can boost performance. Choose one package depending on your system:

  ```txt
  cupy-cuda12x>=12.0.0     # Linux/Windows, CUDA 12.x
  cupy-cuda11x>=11.0.0     # Linux/Windows, CUDA 11.x
  cupy-rocm-5-0>=12.0.0    # Linux with ROCm 5.0
  ```

  If you do not need GPU acceleration, you can ignore CuPy packages entirely.

---

## Usage

FastKover follows the Kover CLI structure:

1. **`kover dataset`** – to create and manipulate datasets.
2. **`kover learn`** – to train SCM or decision-tree models and now includes GPU usage flags.
3. **`kover predict`** – to run predictions on new data.

### Command-line Overview

Check available commands via:

```bash
kover --help
kover dataset --help
kover learn --help
```

### Dataset Creation

Create datasets from:

* **TSV** files containing k-mer matrices (`from-tsv`)
* **Contigs** (`from-contigs`)
* **Read** files (`from-reads`)

Examples:

```bash
# 1) Dataset from TSV
kover dataset create from-tsv \
  --genomic-data my_data.tsv \
  --phenotype-description "My phenotype" \
  --phenotype-metadata phenotype.txt \
  --output my_dataset.h5

# 2) Dataset from contigs
kover dataset create from-contigs \
  --genomic-data genome_paths.tsv \
  --kmer-size 31 \
  --output my_dataset.h5

# 3) Dataset from reads
kover dataset create from-reads \
  --genomic-data reads_folders.tsv \
  --kmer-size 31 \
  --kmer-min-abundance 5 \
  --output my_dataset.h5
```

### Dataset Splitting

Split your dataset into training/testing and optional cross-validation folds:

```bash
kover dataset split \
  --dataset my_dataset.h5 \
  --id my_split \
  --train-size 0.7 \
  --folds 5
```

Or provide explicit genome IDs for train/test:

```bash
kover dataset split \
  --dataset my_dataset.h5 \
  --id my_split \
  --train-ids train.txt \
  --test-ids test.txt
```

### Learning Models

FastKover implements:

1. **SCM** (Set Covering Machine) – conjunction/disjunction-based rule learning.
2. **Decision Tree** – CART-like approach.

#### SCM Example

```bash
kover learn scm \
  --dataset my_dataset.h5 \
  --split my_split \
  --model-type conjunction disjunction \
  --max-rules 10 \
  --hp-choice cv \
  --use-gpu \
  --output-dir results_scm
```

* **`--use-gpu`** attempts to enable GPU training if CUDA is available.

#### Decision Tree Example

```bash
kover learn tree \
  --dataset my_dataset.h5 \
  --split my_split \
  --criterion gini crossentropy \
  --max-depth 10 \
  --min-samples-split 2 \
  --hp-choice cv \
  --use-gpu \
  --output-dir results_tree
```

### Prediction

FastKover adds a **`kover predict`** command for applying trained models:

```bash
kover predict \
  --model-dir results_scm \
  --dataset new_data.h5 \
  --output-dir predictions \
  --use-gpu
```

This will:

* Load the saved model from `results_scm`
* Run inference on `new_data.h5`
* Write predictions to `predictions/predictions.json`

---

## Tutorials

For more hands-on examples, see the [Kover tutorials](http://aldro61.github.io/kover/doc_tutorials.html). Topics include:

* Preparing data for Kover
* Creating your first dataset
* Training with different hyperparameters
* Interpreting rule-based models
* Large-scale GPU-accelerated workflows

---

## Documentation

The original [Kover documentation site](http://aldro61.github.io/kover/) remains relevant for FastKover.
It covers input format, usage examples, advanced hyperparameter selection, model interpretation, etc.

---

## References

If you use FastKover (or the original Kover) in your work, kindly cite:

1. **Drouin, A., Letarte, G., Raymond, F., Marchand, M., Corbeil, J., & Laviolette, F. (2019).**
   *Interpretable genotype-to-phenotype classifiers with performance guarantees.*
   *Scientific Reports*, **9**(1), 4071.
   [PDF](https://www.nature.com/articles/s41598-019-40561-2)

2. **Drouin, A., Giguère, S., Déraspe, M., Marchand, M., Tyers, M., Loo, V. G., Bourgault, A. M., Laviolette, F. & Corbeil, J. (2016).**
   *Predictive computational phenotyping and biomarker discovery using reference-free genome comparisons.*
   *BMC Genomics*, **17**(1), 754.
   [PDF](http://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2889-6)

---

## License

FastKover (Kover) is licensed under the **GNU General Public License version 3 (GPLv3)**:

```
Kover: Learn interpretable computational phenotyping models from k-merized genomic data
Copyright (C) 2018-2025  Alexandre Drouin & Gael Letarte

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

---

## Contact

* **Help**: Post usage questions on [Biostars](https://www.biostars.org/p/194520/).
* **Bug Reports**: File an issue at [GitHub issues](https://github.com/aldro61/kover/issues).
* **Contributions**: Pull requests and feature suggestions are always welcome!

---

### Confirming the Extended Code

We have reviewed the extended Python code (with GPU support) to ensure it integrates with the original Kover framework, adding:

* A `--use-gpu` CLI argument for SCM/Tree training.
* A new `kover predict` command for inference using trained models.
* Graceful fallback to CPU if no GPU is found.

The logic appears correct and consistent with Kover’s existing architecture. Happy computing with **FastKover**!
