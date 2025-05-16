FastKover  
---
> **Note:** This project is an **edited version** of [aldro61/kover](https://github.com/aldro61/kover). We have added optional GPU support via PyTorch, extended CLI commands (including `kover predict`), and other performance enhancements.
> The original repository is still actively maintained by [aldro61](https://github.com/aldro61) and [gletarte](https://github.com/gletarte). Please see [aldro61/kover](https://github.com/aldro61/kover) for the unmodified code, original documentation, and official releases.
 

FastKover (a fork of **Kover**) is an *out-of-core* implementation of rule-based machine learning algorithms, **modified** to add **GPU-based training** and other improvements while preserving the original code’s ability to learn interpretable, k-mer-based models for genomic biomarker discovery.

---

## Table of Contents

1. [Introduction](#introduction)
2. [What’s New in FastKover](#whats-new-in-fastkover)
3. [Acknowledgment of Original Repository](#acknowledgment-of-original-repository)
4. [Installation](#installation)

   1. [Prerequisites](#prerequisites)
   2. [Installing via Docker](#installing-via-docker)
   3. [Manual Installation](#manual-installation)
   4. [Optional GPU Dependencies](#optional-gpu-dependencies)
5. [Usage](#usage)

   1. [Dataset Creation](#dataset-creation)
   2. [Dataset Splitting](#dataset-splitting)
   3. [Learning Models](#learning-models)
   4. [Prediction](#prediction)
6. [Tutorials & Documentation](#tutorials--documentation)
7. [References](#references)
8. [License](#license)
9. [Contact](#contact)

---

## Introduction

FastKover aims to bridge the gap between **high interpretability** and **high scalability** in genomic data analysis. It builds on **Kover**’s rule-based framework — grounded in **sample compression theory** — which yields compact, comprehensible models from **k-merized** genomic data. Now, with **GPU acceleration**, you can train on large datasets more rapidly.

Examples of use cases include:

* **Antimicrobial resistance** prediction (modeling genotype-to-phenotype associations).
* **Biomarker discovery** across many species.
* Any setting requiring interpretable classification of sequences.

For more details on the underlying theory and approach, see these publications:

> **Drouin, A. et al.** (2019). *Interpretable genotype-to-phenotype classifiers with performance guarantees.*
> Scientific Reports, **9**(1), 4071.
> [\[PDF\]](https://www.nature.com/articles/s41598-019-40561-2)

> **Drouin, A. et al.** (2016). *Predictive computational phenotyping and biomarker discovery using reference-free genome comparisons.*
> BMC Genomics, **17**(1), 754.
> [\[PDF\]](http://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2889-6)

### Video Lecture

The Set Covering Machine (SCM) implementation in Kover was highlighted in this Google tech talk:

[![Google tech talk](https://img.youtube.com/vi/uomMdBdEwnk/0.jpg)](https://www.youtube.com/watch?v=uomMdBdEwnk)

---

## What’s New in FastKover

1. **GPU Support**:

   * Training (`kover learn scm/tree`) can leverage PyTorch tensors on CUDA/ROCm devices.
   * Command-line flags (e.g., `--use-gpu`) let you enable or disable GPU usage.
   * Fallback to CPU if a GPU is not available.

2. **New Commands**:

   * `kover predict`: Apply a trained model (SCM or tree) to a new dataset for inference.

3. **Enhanced Performance**:

   * Out-of-core approach still used for large k-mer data.
   * Optional CuPy integration for certain array operations.

4. **Backward Compatibility**:

   * Dataset and output formats remain the same.
   * Original Dockerfile updated for GPU environments.

---

## Acknowledgment of Original Repository

This code **originates** from the open-source repository [aldro61/kover](https://github.com/aldro61/kover).
The core Kover approach and much of the code (dataset creation, SCM, and decision tree logic) are by **A. Drouin & G. Letarte** et al. We **credit** their work and have **extended** it with additional GPU-related features and a few CLI enhancements.

---

## Installation

### Prerequisites

You must install or have available on your system:

* **CMake**
* **GNU C++ compiler (g++)**
* **GNU Fortran (gfortran)**
* **HDF5 library**
* **Python 2.7.x** (original Kover is Python 2–only; adapt if needed)
* **Python dev headers** (e.g., `python-dev` on Ubuntu)
* **NumPy** & **SciPy** (ideally installed manually for performance)

Python packages like **Cython**, **h5py**, **pandas**, **progressbar** will typically be installed automatically by `pip`.

### Installing via Docker

We provide a Docker image (forked from [aldro61/kover](https://hub.docker.com/r/aldro61/kover)) with GPU-ready modifications:

```bash
docker pull aldro61/kover
docker run -it --rm -v /path/to/data:/data aldro61/kover /bin/bash
```

Inside the container, you can run:

```bash
kover --version
kover dataset ...
kover learn ...
```

To use a GPU, ensure you have NVIDIA Docker (or similar) set up, then:

```bash
docker run --gpus all -it --rm aldro61/kover /bin/bash
```

### Manual Installation

1. **Clone** this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/kover-fast.git
   cd kover-fast
   ```

2. **Install** via the provided script:

   ```bash
   ./install.sh
   ```

   This compiles Kover (FastKover) and installs all required dependencies. A `bin/` directory with the `kover` executable is created.

3. (Optional) **Add** it to your `$PATH`:

   ```bash
   export PATH=$(pwd)/bin:$PATH
   ```

4. Test:

   ```bash
   kover --version
   ```

   You should see something like `cli-2.0.x` or similar.

### Optional GPU Dependencies

* **PyTorch**: Required for GPU training (see [pytorch.org](https://pytorch.org/) for the appropriate wheel).
* **CuPy**: Further acceleration for some numeric operations. Install one of:

  * `cupy-cuda12x`
  * `cupy-cuda11x`
  * `cupy-rocm-5-0`

If you do not need GPU acceleration, you can skip these.

---

## Usage

FastKover’s CLI follows the same structure as original Kover:

1. **`kover dataset`** – create and manipulate datasets.
2. **`kover learn`** – train SCM or decision trees.
3. **`kover predict`** – run inference on new datasets.

### Dataset Creation

```bash
kover dataset create from-tsv \
  --genomic-data my_data.tsv \
  --phenotype-description "My phenotype" \
  --phenotype-metadata phenotype.txt \
  --output my_dataset.h5
```

Likewise `from-contigs` or `from-reads` subcommands are available.

### Dataset Splitting

```bash
kover dataset split \
  --dataset my_dataset.h5 \
  --id my_split \
  --train-size 0.7 \
  --folds 5
```

### Learning Models

#### SCM Example

```bash
kover learn scm \
  --dataset my_dataset.h5 \
  --split my_split \
  --model-type conjunction disjunction \
  --max-rules 10 \
  --hp-choice cv \
  --use-gpu \
  --output-dir scm_results
```

* `--use-gpu` attempts to utilize a GPU via PyTorch.

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
  --output-dir tree_results
```

### Prediction

```bash
kover predict \
  --model-dir scm_results \
  --dataset new_data.h5 \
  --output-dir predictions \
  --use-gpu
```

Saves predictions in `predictions/predictions.json`.

---

## Tutorials & Documentation

* The **original** Kover docs remain largely relevant: [http://aldro61.github.io/kover/](http://aldro61.github.io/kover/)
  (Data formats, usage flow, examples.)
* Additional **GPU usage tips** can be found in this fork’s repository wiki (if applicable).

---

## References

If you use this project in your work, please cite the original Kover references:

1. **Drouin, A., Letarte, G., Raymond, F., Marchand, M., Corbeil, J., & Laviolette, F.** (2019).
   *Interpretable genotype-to-phenotype classifiers with performance guarantees.*
   Scientific Reports, 9(1), 4071.
   [\[PDF\]](https://www.nature.com/articles/s41598-019-40561-2)

2. **Drouin, A., Giguère, S., Déraspe, M., Marchand, M., Tyers, M., Loo, V. G., Bourgault, A. M., Laviolette, F. & Corbeil, J.** (2016).
   *Predictive computational phenotyping and biomarker discovery using reference-free genome comparisons.*
   BMC Genomics, 17(1), 754.
   [\[PDF\]](http://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2889-6)

---

## License

This project is licensed under **GPL-3.0**, the same as the original Kover:

```
Kover: Learn interpretable computational phenotyping models from k-merized genomic data
Copyright (C) 2018 …
This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3
or (at your option) any later version.
```

---

## Contact

* **Usage Questions**: Post on [Biostars](https://www.biostars.org/p/194520/) or open a discussion.
* **Issues / Bugs**: Please file a GitHub Issue in this fork’s repository.
* **Original Authors**: See [https://github.com/aldro61/kover/graphs/contributors](https://github.com/aldro61/kover/graphs/contributors) for the original code’s maintainers.

Thank you for using **FastKover** (our edited, GPU-extended Kover fork)!
