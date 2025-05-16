# Kover

Kover is a Python package for machine learning (e.g., Set Covering Machine (SCM) and Classification and Regression Trees (CART)) with optional GPU acceleration. This README provides a very detailed guide covering installation, usage, testing, and troubleshooting, as well as a summary of differences from the old Kover.

## Overview

Kover is designed to provide fast, scalable machine learning (using SCM and CART learners) with an optional GPU acceleration layer. In addition to CPU-based computations (using NumPy), Kover now supports GPU acceleration (via CuPy) on Linux (NVIDIA CUDA or AMD ROCm) machines. (Note: GPU support is not available on macOS (Apple Silicon or Intel) because CuPy does not support macOS.) The package also includes a CLI (e.g. "kover test-gpu") and updated learner modules (e.g. SCM and CART) with GPU-accelerated predict functions.

## Prerequisites

- **Python:** Version 3.7 or higher.
- **NumPy:** Required for CPU computations.
- **h5py:** Required for data handling (e.g., reading HDF5 files).
- **GPU Support (Optional):** To enable GPU acceleration, you need:
  - A CUDA-capable NVIDIA GPU (or AMD GPU with ROCm) on a Linux machine.
  - The appropriate CUDA Toolkit (or ROCm) installed.
  – CuPy (a GPU-accelerated NumPy replacement) installed (see below).

**Note:** GPU support is not available on macOS (Apple Silicon or Intel) because CuPy does not support macOS.

## Installation

### Base Installation (CPU Only)

1. Clone the repository (or download the source) and navigate to the project root (e.g., `/Users/haley/kover_new`).
2. (Optional) Create a virtual environment (recommended):
   ```sh
   python3 -m venv kover_test_env
   source kover_test_env/bin/activate
   ```
3. Install the package (in development mode) along with the minimal dependencies (NumPy and h5py):
   ```sh
   pip install -e .
   ```
   (This installs the package defined in `setup.py`.)

### Optional GPU Support (Linux Only)

If you are on a Linux machine with a CUDA-capable NVIDIA GPU (or AMD GPU with ROCm), you can install CuPy as an extra dependency. (GPU support is not available on macOS.)

- **For CUDA 12.x (NVIDIA):**
  ```sh
  pip install cupy-cuda12x
  ```
- **For CUDA 11.x (NVIDIA):**
  ```sh
  pip install cupy-cuda11x
  ```
- **For ROCm (AMD):**
  ```sh
  pip install cupy-rocm-5-0
  ```

Alternatively, you can install the package with the "gpu" extra (which installs the appropriate CuPy version for your platform):
```sh
pip install -e .[gpu]
```

## Usage

### Command-Line Interface (CLI)

Kover provides a CLI (via the `kover` command) for various tasks. For example, to test GPU support (or fallback to CPU if GPU is not available), run:

```sh
kover test-gpu
```

This command checks for GPU availability (using CuPy) and runs a simple matrix multiplication test. If CuPy is not installed (or if you are on macOS), it falls back to CPU.

### Using Kover in Your Code

You can import Kover's modules (e.g., GPU utilities, learners, experiments) in your Python scripts. For example:

#### GPU Utilities (e.g. Transfer, Matrix Multiplication)

```python
from kover.utils.gpu import to_gpu, to_cpu, gpu_matrix_multiply

# Transfer a NumPy array to GPU (if available) and perform a matrix multiplication
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
a_gpu = to_gpu(a)
b_gpu = to_gpu(b)
result_gpu = gpu_matrix_multiply(a_gpu, b_gpu)
result_cpu = to_cpu(result_gpu)
print(result_cpu)
```

#### Using the Predict Function (SCM or CART)

The SCM and CART learners (located in `kover/core/kover/learning/learners/scm.py` and `kover/core/kover/learning/learners/cart.py`, respectively) each have a predict function. For example, in SCM (or CART) you can instantiate a learner (e.g. after training) and call its predict method:

```python
from kover.learning.learners.scm import SCM  # (or CART)
# (Assume you have trained a model, e.g. "model")
# Then, to predict (using GPU acceleration if available):
X_test = ...  # your test data (a NumPy array)
predictions = model.predict(X_test)
```

In addition, the experiment modules (e.g. `experiment_scm.py` and `experiment_cart.py`) also call (or wrap) these predict functions (using GPU acceleration if available).

## Testing

### Testing GPU Support

- **On a CUDA-capable Linux machine:**  
  Install CuPy (e.g., `cupy-cuda12x`) and run:
  ```sh
  kover test-gpu
  ```
  You should see output indicating that GPU support is available (e.g., "GPU available: NVIDIA GeForce RTX 3080 (10.0GB)").

- **On macOS (Apple Silicon or Intel):**  
  CuPy is not supported on macOS. Running `kover test-gpu` will always fall back to CPU (with a warning "CuPy not found. GPU support will be disabled.").

### Running Unit Tests

You can run additional unit tests (if available) using your testing framework (e.g., pytest). For example, from the project root (or the `kover/core` directory), run:
```sh
pytest
```

## Troubleshooting

### GPU Issues

- **"CuPy not found":**  
  This means CuPy is not installed (or not available on your platform). On macOS, CuPy is not supported. On Linux, ensure that you have installed the correct CuPy package (e.g., `cupy-cuda12x` for CUDA 12.x).

- **"No CUDA device available":**  
  Verify that your NVIDIA GPU is recognized (e.g., run `nvidia-smi` on Linux) and that the CUDA drivers are installed.

- **"Error checking GPU":**  
  Check your CUDA (or ROCm) installation and that CuPy is installed correctly.

### Dependency Issues

- **Missing `h5py`:**  
  If you see an error like "No module named 'h5py'", install h5py:
  ```sh
  pip install h5py
  ```

- **Python Version:**  
  Ensure that your Python version is at least 3.7 (as specified in `setup.py`).

## What's New (Differences from Old Kover)

- **GPU Acceleration:**  
  The new Kover now supports GPU acceleration (using CuPy) on Linux (NVIDIA CUDA or AMD ROCm) machines. (Note: GPU support is not available on macOS (Apple Silicon or Intel) because CuPy does not support macOS.) This allows computationally intensive operations (e.g. matrix multiplication, rule utility computation, Gini impurity calculations, and prediction) to be accelerated on a GPU.

- **CLI (Command-Line Interface):**  
  A new CLI (via the "kover" command) is provided (e.g. "kover test-gpu") to test GPU support (or fallback to CPU) and to run additional tasks.

- **Updated Learner Modules:**  
  The SCM and CART learner modules (e.g. in `kover/core/kover/learning/learners/scm.py` and `kover/core/kover/learning/learners/cart.py`) have been updated to use GPU acceleration (if available) for operations (e.g. predict, rule utility computation, Gini impurity, etc.). (In addition, the experiment modules (e.g. experiment_scm.py and experiment_cart.py) also call (or wrap) these predict functions.)

- **Fallback to CPU:**  
  If GPU support (CuPy) is not available (or on macOS), Kover gracefully falls back to CPU (using NumPy) so that your code continues to run.

- **Detailed Documentation:**  
  A very detailed README (this file) is provided covering installation (including optional GPU support), usage (CLI, GPU utilities, predict functions, etc.), testing (GPU test, unit tests), and troubleshooting (GPU issues, dependency issues).

---

**Note:** This README is intended for developers and users of the Kover package. If you have any further questions or issues, please open an issue on the project's repository. 