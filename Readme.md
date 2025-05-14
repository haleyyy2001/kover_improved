# Kover - Improved Version 2.0.1

**Kover: Learn interpretable computational phenotyping models from k-merized genomic data**

## Overview

This is an improved version of Kover, a tool for learning interpretable computational phenotyping models from k-merized genomic data. Kover implements machine learning algorithms designed specifically for genomic data, allowing you to build interpretable classification models that predict phenotypes (like antibiotic resistance) based on the presence or absence of k-mers (DNA substrings of length k).

## What's New in This Version

### Major Improvements

1. **Parallelization Support**
   - Added `--parallel`/`--threads` parameter to dataset creation commands to enable multi-threading
   - Significantly speeds up processing of large genomic datasets
   - Available for all three data source types: `from-tsv`, `from-contigs`, and `from-reads`

2. **New "Predict" Functionality**
   - Added a new `predict` subcommand to the `learn` tool
   - Allows applying pre-trained models to new datasets without re-training
   - Useful for applying existing models to new isolates or validation sets

3. **Enhanced Error Handling**
   - More detailed error messages
   - Better error catching with try/except blocks
   - Provides useful suggestions when errors occur

4. **Code Modernization**
   - Updated from Python 2 to Python 3 syntax
   - Better input validation
   - More consistent parameter handling

### Minor Improvements

- Additional debugging output when verbose mode is enabled
- More informative progress logging
- Clearer documentation in help text and error messages
- Better command-line interface with more consistent argument handling

## Using the Improved Kover

### Installation

Installation instructions remain the same as the original Kover.

### Basic Usage

Kover has three main command groups:

```bash
kover dataset    # For creating and managing datasets
kover learn      # For learning predictive models
kover --help     # Display help information
```

### Creating Datasets

The dataset creation commands now support parallelization:

```bash
# Create a dataset from TSV with parallelization
kover dataset create from-tsv --genomic-data data.tsv --output output.kover --parallel 4 --progress

# Create a dataset from contigs with parallelization
kover dataset create from-contigs --genomic-data list.txt --output output.kover --kmer-size 31 --parallel 4 --progress

# Create a dataset from reads with parallelization
kover dataset create from-reads --genomic-data list.txt --output output.kover --kmer-size 31 --parallel 4 --progress
```

### Learning Models

The learning process remains largely the same but with better error handling:

```bash
# Create a dataset split
kover dataset split --dataset my_data.kover --id my_split --train-size 0.8 --folds 5

# Train an SCM model
kover learn scm --dataset my_data.kover --split my_split --output-dir results_scm

# Train a decision tree model
kover learn tree --dataset my_data.kover --split my_split --output-dir results_tree
```

### New: Using Predict Functionality

The new predict command allows you to apply a trained model to a new dataset:

```bash
# Apply a pre-trained model to a new dataset
kover learn predict --model-dir results_scm --dataset new_data.kover --split test_split --output-file predictions.tsv
```

## Example Workflow

1. **Create a dataset**
   ```bash
   kover dataset create from-contigs --genomic-data genome_list.txt --phenotype-description "Antibiotic resistance" --phenotype-metadata metadata.txt --output resistant.kover --parallel 8 --progress
   ```

2. **Split the dataset for training/testing**
   ```bash
   kover dataset split --dataset resistant.kover --id split_80_20 --train-size 0.8 --folds 5
   ```

3. **Train an SCM model**
   ```bash
   kover learn scm --dataset resistant.kover --split split_80_20 --output-dir model_output
   ```

4. **Apply the model to new data**
   ```bash
   kover learn predict --model-dir model_output --dataset new_genomes.kover --split test --output-file predictions.tsv
   ```

## Tips for Best Performance

- Use the `--parallel` parameter with the number of CPU cores available for faster processing
- Enable progress bars with `--progress` for better monitoring of long-running operations
- Use `--verbose` to get detailed information about the execution
- When dealing with large datasets, consider tuning the compression parameter (`--compression`) for a balance between file size and speed

## Known Issues

- The current prediction implementation is a placeholder - users should implement the actual model loading and prediction functionality based on their specific needs

## References

If you use Kover in your work, please cite the papers mentioned in the `--cite` option.
