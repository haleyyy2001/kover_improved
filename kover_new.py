#!/usr/bin/env python
"""
Kover (GPU-Extended): Learn interpretable computational phenotyping models from k-merized genomic data

This extended version demonstrates partial GPU usage. It preserves the original command-line 
structure (dataset create/info/split, learn scm/tree/predict) but uses GPU libraries in the 
"from-reads" creation step (hypothetically) and in "learn tree" if --use-gpu is specified.

-------
LICENSE: Same GPLv3 as original Kover
-------
"""

import argparse
import logging
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from tempfile import gettempdir
from pkg_resources import get_distribution
from sys import argv
from os.path import abspath, exists, join
from os import mkdir
import sys
import os

print("[DEBUG] Python executable:", sys.executable)
print("[DEBUG] PATH:", os.environ["PATH"])
sys.stdout.flush()

KOVER_DESCRIPTION = "Kover (GPU-Extended): Learn interpretable computational phenotyping models from k-merized genomic data"
VERSION = "2.1.0-gpu"


# ----------------------------------------------------------------------
# HYPOTHETICAL GPU-BASED K-MER COUNTING (EXAMPLE)
# ----------------------------------------------------------------------
def gpu_count_kmers(reads_folder_list, kmer_size, abundance_min, threads):
    """
    Hypothetical function that uses a GPU-based library or external binary 
    to count k-mers from reads. For example, you might use:
      - A custom CUDA/CuPy approach
      - An external binary like MetaHipMer with GPU support
    This is only a placeholder.
    """
    logging.debug("Starting GPU-based k-mer counting with k=%d, min_abundance=%d", kmer_size, abundance_min)
    # You would implement your real GPU logic here (or call external).
    # For demonstration, we'll just print a message:
    print("[GPU] Counting k-mers from reads using a hypothetical GPU approach ...")
    # Return some placeholder result or path
    # In reality, you'd produce a matrix or file that the rest of Kover can read.
    return "gpu_kmers_output.tsv"


# ----------------------------------------------------------------------
# DATASET CREATION TOOL
# ----------------------------------------------------------------------

class KoverDatasetCreationTool(object):
    def __init__(self):
        self.available_data_sources = ['from-tsv', 'from-contigs', 'from-reads']

    def from_tsv(self):
        from progressbar import Bar, Percentage, ProgressBar, Timer
        parser = argparse.ArgumentParser(prog="kover dataset create from-tsv",
                                         description='Creates a Kover dataset from genomic data and optionally '
                                                     'phenotypic metadata')
        parser.add_argument('--genomic-data', help='A tab-separated file containing the k-mer matrix.',
                            required=True)
        parser.add_argument('--phenotype-description', help='An informative description that is assigned to the'
                                                            ' phenotypic metadata.')
        parser.add_argument('--phenotype-metadata', help='A file containing the phenotypic metadata.')
        parser.add_argument('--output', help='The Kover dataset to be created.', required=True)
        parser.add_argument('--compression', type=int, default=4,
                            help='The gzip compression level (0 - 9). 0 means no compression (default=4).')
        parser.add_argument('-x', '--progress', action='store_true', help='Shows a progress bar for the execution.')
        parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Sets verbosity level.')
        parser.add_argument('--parallel', '--threads', type=int, default=1,
                            help='Number of parallel threads to use (CPU). Default=1.')

        if len(argv) == 4:
            argv.append("--help")
        args = parser.parse_args(argv[4:])

        if (args.phenotype_description is not None and args.phenotype_metadata is None) or \
           (args.phenotype_description is None and args.phenotype_metadata is not None):
            print("Error: The phenotype description and metadata file must be specified together.")
            exit(1)

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG,
                                format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s: %(message)s")
        logging.debug("Creating dataset from TSV...")

        if args.progress:
            progress_vars = {"current_task": None, "pbar": None}
            def progress(task_name, p):
                if task_name != progress_vars["current_task"]:
                    if progress_vars["pbar"] is not None:
                        progress_vars["pbar"].finish()
                    progress_vars["current_task"] = task_name
                    progress_vars["pbar"] = ProgressBar(widgets=['%s: ' % task_name, Percentage(), Bar(), Timer()],
                                                        maxval=1.0)
                    progress_vars["pbar"].start()
                else:
                    progress_vars["pbar"].update(p)
        else:
            progress = None

        try:
            from kover.dataset.create import from_tsv
            from_tsv(tsv_path=args.genomic_data,
                     output_path=args.output,
                     phenotype_description=args.phenotype_description,
                     phenotype_metadata_path=args.phenotype_metadata,
                     gzip=args.compression,
                     progress_callback=progress,
                     threads=args.parallel)  # hypothetical param
        except Exception as e:
            logging.error("Dataset creation from TSV failed: %s", str(e))
            print("Error: Could not create dataset from TSV. See --verbose for details.")
            exit(1)

        if args.progress:
            progress_vars["pbar"].finish()
        logging.debug("Successfully created dataset from TSV.")

    def from_contigs(self):
        parser = argparse.ArgumentParser(prog="kover dataset create from-contigs",
                                         description='Creates a Kover dataset from genomic data (contigs)')
        parser.add_argument('--genomic-data', required=True,
                            help='TSV with lines: GENOME_ID{tab}PATH to a fasta file of contigs.')
        parser.add_argument('--phenotype-description', help='Description assigned to phenotypic metadata.')
        parser.add_argument('--phenotype-metadata', help='File containing the phenotypic metadata.')
        parser.add_argument('--output', required=True, help='The Kover dataset to be created.')
        parser.add_argument('--kmer-size', default=31, help='K-mer size, max=128. (default=31).')
        parser.add_argument('--singleton-kmers', action='store_true', default=False,
                            help='Include k-mers that occur in only one genome.')
        parser.add_argument('--n-cpu', '--n-cores', default=0,
                            help='Number of CPU cores used by external tools (DSK). 0=all cores.')
        parser.add_argument('--compression', type=int, default=4,
                            help='Gzip compression level (0-9). (default=4).')
        parser.add_argument('--temp-dir', default=gettempdir(),
                            help='Directory for temporary files. (default=system temp).')
        parser.add_argument('-x', '--progress', action='store_true', help='Show progress bar.')
        parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbosity.')
        parser.add_argument('--parallel', '--threads', type=int, default=1,
                            help='Number of parallel threads (CPU). Default=1.')

        if len(argv) == 4:
            argv.append("--help")
        args = parser.parse_args(argv[4:])

        if (args.phenotype_description is not None and args.phenotype_metadata is None) or \
           (args.phenotype_description is None and args.phenotype_metadata is not None):
            print("Error: phenotype description and metadata must be specified together.")
            exit(1)

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        logging.debug("Creating dataset from contigs with possible CPU parallelization, but no GPU support here.")

        filter_option = "singleton" if not args.singleton_kmers else "nothing"

        from kover.dataset.create import from_contigs
        try:
            from_contigs(contig_list_path=args.genomic_data,
                         output_path=args.output,
                         kmer_size=args.kmer_size,
                         filter_singleton=filter_option,
                         phenotype_description=args.phenotype_description,
                         phenotype_metadata_path=args.phenotype_metadata,
                         gzip=args.compression,
                         temp_dir=args.temp_dir,
                         nb_cores=args.n_cpu,
                         verbose=args.verbose,
                         progress=args.progress,
                         threads=args.parallel)
        except Exception as e:
            logging.error("Error creating dataset from contigs: %s", e)
            print("Could not create dataset from contigs. Use --verbose for details.")
            exit(1)

        logging.debug("Successfully created dataset from contigs.")

    def from_reads(self):
        parser = argparse.ArgumentParser(prog="kover dataset create from-reads",
                                         description='Creates a Kover dataset from reads.')
        parser.add_argument('--genomic-data', required=True,
                            help='TSV lines: GENOME_ID{tab}PATH to a directory with fastq(.gz) reads.')
        parser.add_argument('--phenotype-description', help='Description assigned to phenotype.')
        parser.add_argument('--phenotype-metadata', help='File containing phenotypic metadata.')
        parser.add_argument('--output', required=True, help='The Kover dataset output file.')
        parser.add_argument('--kmer-size', default=31, help='K-mer size, max=128.')
        parser.add_argument('--kmer-min-abundance', default=1, help='Minimum abundance for k-mers. (default=1).')
        parser.add_argument('--singleton-kmers', action='store_true', default=False,
                            help='Include k-mers that occur only in one genome (disabled by default).')
        parser.add_argument('--n-cpu', '--n-cores', default=0,
                            help='Number of CPU cores for e.g. DSK (0=all).')
        parser.add_argument('--compression', type=int, default=4,
                            help='Gzip compression level (0-9). (default=4).')
        parser.add_argument('--temp-dir', default=gettempdir(),
                            help='Directory for temporary files.')
        parser.add_argument('-x', '--progress', action='store_true', help='Show progress bar.')
        parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbosity.')
        parser.add_argument('--parallel', '--threads', type=int, default=1,
                            help='Number of parallel threads (CPU). Default=1.')
        parser.add_argument('--use-gpu', action='store_true',
                            help='If set, attempt to do GPU-based k-mer counting. (Hypothetical)')

        if len(argv) == 4:
            argv.append("--help")
        args = parser.parse_args(argv[4:])

        if (args.phenotype_description is not None and args.phenotype_metadata is None) or \
           (args.phenotype_description is None and args.phenotype_metadata is not None):
            print("Error: The phenotype description and metadata must be specified together.")
            exit(1)

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        logging.debug("Creating dataset from reads...")

        from kover.dataset.create import from_reads
        filter_option = "singleton" if not args.singleton_kmers else "nothing"

        # Here's the GPU-based demonstration (completely hypothetical).
        if args.use_gpu:
            # Hypothetical GPU approach to produce a matrix or file:
            gpu_output_tsv = gpu_count_kmers(reads_folder_list=args.genomic_data,
                                             kmer_size=int(args.kmer_size),
                                             abundance_min=int(args.kmer_min_abundance),
                                             threads=int(args.parallel))
            # Then pass that file to "from_tsv" internally or wrap it
            logging.debug("Now using the generated GPU-based tsv: %s", gpu_output_tsv)
            # (One might skip 'from_reads' and just do 'from_tsv' if you have the matrix. 
            #  But let's assume from_reads can accept precomputed partial results.)

            # For demonstration, we pass it on to 'from_reads' but in reality you'd either:
            #   - unify from_reads to detect the GPU TSV 
            #   - or store the matrix in a known location
            # The rest is CPU-based for now.

        try:
            from_reads(reads_folders_list_path=args.genomic_data,
                       output_path=args.output,
                       kmer_size=args.kmer_size,
                       abundance_min=args.kmer_min_abundance,
                       filter_singleton=filter_option,
                       phenotype_description=args.phenotype_description,
                       phenotype_metadata_path=args.phenotype_metadata,
                       gzip=args.compression,
                       temp_dir=args.temp_dir,
                       nb_cores=args.n_cpu,
                       verbose=args.verbose,
                       progress=args.progress,
                       threads=args.parallel)
        except Exception as e:
            logging.error("Error creating dataset from reads: %s", e)
            print("Could not create dataset from reads. Use --verbose for details.")
            exit(1)

        logging.debug("Successfully created dataset from reads (GPU partial).")

# ----------------------------------------------------------------------
# DATASET TOOL
# ----------------------------------------------------------------------

class KoverDatasetTool(object):
    def __init__(self):
        self.available_commands = ['create', 'info', 'split']

    def create(self):
        creation_tool = KoverDatasetCreationTool()

        parser = argparse.ArgumentParser(usage=
        '''%(prog)s dataset create <data source> [<args>]
The two available data sources are:
from-tsv      Create a dataset from a k-mer matrix TSV
from-contigs  Create a dataset from contigs
from-reads    Create a dataset from reads (optionally GPU-based)
''')
        parser.add_argument('datasource', choices=creation_tool.available_data_sources,
                            help='Type of genomic data (from-tsv, from-contigs, from-reads)')
        if len(argv) == 3:
            argv.append("--help")
        args = parser.parse_args(argv[3:4])
        getattr(creation_tool, args.datasource.replace("-", "_"))()

    def info(self):
        parser = argparse.ArgumentParser(prog="kover dataset info",
                                         description='Prints information about a dataset.')
        parser.add_argument('--dataset', required=True,
                            help='The Kover dataset for which info is requested.')
        parser.add_argument('--all', action='store_true', help='Print all info.')
        parser.add_argument('--genome-type', action='store_true', help='Print genome type.')
        parser.add_argument('--genome-source', action='store_true', help='Print genome source path.')
        parser.add_argument('--genome-ids', action='store_true', help='Print genome identifiers.')
        parser.add_argument('--genome-count', action='store_true', help='Print number of genomes.')
        parser.add_argument('--kmers', action='store_true', help='Print k-mer sequences (fasta).')
        parser.add_argument('--kmer-len', action='store_true', help='Print length of k-mers.')
        parser.add_argument('--kmer-count', action='store_true', help='Print number of k-mers.')
        parser.add_argument('--phenotype-description', action='store_true', help='Print phenotype description.')
        parser.add_argument('--phenotype-metadata', action='store_true', help='Print phenotype metadata source.')
        parser.add_argument('--phenotype-tags', action='store_true', help='Print phenotype tags.')
        parser.add_argument('--splits', action='store_true', help='Print available splits.')
        parser.add_argument('--uuid', action='store_true', help='Print dataset unique identifier.')
        parser.add_argument('--compression', action='store_true', help='Print compression info.')
        parser.add_argument('--classification-type', action='store_true', help='Print classification type.')

        if len(argv) == 3:
            argv.append("--help")
        args = parser.parse_args(argv[3:])

        from kover.dataset import KoverDataset
        dataset = KoverDataset(args.dataset)

        if args.genome_type or args.all:
            print("Genome type:", dataset.genome_source_type)
            print()
        if args.genome_source or args.all:
            print("Genome source:", dataset.genome_source)
            print()
        if args.genome_ids or args.all:
            print("Genome IDs:")
            for id_ in dataset.genome_identifiers:
                print(id_)
            print()
        if args.genome_count or args.all:
            print("Genome count:", dataset.genome_count)
            print()
        if args.kmers or args.all:
            print("Kmer sequences (fasta):")
            for i, k in enumerate(dataset.kmer_sequences):
                print(">k%d" % (i + 1))
                print(k)
            print()
        if args.kmer_len or args.all:
            print("K-mer length:", dataset.kmer_length)
            print()
        if args.kmer_count or args.all:
            print("K-mer count:", dataset.kmer_count)
            print()
        if args.phenotype_description or args.all:
            print("Phenotype description:", dataset.phenotype.description)
            print()
        if args.phenotype_metadata or args.all:
            if dataset.phenotype.description != "NA":
                print("Phenotype metadata source:", dataset.phenotype.metadata_source)
            else:
                print("No phenotype metadata.")
            print()
        if args.phenotype_tags or args.all:
            print("Phenotype tags:", ", ".join(dataset.phenotype.tags))
            print()
        if args.compression or args.all:
            print("Compression:", dataset.compression)
            print()
        if args.classification_type or args.all:
            print("Classification type:", dataset.classification_type)
            print()
        if args.splits or args.all:
            splits = dataset.splits
            if len(splits) > 0:
                print("Splits available for learning:")
                for s in splits:
                    print(s)
            else:
                print("No splits available for learning.")

    def split(self):
        parser = argparse.ArgumentParser(prog="kover dataset split",
                                         description='Split dataset file into train/test and optionally folds.')
        parser.add_argument('--dataset', required=True, help='Kover dataset to be split.')
        parser.add_argument('--id', required=True, help='Identifier for the resulting split.')
        parser.add_argument('--train-size', type=float, default=0.5,
                            help='Proportion for training (default=0.5). Alternatively specify --train-ids/--test-ids.')
        parser.add_argument('--train-ids', help='File with genome IDs for training.')
        parser.add_argument('--test-ids', help='File with genome IDs for testing.')
        parser.add_argument('--folds', type=int, default=0,
                            help='Number of k-fold cross-validation folds to create. 0=none.')
        parser.add_argument('--random-seed', type=int, help='Random seed (if not provided, random).')
        parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbosity.')
        parser.add_argument('-x', '--progress', action='store_true', help='Show progress bar.')

        if len(argv) == 3:
            argv.append("--help")
        args = parser.parse_args(argv[3:])

        if args.folds == 1:
            print("Error: folds must be 0 or >= 2.")
            exit(1)

        if (args.train_ids and not args.test_ids) or (args.test_ids and not args.train_ids):
            print("Error: must specify both train-ids and test-ids or neither.")
            exit(1)

        from kover.dataset.split import split_with_ids, split_with_proportion
        from progressbar import Bar, Percentage, ProgressBar, Timer
        from random import randint

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)

        if args.random_seed is None:
            args.random_seed = randint(0, 4294967295)

        if args.progress:
            progress_vars = {"current_task": None, "pbar": None}
            def progress(task_name, p):
                if task_name != progress_vars["current_task"]:
                    if progress_vars["pbar"] is not None:
                        progress_vars["pbar"].finish()
                    progress_vars["current_task"] = task_name
                    progress_vars["pbar"] = ProgressBar(
                        widgets=['%s: ' % task_name, Percentage(), Bar(), Timer()],
                        maxval=1.0)
                    progress_vars["pbar"].start()
                else:
                    progress_vars["pbar"].update(p)
        else:
            progress = None

        if args.train_ids and args.test_ids:
            split_with_ids(input=args.dataset,
                           split_name=args.id,
                           train_ids_file=args.train_ids,
                           test_ids_file=args.test_ids,
                           random_seed=args.random_seed,
                           n_folds=args.folds,
                           progress_callback=progress)
        else:
            split_with_proportion(input=args.dataset,
                                  split_name=args.id,
                                  train_prop=args.train_size,
                                  random_seed=args.random_seed,
                                  n_folds=args.folds,
                                  progress_callback=progress)
        if args.progress:
            progress_vars["pbar"].finish()


# ----------------------------------------------------------------------
# LEARNING TOOL
# ----------------------------------------------------------------------

class KoverLearningTool(object):
    def __init__(self):
        # We add 'predict' from our prior extended version
        self.available_commands = ['scm', 'tree', 'predict']

    def scm(self):
        """
        In this example, we do NOT attempt to rewrite the SCM code for GPU.
        SCM remains CPU-based.
        """
        parser = argparse.ArgumentParser(prog='kover learn scm',
                                         description='Learn a conjunction/disjunction model (SCM). [CPU-based]')
        parser.add_argument('--dataset', required=True)
        parser.add_argument('--split', required=True)
        parser.add_argument('--model-type', choices=['conjunction','disjunction'], nargs='+',
                            default=['conjunction','disjunction'])
        parser.add_argument('--p', type=float, nargs='+',
                            default=[0.1,0.316,0.562,1.0,1.778,3.162,10.0,999999.0])
        parser.add_argument('--kmer-blacklist', help='List of k-mers to remove (fasta or txt).', required=False)
        parser.add_argument('--max-rules', type=int, default=10)
        parser.add_argument('--max-equiv-rules', type=int, default=10000)
        parser.add_argument('--hp-choice', choices=['bound','cv','none'], default='cv')
        parser.add_argument('--bound-max-genome-size', type=int)
        parser.add_argument('--random-seed', type=int)
        parser.add_argument('--n-cpu', '--n-cores', type=int, default=1)
        parser.add_argument('--output-dir', default='.')
        parser.add_argument('-x','--progress', action='store_true')
        parser.add_argument('-v','--verbose', action='store_true', default=False)
        parser.add_argument('--authorized-rules', type=str, default="", help=argparse.SUPPRESS)

        if len(argv) == 3:
            argv.append("--help")
        args = parser.parse_args(argv[3:])

        import logging
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        from datetime import timedelta
        from time import time
        from json import dump as json_dump
        from kover.dataset import KoverDataset
        from kover.learning.experiments.experiment_scm import learn_SCM
        from progressbar import Bar, Percentage, ProgressBar, Timer

        pre_dataset = KoverDataset(args.dataset)
        dataset_kmer_count = pre_dataset.kmer_count
        classification_type = pre_dataset.classification_type
        phenotype_tags = pre_dataset.phenotype.tags[...]
        if classification_type != "binary":
            print("Error: SCM only supports binary classification.")
            exit(1)
        try:
            pre_dataset.get_split(args.split)
        except:
            print(f"Error: The split ({args.split}) does not exist. Use 'kover dataset split'.")
            exit(1)
        if args.hp_choice == "cv" and len(pre_dataset.get_split(args.split).folds) < 2:
            print("Error: The split must have >=2 folds for cross-validation.")
            exit(1)
        del pre_dataset

        if args.progress:
            progress_vars = {"current_task": None, "pbar": None}
            def progress(task_name, p):
                if task_name != progress_vars["current_task"]:
                    if progress_vars["pbar"] is not None:
                        progress_vars["pbar"].finish()
                    progress_vars["current_task"] = task_name
                    progress_vars["pbar"] = ProgressBar(widgets=['%s: ' % task_name, Percentage(), Bar(), Timer()],
                                                        maxval=1.0)
                    progress_vars["pbar"].start()
                else:
                    progress_vars["pbar"].update(p)
        else:
            progress = None

        if args.bound_max_genome_size is None:
            args.bound_max_genome_size = dataset_kmer_count
        args.bound_delta = 0.05

        start = time()
        bhp, bhp_score, train_metrics, test_metrics, model, rule_imps, eq_rules, classifications = learn_SCM(
            dataset_file=args.dataset,
            split_name=args.split,
            model_type=args.model_type,
            p=args.p,
            kmer_blacklist_file=args.kmer_blacklist,
            max_rules=args.max_rules,
            max_equiv_rules=args.max_equiv_rules,
            bound_delta=args.bound_delta,
            bound_max_genome_size=args.bound_max_genome_size,
            parameter_selection=args.hp_choice,
            n_cpu=args.n_cpu,
            random_seed=args.random_seed,
            authorized_rules=args.authorized_rules,
            progress_callback=progress
        )
        running_time = timedelta(seconds=time()-start)
        if args.progress:
            progress_vars["pbar"].finish()

        # Construct and save a report (same as original code)...

        from kover.dataset import KoverDataset
        ds = KoverDataset(args.dataset)
        sp = ds.get_split(args.split)
        phenotype_tags = ds.phenotype.tags[...]
        report = f"Kover SCM GPU-Extended (but SCM itself is CPU)\n{'='*32}\n"
        report += f"\nRunning time: {running_time}\n"
        # Add your usual details, etc....

        print(report)

        if not exists(args.output_dir):
            mkdir(args.output_dir)
        with open(join(args.output_dir, "report.txt"), "w") as f:
            f.write(report)

        # Save JSON results
        results = {
            # ...
        }
        with open(join(args.output_dir, 'results.json'), 'w') as f:
            json_dump(results, f)

    def tree(self):
        """
        This subcommand demonstrates partial GPU usage via RAPIDS cuML's DecisionTreeClassifier.
        If --use-gpu is specified, we attempt to import cuML.
        Otherwise, we fall back to the original CPU-based approach.
        """
        parser = argparse.ArgumentParser(prog='kover learn tree (GPU-Extended)',
                                         description='Learn a decision tree model (CPU or GPU if --use-gpu).')
        parser.add_argument('--dataset', required=True)
        parser.add_argument('--split', required=True)
        parser.add_argument('--criterion', type=str, nargs='+',
                            choices=['gini','crossentropy'], default='gini', required=False)
        parser.add_argument('--max-depth', type=int, nargs='+', default=10, required=False)
        parser.add_argument('--min-samples-split', type=int, nargs='+', default=2, required=False)
        parser.add_argument('--class-importance', type=str, nargs='+', default=None, required=False)
        parser.add_argument('--kmer-blacklist', required=False)
        parser.add_argument('--hp-choice', choices=['bound','cv'], default='cv')
        parser.add_argument('--bound-max-genome-size', type=int)
        parser.add_argument('--n-cpu','--n-cores', type=int, default=1)
        parser.add_argument('--output-dir', default='.')
        parser.add_argument('-x','--progress', action='store_true')
        parser.add_argument('-v','--verbose', action='store_true', default=False)
        parser.add_argument('--authorized-rules', type=str, default="", help=argparse.SUPPRESS)
        parser.add_argument('--use-gpu', action='store_true',
                            help='If set, use RAPIDS cuML DecisionTree for GPU-based training.')

        if len(argv) == 3:
            argv.append("--help")
        args = parser.parse_args(argv[3:])

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        from datetime import timedelta
        from time import time
        from json import dump as json_dump
        from kover.dataset import KoverDataset
        from progressbar import Bar, Percentage, ProgressBar, Timer

        # 1) Load dataset
        ds = KoverDataset(args.dataset)
        try:
            ds.get_split(args.split)
        except:
            print(f"Error: The split '{args.split}' does not exist. Use 'kover dataset split'.")
            exit(1)
        if args.hp-choice == "cv" and len(ds.get_split(args.split).folds) < 2:
            print("Error: for cv, split must have >=2 folds.")
            exit(1)

        dataset_kmer_count = ds.kmer_matrix.shape[1]
        classification_type = ds.classification_type
        if classification_type not in ["binary","multiclass"]:
            print("Error: unexpected classification type.")
            exit(1)

        # 2) Possibly parse class importances, etc. (omitted for brevity)...

        if args.progress:
            progress_vars = {"current_task": None, "pbar": None}
            def progress(task_name, p):
                if task_name != progress_vars["current_task"]:
                    if progress_vars["pbar"] is not None:
                        progress_vars["pbar"].finish()
                    progress_vars["current_task"] = task_name
                    progress_vars["pbar"] = ProgressBar(
                        widgets=['%s: ' % task_name, Percentage(), Bar(), Timer()],
                        maxval=1.0)
                    progress_vars["pbar"].start()
                else:
                    progress_vars["pbar"].update(p)
        else:
            progress = None

        if args.bound_max_genome_size is None:
            args.bound_max_genome_size = dataset_kmer_count

        # 3) If --use-gpu, call our GPU-based approach with cuML
        use_gpu = args.use_gpu
        start = time()

        if use_gpu:
            # We do a simplified approach: we won't do hyperparam search or interpretability at the same depth.
            # We'll just show how you'd load data, train on GPU, get predictions.
            # In a real scenario, you'd replicate the multi-hp approach from "experiment_cart" in Kover code.

            try:
                import cudf
                from cuml.tree import DecisionTreeClassifier
            except ImportError:
                print("Error: cuML not installed or not found. Re-run with --use-gpu removed or install RAPIDS.")
                exit(1)

            split_obj = ds.get_split(args.split)
            X_train = ds.kmer_matrix[split_obj.train_genome_idx,:]
            y_train = ds.phenotype.metadata[split_obj.train_genome_idx]
            X_test = ds.kmer_matrix[split_obj.test_genome_idx,:]
            y_test = ds.phenotype.metadata[split_obj.test_genome_idx]

            # Convert to GPU-friendly DataFrames
            gX_train = cudf.DataFrame(X_train)
            gy_train = cudf.Series(y_train)
            gX_test  = cudf.DataFrame(X_test)
            gy_test  = cudf.Series(y_test)

            # Create and fit the GPU-based model
            model = DecisionTreeClassifier(
                max_depth=max(args.max_depth) if isinstance(args.max_depth, list) else args.max_depth,
                split_criterion='gini' if 'gini' in args.criterion else 'entropy',
                min_samples_split=min(args.min_samples_split) if isinstance(args.min_samples_split,list) else args.min_samples_split
                # more parameters as needed
            )
            model.fit(gX_train, gy_train)
            preds = model.predict(gX_test)
            preds = preds.to_array()  # Move to CPU memory

            # Evaluate or do partial metrics (placeholder)
            test_accuracy = float((preds == y_test).sum()) / len(y_test) if len(y_test)>0 else None

            running_time = timedelta(seconds=(time()-start))
            # Build a mini report
            report = "Kover Tree with GPU (cuML) Example\n"
            report += f"Test accuracy: {test_accuracy}\n"
            report += f"Running time: {running_time}\n"
            print(report)

            # Save a minimal output
            if not exists(args.output_dir):
                mkdir(args.output_dir)
            with open(join(args.output_dir, "report.txt"), "w") as f:
                f.write(report)

            # In real Kover, you'd store model rules, interpretability data, etc.
            # This is left out for brevity.

        else:
            # Fall back to the CPU-based original CART
            from kover.learning.experiments.experiment_cart import learn_CART
            best_hp, best_hp_score, train_metrics, test_metrics, model, rule_importances, eq_rules, classifications = \
                learn_CART(dataset_file=args.dataset,
                           split_name=args.split,
                           criterion=args.criterion,
                           max_depth=args.max_depth,
                           min_samples_split=args.min_samples_split,
                           class_importance=None,  # for brevity
                           bound_delta=0.05,
                           bound_max_genome_size=args.bound_max_genome_size,
                           kmer_blacklist_file=args.kmer_blacklist,
                           parameter_selection=args.hp_choice,
                           authorized_rules=args.authorized_rules,
                           n_cpu=args.n_cpu,
                           progress_callback=progress)
            running_time = timedelta(seconds=(time()-start))

            # Construct a minimal text output or replicate the original code's reporting...
            print("Kover Tree Training (CPU) finished.")
            print(f"Running time: {running_time}")
            # etc....
            if not exists(args.output_dir):
                mkdir(args.output_dir)
            with open(join(args.output_dir, "report.txt"), "w") as f:
                f.write("CPU-based CART training done.\n")

        if args.progress:
            progress_vars["pbar"].finish()

    def predict(self):
        """
        Subcommand to apply a previously-trained model to a new dataset.
        We'll keep it CPU-based for simplicity. 
        If your model was GPU-based, you'd have to load it similarly 
        (e.g., re-instantiate a cuML model from saved state).
        """
        parser = argparse.ArgumentParser(prog='kover learn predict',
                                         description='Apply an existing model to a dataset split.')
        parser.add_argument('--model-dir', required=True)
        parser.add_argument('--dataset', required=True)
        parser.add_argument('--split', required=True)
        parser.add_argument('--output-file', default='predictions.tsv')
        parser.add_argument('-v','--verbose',action='store_true',default=False)
        parser.add_argument('-x','--progress',action='store_true')
        if len(argv) == 3:
            argv.append("--help")
        args = parser.parse_args(argv[3:])

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        from kover.dataset import KoverDataset
        ds = KoverDataset(args.dataset)
        try:
            sp = ds.get_split(args.split)
        except:
            print(f"Error: Split {args.split} not found in dataset.")
            exit(1)

        # Real code: load model from model-dir, e.g. parse JSON, check if SCM or CART, or if GPU was used, etc.
        # Then do predictions, write to predictions file.

        print(f"[PREDICT] Loaded model from {args.model_dir}. Using dataset {args.dataset} (split '{args.split}').")
        print(f"[PREDICT] Output will be at: {args.output_file}")

# ----------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# ----------------------------------------------------------------------

class CommandLineInterface(object):
    def __init__(self):
        self.available_commands = ['dataset','learn']

        parser = argparse.ArgumentParser(description=KOVER_DESCRIPTION)
        parser.add_argument('--cite', action='store_true')
        parser.add_argument('--license', action='store_true')
        parser.add_argument('--version', action='store_true')
        parser.add_argument('command', choices=self.available_commands, help='Main command: dataset or learn')

        if len(argv) == 1:
            argv.append("--help")

        # If first argument is an option like --cite, --license, etc.
        if argv[1].startswith("--"):
            args = parser.parse_args([argv[1], "learn"])
            if args.license:
                print(
"""Kover (GPU-Extended): GPLv3 License
...
See original license text for details.
""")
            elif args.version:
                print(f"cli-{VERSION}")
                try:
                    print(f"core-{get_distribution('kover').version}")
                except:
                    print("core version unknown (kover not installed via pip).")
            elif args.cite:
                print("""Cite Kover:
@article{Drouin2019, ... }
@article{Drouin2016, ... }
""")
            else:
                # Possibly user did --help or something
                args = parser.parse_args(argv[1:2])
                if not hasattr(self, args.command):
                    print(f"kover: '{args.command}' is not recognized. See '{argv[0]} --help'.")
                    parser.print_help()
                    exit(1)
                getattr(self, args.command)()

        else:
            # The user specified dataset or learn
            args = parser.parse_args(argv[1:2])
            if not hasattr(self, args.command):
                print(f"kover: '{args.command}' not recognized. See '{argv[0]} --help'.")
                parser.print_help()
                exit(1)
            getattr(self, args.command)()

    def dataset(self):
        dataset_tool = KoverDatasetTool()
        parser = argparse.ArgumentParser(usage=
        '''%(prog)s dataset <command> [<args>]
create    Create Kover datasets
split     Split a dataset
info      Show dataset info
''')
        parser.add_argument('command', choices=dataset_tool.available_commands)
        if len(argv) == 2:
            argv.append("--help")
        args = parser.parse_args(argv[2:3])
        getattr(dataset_tool, args.command)()

    def learn(self):
        learning_tool = KoverLearningTool()
        parser = argparse.ArgumentParser(usage=
        '''%(prog)s learn <experiment> [<args>]
scm      SCM model (CPU)
tree     Decision Tree model (CPU or GPU via --use-gpu)
predict  Apply model to a new dataset
''')
        parser.add_argument('command', choices=learning_tool.available_commands)
        if len(argv) == 2:
            argv.append("--help")
        args = parser.parse_args(argv[2:3])
        getattr(learning_tool, args.command)()


if __name__ == '__main__':
    CommandLineInterface()
