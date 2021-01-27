"""
This script downloads a tabular dataset into a PipelineData object

This is necessary because as of 5.11.2020, the dataset feature in R's
azuremlsdk package seems to be broken
"""
import argparse

from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", type=str)
parser.add_argument("--outpath", type=str)

args = parser.parse_args()

run = Run.get_context()

dataset = run.input_datasets[args.dataset_name]
dataset.to_csv_files().download(target_path=args.outpath)
