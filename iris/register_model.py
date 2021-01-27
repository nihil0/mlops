"""
This script registers a model and tags it with metadata
"""

import argparse
import os

from azureml.core import Model, Run

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str)
parser.add_argument(
    "--model-dir",
    type=str,
    help="The directory where model is stored. "
    "It is assumed that model binary is called model.rds",
)
parser.add_argument(
    "--tag",
    type=lambda x: x.split("="),
    action="append",
    help="metadata tag expressed as key='some value'",
    dest="tags",
)
args = parser.parse_args()

tag_dict = {m[0]: m[1] for m in args.tags}

run = Run.get_context()
Model.register(
    workspace=run.experiment.workspace,
    model_name=args.model_name,
    model_path=os.path.join(args.model_dir, "model.rds"),
    tags=tag_dict,
)
