#!/usr/bin/env bash

set -e 

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"/..

MODEL_NAME=component-cond-pred
MODEL_VERSION=3

rm -rf /tmp/azureml-models
az ml model download -i ${MODEL_NAME}:${MODEL_VERSION} -t /tmp/azureml-models/${MODEL_NAME}/${MODEL_VERSION} -g rgp-show-weu-aml-databricks -w aml-mlops-demo
az ml environment download -d /tmp/azureml-models/env -n component-condition -g rgp-show-weu-aml-databricks -w aml-mlops-demo

docker build -t component-cond-test --build-arg DEPS="$(cat /tmp/azureml-models/env/conda_dependencies.yml)" -f Dockerfile.test .
