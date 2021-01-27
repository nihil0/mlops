#!/usr/bin/env bash

set -u

RG='rgp-show-weu-aml-databricks'

deploy() {
    RESOURCE_DIR=$1
    az group deployment create -g $RG --template-file ${RESOURCE_DIR}/template.json --parameters @${RESOURCE_DIR}/parameters.dev.json
}

deploy azureml-workspace

deploy databricks-workspace
