import argparse
import os
import re

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    print("python-dotenv not installed. Not loading .env")

from azureml.core import (
    ComputeTarget,
    Datastore,
    Experiment,
    RunConfiguration,
    Workspace,
)
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import CondaDependencies, RCranPackage, RSection
from azureml.core.keyvault import Keyvault
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep, RScriptStep

conf_file = os.path.join(os.path.dirname(__file__), "conf.yaml")

with open(conf_file, "r") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    auth_config = conf["auth"]
    compute = conf["compute"]

# Authenticate with AzureML
auth = ServicePrincipalAuthentication(
    tenant_id=auth_config["tenant_id"],
    service_principal_id=auth_config["service_principal_id"],
    service_principal_password=os.environ["SP_SECRET"],
)

ws = Workspace(
    subscription_id=auth_config["subscription_id"],
    resource_group=auth_config["resource_group"],
    workspace_name=auth_config["workspace_name"],
    auth=auth,
)

kv = Keyvault(ws)

# Usually, the  computes already exist, so we just fetch
compute_target = next(
    (m for m in ComputeTarget.list(ws) if m.name == compute["name"]), None
)

# Env for use case

aml = RCranPackage()
aml.name = "azuremlsdk"
aml.version = "1.10.0"

cd = CondaDependencies.create(
    conda_packages=["pandas", "numpy", "matplotlib"],
    pip_packages=[
        "azureml-mlflow==1.17.0",
        "azureml-defaults==1.17.0",
        "azure-storage-blob",
    ],
)


rc = RunConfiguration()
rc.framework = "R"
rc.environment.r = RSection()
# rc.environment.r.cran_packages = [aml]
rc.environment.docker.enabled = True

py_rc = RunConfiguration()
py_rc.framework = "Python"
py_rc.environment.python.conda_dependencies = cd

connstring = kv.get_secret("data-lake-key")
account_key = re.search("AccountKey=([-A-Za-z0-9+/=]+);", connstring).group(1)

datalake = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name="top_secret_data_lake",
    container_name="data",
    account_name="topsecretdata",
    account_key=account_key,
)


trained_model_dir = PipelineData(
    "trained_model", datastore=ws.get_default_datastore(), is_directory=True
)
download_model = PythonScriptStep(
    name="Download model from model repository",
    script_name="download_model.py",
    arguments=[
        "--model-name",
        "iris-r-classifier",
        "--model-dir",
        trained_model_dir,
    ],  # noqa
    outputs=[trained_model_dir],
    compute_target=compute_target,
    source_directory=".",
    runconfig=py_rc,
    allow_reuse=False,
)

predictions = PipelineData(
    name="predictions",
    datastore=ws.get_default_datastore(),
    output_path_on_compute="/tmp/scored.csv",
    output_mode="upload",
)
scoredata = DataReference(
    datastore=datalake,
    data_reference_name="scoredata",
    path_on_datastore="iris-score.csv",
)
inference_step = RScriptStep(
    name="Score new data",
    script_name="R/score.R",
    arguments=[trained_model_dir, scoredata],
    inputs=[trained_model_dir, scoredata],
    outputs=[predictions],
    compute_target=compute_target,
    source_directory=".",
    runconfig=rc,
    allow_reuse=False,
)

load_staging = PythonScriptStep(
    name="Load staging container",
    script_name="load_predictions_to_staging.py",
    arguments=[predictions.as_download()],
    inputs=[predictions],
    compute_target=compute_target,
    runconfig=py_rc,
    allow_reuse=False,
)

pipeline = Pipeline(
    workspace=ws,
    steps=[download_model, inference_step, load_staging],
    description="Scores Iris classifier against new iris dataset",
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")

    args = parser.parse_args()

    if args.publish:
        p = pipeline.publish(
            name="iris-classifier-score-r",
            description="Score iris classifer on new dataset",
        )
        print(f"Published Score Pipeline ID: {p.id}")

    else:
        Experiment(ws, "score-iris-model").submit(pipeline).wait_for_completion(  # noqa
            show_output=True
        )
