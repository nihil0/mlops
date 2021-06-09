import yaml
import os
import re
import argparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    print("python-dotenv not installed. Not loading .env")
    pass

from azureml.core import (
    Datastore,
    RunConfiguration,
    Experiment,
    Workspace,
    ComputeTarget,
)

from azureml.data.data_reference import DataReference
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import RSection, RCranPackage, CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import (
    RScriptStep,
    PythonScriptStep,
)
from azureml.core.keyvault import Keyvault

from pyapacheatlas.core import AtlasEntity, AtlasProcess
from pyapacheatlas.core import PurviewClient
import pyapacheatlas.auth


conf_file = os.path.join(os.path.dirname(__file__), "conf.yaml")

with open(conf_file, "r") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    auth_config = conf["auth"]
    compute = conf["compute"]

tenant_id = auth_config["tenant_id"]
client_id = auth_config["service_principal_id"]
client_secret = os.environ["SP_SECRET"]

# Authenticate with Purview and instantiate client
auth = pyapacheatlas.auth.ServicePrincipalAuthentication(
    tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
)
client = PurviewClient(account_name="ml-purview", authentication=auth)

# Get input and output datasets 
iris_score = client.get_entity(
    guid="3f92f951-a198-4cdf-a303-39a7164484b7"
)["entities"][0]

iris_predicted = client.get_entity(
    guid="5dc733af-52bc-487a-8351-a807210c28d9"
)["entities"][0]

# Define ML pipeline as AtlasProcess and upload
my_pipeline = AtlasProcess(
    name="iris_score_pipeline",
    typeName="azureml_pipeline",
    qualified_name="https://westeurope.api.azureml.ms/pipelines/v1.0/subscriptions/d50ade7c-2587-4da8-9c63-fc828541722c/resourceGroups/rgp-show-weu-aml-databricks/providers/Microsoft.MachineLearningServices/workspaces/aml-mlops-demo/PipelineRuns/PipelineSubmit/a00b0cec-769f-4623-a795-e7b7968bb405",
    description="Iris score pipeline. This was updated last on: 8.6.2021",
    guid=-1,
    outputs=[
        AtlasEntity(
            name=iris_predicted["attributes"]["name"],
            typeName=iris_predicted["typeName"],
            qualified_name=iris_predicted["attributes"]["qualifiedName"],
            guid=iris_predicted["guid"],
        )
    ],
    inputs=[
        AtlasEntity(
            name=iris_score["attributes"]["name"],
            typeName=iris_score["typeName"],
            qualified_name=iris_score["attributes"]["qualifiedName"],
            guid=iris_score["guid"],
        )
    ]
)
client.upload_entities(my_pipeline)

# Authenticate with AzureML
auth = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret,
)

ws = Workspace.get(
    auth_config["workspace_name"],
    subscription_id=auth_config["subscription_id"],
    resource_group=auth_config["resource_group"],
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
