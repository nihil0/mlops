import argparse
import os

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    print("python-dotenv not installed. Not loading .env")

from azureml.core import (
    ComputeTarget,
    Dataset,
    Datastore,
    Experiment,
    RunConfiguration,
    Workspace,
)
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import CondaDependencies, RCranPackage, RSection
from azureml.core.keyvault import Keyvault
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

# Usually, the  cluster already exists, so we just fetch
compute_target = next(
    (m for m in ComputeTarget.list(ws) if m.name == compute["name"]), None
)

# Env for use case

aml = RCranPackage()
aml.name = "azuremlsdk"
aml.version = "1.10.0"

cd = CondaDependencies.create(
    conda_packages=["pandas", "numpy", "matplotlib"],
    pip_packages=["azureml-mlflow==1.17.0", "azureml-defaults==1.17.0"],
)


rc = RunConfiguration()
rc.framework = "R"
rc.environment.r = RSection()
# rc.environment.r.cran_packages = [aml]
rc.environment.docker.enabled = True

py_rc = RunConfiguration()
py_rc.framework = "Python"
py_rc.environment.python.conda_dependencies = cd

sql_datastore = Datastore.register_azure_sql_database(
    workspace=ws,
    datastore_name="modelling_db",
    server_name="dbserver-mlops-demo",
    database_name="asq-mlops-demo",
    username=kv.get_secret("db-user"),
    password=kv.get_secret("db-pass"),
)

traindata = Dataset.Tabular.from_sql_query(
    (sql_datastore, "SELECT * FROM dbo.traindata")
)

outdata = PipelineData("outdata", datastore=ws.get_default_datastore())
download_step = PythonScriptStep(
    name="Load training data from database",
    script_name="download_dataset.py",
    arguments=["--dataset-name", "traindata", "--outpath", outdata],
    inputs=[traindata.as_named_input("traindata")],
    compute_target=compute_target,
    source_directory=".",
    outputs=[outdata],
    runconfig=py_rc,
    allow_reuse=False,
)

model_outpath = PipelineData(
    "modeldir", datastore=ws.get_default_datastore(), is_directory=True
)
train_step = RScriptStep(
    name="Train classifier",
    script_name="R/train.R",
    arguments=[outdata, model_outpath],
    inputs=[outdata],
    outputs=[model_outpath],
    compute_target=compute_target,
    source_directory=".",
    runconfig=rc,
    allow_reuse=False,
)

register_model = PythonScriptStep(
    name="Register model",
    script_name="register_model.py",
    arguments=[
        "--model-name",
        "iris-r-classifier",
        "--model-dir",
        model_outpath,
        "--tag",
        "owner='Neelabh Kashyap'",
        "--tag",
        "team='Data Science'",
        "--tag",
        "comment=prod",
    ],
    inputs=[model_outpath],
    compute_target=compute_target,
    source_directory=".",
    runconfig=py_rc,
    allow_reuse=False,
)

pipeline = Pipeline(
    workspace=ws,
    steps=[download_step, train_step, register_model],
    description="Builds R model for iris dataset",
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")

    args = parser.parse_args()

    if args.publish:
        p = pipeline.publish(
            name="iris-classifier-train-r",
            description="train a classifer on iris dataset and register model",
        )
        print(f"Published Train Pipeline ID: {p.id}")

    else:
        Experiment(ws, "fit-iris-model").submit(pipeline).wait_for_completion(
            show_output=True
        )
