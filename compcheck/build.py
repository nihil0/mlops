import yaml
import os
from pathlib import Path

from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.authentication import ServicePrincipalAuthentication

try:
    import dotenv

    dotenv.load_dotenv()
except ModuleNotFoundError:
    print("python-dotenv not installed. Hope I find the right env variables")
    pass

conf_path = Path(__file__) / ".." / ".." / "conf.yaml"
with open(conf_path.resolve(), "r") as f:
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

env_file_path = Path(__file__) / ".." / ".." / "environment.yaml"
env = Environment.from_conda_specification(
    "component-condition-serving", str(env_file_path.resolve())
)
env.docker.enabled = True

score_file_path = Path(__file__) / ".." / "score.py"
inf_config = InferenceConfig(
    entry_script=str(score_file_path.resolve()), environment=env
)
model = Model(ws, name=conf["metadata"]["model_name"])

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=2,
    description="Webservice to predict non-compliant car components.",
)

svc = Model.deploy(
    workspace=ws,
    name="compcondition",
    models=[model],
    inference_config=inf_config,
    deployment_config=deployment_config,
    overwrite=True,
)

svc.wait_for_deployment(show_output=True)
