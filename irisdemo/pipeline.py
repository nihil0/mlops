from azureml.pipeline.steps import PythonScriptStep, RScriptStep
from azureml.pipeline.core import Pipeline, PipelineData

from azureml.core import Workspace

from azureml.core.runconfig import RunConfiguration
from azureml.core.environment import RSection

from azureml.data.data_reference import DataReference
from azureml.core.authentication import ServicePrincipalAuthentication

from azureml.core import Dataset
from azureml.pipeline.core import PipelineData
from azureml.core.datastore import Datastore
from azureml.pipeline.core import PipelineData
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment


import os
import yaml

def main():
    # Replace this with the Use of Service Principal for authenticating with the Workspace
    ws = Workspace.from_config()

    # Choose a name for your CPU cluster
    cpu_cluster_name = "cpu-cluster"

    # Verify that cluster does not exist already
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                                max_nodes=4)
        cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)

    # Run configuration for R
    rc = RunConfiguration()
    rc.framework = "R"

    # Run configuration for python
    py_rc = RunConfiguration()
    py_rc.framework = "Python"
    py_rc.environment.docker.enabled = True

    # Combine GitHub and Cran packages for R env
    rc.environment.r = RSection()
    rc.environment.docker.enabled = True

    # Upload iris data to the datastore
    # target_path = "iris_data"
    # upload_files_to_datastore(ds,
    #                         list("./iris.csv"),
    #                         target_path = target_path,
    #                         overwrite = TRUE)



    training_data = DataReference(
        datastore=ws.get_default_datastore(),
        data_reference_name="iris_data",
        path_on_datastore="iris_data/iris.csv",
    )
    
    print('Succesfull')
    print(training_data)

    # PipelineData object for newly trained model
    trained_model_dir = PipelineData(
        name="trained_model", datastore=ws.get_default_datastore(), is_directory=True
    )

    # Training and Deployment of model
    train_step = RScriptStep(
        script_name="train-on-amlcompute.R",
        arguments=[training_data, trained_model_dir],
        inputs=[training_data],
        outputs=[trained_model_dir],
        compute_target="cpu-cluster",
        source_directory=".",
        runconfig=rc,
        allow_reuse=True,
    )

    

    print("Step Train created")

    steps = [train_step]

    train_pipeline = Pipeline(workspace=ws, steps=steps)
    train_pipeline.validate()
    pipeline_run = Experiment(ws, 'iris_training').submit(train_pipeline)
    pipeline_run.wait_for_completion(show_output=True)

    published_pipeline = train_pipeline.publish(
        name="iris-train",
        description="Model training/retraining pipeline",
    )
    print(f"Published pipeline: {published_pipeline.name}")
    print(f"for build {published_pipeline.version}")

if __name__ == "__main__":
    main()


