# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.


library(jsonlite)
library(azuremlsdk)

# Load workspace
ws <- load_workspace_from_config()

# Load default datastore
ds <- get_default_datastore(ws)

# Upload iris data to the datastore
target_path <- "iris_data"
upload_files_to_datastore(ds,
                          list("./iris.csv"),
                          target_path = target_path,
                          overwrite = TRUE)

# Create AmlCompute cluster
cluster_name <- "cpu-cluster-1"
compute_target <- get_compute(ws, cluster_name = cluster_name)
if (is.null(compute_target)) {
  vm_size <- "STANDARD_D2_V2"
  compute_target <- create_aml_compute(workspace = ws,
                                       cluster_name = cluster_name,
                                       vm_size = vm_size,
                                       max_nodes = 1)
  
  wait_for_provisioning_completion(compute_target, show_output = TRUE)
}

# Create environment
r_env <- r_environment(name = "r_env")

# Define estimator
est <- estimator(source_directory = ".",
                 entry_script = "./train.R",
                 script_params = list("--data_folder" = ds$path(target_path)),
                 compute_target = compute_target,
                 environment = r_env,
                 )

experiment_name <- "train-r-script-on-amlcompute"
exp <- experiment(ws, experiment_name)

# Submit job and display the run details
run <- submit_experiment(exp, est)
plot_run_details(run)
wait_for_run_completion(run, show_output = TRUE)

# Get the run metrics
metrics <- get_run_metrics(run)
metrics

#Get Trained Model
download_files_from_run(run, prefix="outputs/")
iris_model <- readRDS("./outputs/model_trained.rds")
summary(iris_model)

#Register Model to azureml
model <- register_model(ws, 
                        model_path = "outputs/model_trained.rds", 
                        model_name = "iris_model_trained",
                        description = "Predict class of the Iris flower")

# Deploy Model


model <- get_model(ws,
                   id="iris_model:2"
                  )



# Create inference config
inference_config <- inference_config(
  entry_script = "score.R",
  source_directory = ".",
  environment = r_env)

# Create ACI deployment config
deployment_config <- aci_webservice_deployment_config(cpu_cores = 1,
                                                      memory_gb = 1)

# Deploy the web service
service_name <- paste0('aciservice-', sample(1:100, 1, replace=TRUE))
service <- deploy_model(ws, 
                        service_name, 
                        list(model), 
                        inference_config, 
                        deployment_config)
wait_for_deployment(service, show_output = TRUE)

# Delete cluster
#delete_compute(compute_target)
