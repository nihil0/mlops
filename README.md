# mlops

[![Build Status](https://dev.azure.com/nihil0/MLOps/_apis/build/status/nihil0.mlops?branchName=master)](https://dev.azure.com/nihil0/MLOps/_build/latest?definitionId=3&branchName=master)

My view on how MLOps should be implemented on Microsoft Azure

## The correct steps to train a model using tags with MLOps

1. Make changes to `train.py`
2. Run the model training on local (possibly downsampled) version of the data
3. Update `pipeline.py` and any unit tests
4. When you are ready to train the model, tag the branch with `git tag -a -m "Training with new hyperparams" train-vN.N`
5. Push the branch with the `--follow-tags` flag set
6. Make a pull request. The build is set up so that the model trains but the service is not deployed. (Since `build.py` is not run for PRs)
7. Lead data scientist checks the model training logs, ensures the metrics are okay. Approve PR to trigger a build and deploy the webservice to ACI
