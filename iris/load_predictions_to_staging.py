"""
Loads the generated predictions to a staging container in Data Lake
"""
import os
import sys

from azure.storage.blob import BlobServiceClient
from azureml.core import Run
from azureml.core.keyvault import Keyvault

run = Run.get_context()
kv = Keyvault(run.experiment.workspace)
conn_string = kv.get_secret("data-lake-key")

blob = BlobServiceClient.from_connection_string(
    conn_str=conn_string
).get_blob_client(  # noqa
    "staging", "predictions/iris.csv"
)

with open(os.path.join(sys.argv[1], "scored.csv")) as f:
    blob.upload_blob(f.read(), overwrite=True)
