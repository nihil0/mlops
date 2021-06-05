import requests
import os
import dotenv
from pyapacheatlas.auth import ServicePrincipalAuthentication
from pyapacheatlas.core import PurviewClient

dotenv.load_dotenv()


# authenticate and instantiate client
tenant_id = os.environ["TENANT_ID"]
client_id = os.environ["SP_ID"]
client_secret = os.environ["SP_SECRET"]

auth = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
)

client = PurviewClient(account_name="ml-purview", authentication=auth)

import ipdb; ipdb.set_trace()