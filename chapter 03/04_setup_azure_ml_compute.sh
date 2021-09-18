#!/bin/bash
set -euo pipefail

RG=mldemo
YAML=compute.yml
WS=mldemows

COMPUTE_NAME=amldemocompute

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

# Output commands
set -x

# Create a new compute target
# docs: https://docs.microsoft.com/en-us/cli/azure/ml/compute?view=azure-cli-latest#az_ml_compute_create
az ml compute create --file ${YAML} --resource-group ${RG} --workspace-name ${WS}