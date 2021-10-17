#!/bin/bash
set -euo pipefail

#settings
RG=mldemo
LOC=westus2
WS=mldemows

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

# Output commands
set -x

# Create a resource group
az group create -n ${RG} -l ${LOC}

# Create an ML workspace
# docs: https://docs.microsoft.com/en-us/cli/azure/ext/azure-cli-ml/ml/workspace?view=azure-cli-latest#ext-azure-cli-ml-az-ml-workspace-create
az ml workspace create -w ${WS} -g ${RG} -l ${LOC}
