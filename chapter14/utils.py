from azureml.core.runconfig import RunConfiguration
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import PublishedPipeline

def get_aml_cluster(ws, cluster_name, vm_size='STANDARD_D2_V2', max_nodes=4):
    try:
        cluster = ComputeTarget(workspace=ws, name=cluster_name)
    except ComputeTargetException:
        config = AmlCompute.provisioning_configuration(vm_size=vm_size, max_nodes=max_nodes)
        cluster = ComputeTarget.create(ws, cluster_name, config)
    return cluster

def get_env(name="tf", packages=None):
    packages = packages or []
    packages += ['azureml-defaults']
    environment = Environment(name=name)
    environment.python.conda_dependencies = CondaDependencies.create(pip_packages=packages)
    return environment

def get_run_config(packages=None):
    run_config = RunConfiguration()
    run_config.environment = get_env(name="tf", packages=packages)
    return run_config

def get_pipeline(workspace, pipeline_id):
    for pipeline in PublishedPipeline.list(workspace):
        if pipeline.id == pipeline_id:
            return pipeline
    return None
