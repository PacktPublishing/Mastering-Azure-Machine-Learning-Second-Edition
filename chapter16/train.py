import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

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

# Configure experiment
ws = Workspace.from_config()
exp = Experiment(workspace=ws, name='cifar10_cnn_remote')

# cluster configuration
cluster_name = "mldemocompute"
aml_cluster = get_aml_cluster(ws, name=cluster_name)

# wait until the cluster is ready
aml_cluster.wait_for_completion(show_output=True)

# define the execution script location
script = 'cifar10_cnn_remote.py'
script_folder = os.path.join(os.getcwd(), 'code')

tf_env = get_env(['tensorflow~=2.6.0'])

src = ScriptRunConfig(source_directory=script_folder, 
                      script=script, 
                      compute_target = aml_cluster, 
                      environment = tf_env)
run = exp.submit(src)
run.wait_for_completion(show_output=True)