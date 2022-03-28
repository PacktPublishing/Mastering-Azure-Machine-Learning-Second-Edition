from azureml.core import Environment

def get_current_env(name="user-managed-env"):
    env = Environment(name)
    env.python.user_managed_dependencies = True
    return env

