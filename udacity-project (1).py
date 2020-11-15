#!/usr/bin/env python
# coding: utf-8

# In[44]:


from azureml.core import Workspace, Experiment

ws = Workspace.get(name="quick-starts-ws-126376")
exp = Experiment(workspace=ws, name="udacity-project")

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()


# In[45]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException
# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

cluster_name = "amlcomp"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target {}.'.format(cluster_name))
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size="Standard_D2_V2",
                                                               max_nodes=4)

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
compute_target.wait_for_completion(show_output=True, timeout_in_minutes=20)

print("Azure Machine Learning Compute attached")

cpu_cluster_name = "cpu-cluster"

try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cpu-cluster")
except ComputeTargetException:
    print("Creating new cpu-cluster")
    
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                           min_nodes=0,
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    
cpu_cluster.wait_for_completion(show_output=True)


# In[58]:


from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform
import os

# Specify parameter sampler
### YOUR CODE HERE ###
ps = RandomParameterSampling({
        "learning_rate": normal(10, 3),
        "keep_probability": uniform(0.05, 0.1),
        "batch_size": choice(16, 32, 64, 128)
    })

# Specify a Policy
### YOUR CODE HERE ###
policy = BanditPolicy(slack_factor = 0.2, evaluation_interval=1, delay_evaluation=5)  

if "training" not in os.listdir():
    os.mkdir("./training")

# Create a SKLearn estimator for use with train.py
### YOUR CODE HERE ###
est = SKLearn(source_directory= './',entry_script='train.py',compute_target = compute_target )

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
### YOUR CODE HERE ###

hyperdrive_config = HyperDriveConfig(estimator = est,
                                hyperparameter_sampling=ps,
                                policy=policy,
                                primary_metric_name='accuracy',
                                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                max_total_runs=3,
                                max_concurrent_runs=3)


# In[59]:


# Submit your hyperdrive run to the experiment and show run details with the widget.

### YOUR CODE HERE ###
run = exp.submit(hyperdrive_config)
RunDetails(run).show()


# In[ ]:


import joblib
# Get your best run and save the model from that run.

### YOUR CODE HERE ###


# In[ ]:


from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

### YOUR CODE HERE ###


# In[42]:


from train import clean_data


# In[ ]:


from azureml.train.automl import AutoMLConfig

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task=,
    primary_metric=,
    training_data=,
    label_column_name=,
    n_cross_validations=)


# In[2]:


# Submit your automl run

### YOUR CODE HERE ###


# In[ ]:


# Retrieve and save your best automl model.

### YOUR CODE HERE ###

