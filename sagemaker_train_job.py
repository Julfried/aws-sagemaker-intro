import sagemaker
import boto3
import sagemaker.logs
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator
import logging

# Config logger
# Setup both SageMaker and Botocore loggers
sagemaker_logger = logging.getLogger("sagemaker")
botocore_logger = logging.getLogger("botocore")
sagemaker_logger.setLevel(logging.INFO)
botocore_logger.setLevel(logging.INFO)

# Add stream handler to see logs in console
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
sagemaker_logger.addHandler(handler)
botocore_logger.addHandler(handler)

# Create a SageMaker session
session = boto3.Session(region_name='eu-central-1')
sagemaker_session = sagemaker.Session(boto_session=session)

# Get Execution role
role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)

# Define a PyTorch estimator
estimator = Estimator(
    role=role,
    image_uri="763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker",
    session=sagemaker_session,
    dependencies=['requirements.txt'],
    source_dir='./',
    entry_point='train.py', 
    instance_count=1,
    instance_type='ml.t3.2xlarge',
    output_path='s3://ffg-bp/pytorch-mnist',
)

estimator.fit(inputs='s3://ffg-bp/test', job_name='pytorch-mnist')
