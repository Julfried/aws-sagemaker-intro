import sagemaker
import boto3
import sagemaker.logs
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator

# Create a SageMaker session
session = boto3.Session(region_name='eu-central-1')
sagemaker_session = sagemaker.Session(boto_session=session)

# Get Execution role
role = "arn:aws:iam::008038232967:role/service-role/SageMaker-BP2023_FHDataScientist"

# Define a PyTorch estimator
estimator = Estimator(
    role=role,
    image_uri="763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker",
    session=sagemaker_session,
    source_dir='./',
    entry_point='train.py', 
    instance_count=1,
    instance_type='ml.t3.xlarge',
    output_path='s3://ffg-bp/pytorch-mnist',
)

estimator.fit(wait=False)
print(estimator.latest_training_job)
