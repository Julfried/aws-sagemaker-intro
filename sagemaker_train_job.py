import sagemaker
import boto3
import sagemaker.logs
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base

# Create a SageMaker session
session = boto3.Session(region_name='eu-central-1')
sagemaker_session = sagemaker.Session(boto_session=session)

# Define the execution role
role = "arn:aws:iam::008038232967:role/service-role/SageMaker-BP2023_FHDataScientist"

# Define a Sagemaker estimator
estimator = Estimator(
    role=role,
    image_uri="763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker",
    session=sagemaker_session,
    dependencies=["requirements.txt", "model.py"],
    entry_point='train.py', 
    instance_count=1,
    instance_type='ml.t3.xlarge',
    output_path='s3://ffg-bp/pytorch-mnist',
)

estimator.fit(job_name=name_from_base("mnist-example"))