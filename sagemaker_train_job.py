import sagemaker
import boto3
import sagemaker.logs
from sagemaker.estimator import Estimator

# Create a SageMaker session
session = boto3.Session(region_name='eu-central-1')
sagemaker_session = sagemaker.Session(boto_session=session)

# Define the execution role
role = "arn:aws:iam::008038232967:role/service-role/SageMaker-BP2023_FHDataScientist"

# Define a Sagemaker estimator
estimator = Estimator(
    base_job_name="mnist-example",
    role=role,
    image_uri="763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker",
    session=sagemaker_session,
    dependencies=["requirements.txt", "model.py"],
    entry_point='train.py', 
    instance_count=1,
    instance_type='ml.t3.xlarge',
    output_path="s3://ffg-bp/pytorch-mnist_output",
    code_location="s3://ffg-bp/pytorch-mnist_source",
    hyperparameters={
        "batch-size": str(128),
        "lr": str(0.001),
        "epochs": str(10)
    }
)

estimator.fit()