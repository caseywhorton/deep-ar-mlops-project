"""
Implements a get_pipeline(**kwargs) method.
"""
##### Import Packages #####
import pandas as pd
import json
import boto3
import pathlib
import io
import sagemaker

from sagemaker.deserializers import CSVDeserializer
from sagemaker.serializers import CSVSerializer

from sagemaker.processing import (
    ProcessingInput, 
    ProcessingOutput, 
    ScriptProcessor
)
from sagemaker.inputs import TrainingInput

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, 
    TrainingStep, 
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.parameters import (
    ParameterInteger, 
    ParameterFloat, 
    ParameterString, 
    ParameterBoolean
)
from sagemaker.workflow.clarify_check_step import (
    ModelBiasCheckConfig, 
    ClarifyCheckStep, 
    ModelExplainabilityCheckConfig
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.lambda_helper import Lambda

from sagemaker.model_metrics import (
    MetricsSource, 
    ModelMetrics, 
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines

from sagemaker.image_uris import retrieve
import os
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig

from sagemaker.transformer import Transformer

from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.transformer import Transformer
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep

##### Define functions #####

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

# This is the pipeline definition 
def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="WeatherPackageGroup",
    pipeline_name="WeatherPipeline",
    base_job_prefix="Weather",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    #####
    # Instantiate AWS services session and client objects
    print('***** BASE_DIR', BASE_DIR)
    print('***** Get Sagemaker Session *****')
    sess = sagemaker.Session()
    sagemaker_session = get_session(region, default_bucket)
    print('***** Get Role *****')
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    print('**** Role *****')
    print('role:',role)
    print('***** Set run parameters *****')
    write_bucket = "cw-sagemaker-domain-2"
    write_prefix = "deep_ar"
    
    region = sess.boto_region_name
    region = 'us-east-1'
    s3_client = boto3.client("s3", region_name=region)
    sm_client = boto3.client("sagemaker", region_name=region)
    sm_runtime_client = boto3.client("sagemaker-runtime")
    
    # Fetch SageMaker execution role
    #sagemaker_role = sagemaker.get_execution_role()
    preprocessing_image_uri = "536826985609.dkr.ecr.us-east-1.amazonaws.com/cw-sagemaker"
    training_image_uri = retrieve("forecasting-deepar", region)
    
    # S3 locations used for parameterizing the notebook run
    read_bucket = "cw-sagemaker-domain-1"
    read_prefix = "deep_ar/data/raw" 
    
    # S3 location where raw data to be fetched from
    raw_data_key = f"s3://{read_bucket}/{read_prefix}"
    
    # S3 location where processed data to be uploaded
    processed_data_key = f"{write_prefix}/data/processed"
    
    # S3 location where train data to be uploaded
    train_data_key = f"{write_prefix}/data/train"
    
    # S3 location where validation data to be uploaded
    validation_data_key = f"{write_prefix}/data/validation"
    
    # S3 location where test data to be uploaded
    test_data_key = f"{write_prefix}/data/test"
    
    # Full S3 paths
    weather_data_uri = f"{raw_data_key}/*.csv" #ok
    output_data_uri = f"s3://{write_bucket}/{write_prefix}/data/" #ok
    scripts_uri = f"s3://{write_bucket}/{write_prefix}/scripts" #ok
    estimator_output_uri = f"s3://{write_bucket}/{write_prefix}/training_jobs"
    processing_output_uri = f"s3://{write_bucket}/{write_prefix}/processing_jobs"
    model_eval_output_uri = f"s3://{write_bucket}/{write_prefix}/model_eval"
    clarify_bias_config_output_uri = f"s3://{write_bucket}/{write_prefix}/model_monitor/bias_config"
    clarify_explainability_config_output_uri = f"s3://{write_bucket}/{write_prefix}/model_monitor/explainability_config"
    bias_report_output_uri = f"s3://{write_bucket}/{write_prefix}/clarify_output/pipeline/bias"
    explainability_report_output_uri = f"s3://{write_bucket}/{write_prefix}/clarify_output/pipeline/explainability"

    # Set names of pipeline objects
    experiment_name = "weather-forecast-model"
    pipeline_name = "weather-forecast-model-pipeline"
    pipeline_model_name = "weather-forecast-model"
    model_package_group_name = "weather-forecast-model-group"
    base_job_name_prefix = "weather-forecast"
    endpoint_config_name = f"{pipeline_model_name}-endpoint-config"
    endpoint_name = f"{pipeline_model_name}-endpoint"
    
    # Set data parameters
    target_col = "target"
    
    # Set instance types and counts
    process_instance_type = "ml.m5.large"
    process_instance_type = "ml.c5.xlarge"
    train_instance_count = 1
    train_instance_type = "ml.m4.xlarge"
    predictor_instance_count = 1
    predictor_instance_type = "ml.m4.xlarge"
    clarify_instance_count = 1
    clarify_instance_type = "ml.m4.xlarge"


    # Set up pipeline input parameters

    # Set processing instance type
    print('***** Set instance types *****')
   # Set up pipeline input parameters

    # Set processing instance type
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value=process_instance_type,
    )
    
    # Set training instance type
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=train_instance_type,
    )
    
    # Set training instance count
    train_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=train_instance_count
    )
    
    # Set deployment instance type
    deploy_instance_type_param = ParameterString(
        name="DeployInstanceType",
        default_value=predictor_instance_type,
    )
    
    # Set deployment instance count
    deploy_instance_count_param = ParameterInteger(
        name="DeployInstanceCount",
        default_value=predictor_instance_count
    )
    
    # Set Clarify check instance type
    clarify_instance_type_param = ParameterString(
        name="ClarifyInstanceType",
        default_value=clarify_instance_type,
    )
    
    # Set model bias check params
    skip_check_model_bias_param = ParameterBoolean(
        name="SkipModelBiasCheck", 
        default_value=False
    )
    
    register_new_baseline_model_bias_param = ParameterBoolean(
        name="RegisterNewModelBiasBaseline",
        default_value=False
    )
    
    supplied_baseline_constraints_model_bias_param = ParameterString(
        name="ModelBiasSuppliedBaselineConstraints", 
        default_value=""
    )
    
    # Set model explainability check params
    skip_check_model_explainability_param = ParameterBoolean(
        name="SkipModelExplainabilityCheck", 
        default_value=False
    )
    
    register_new_baseline_model_explainability_param = ParameterBoolean(
        name="RegisterNewModelExplainabilityBaseline",
        default_value=False
    )
    
    supplied_baseline_constraints_model_explainability_param = ParameterString(
        name="ModelExplainabilitySuppliedBaselineConstraints", 
        default_value=""
    )
    
    # Set model approval param
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )
    
    input_data_uri = f"s3://{write_bucket}/{write_prefix}/data/train" #ok
    print('***** Get pipeline session *****')
    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount",
                                                 default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=input_data_uri,
    )

    # processing step for feature engineering

    #inputs = []
    print('***** Preprocessing *****')
    outputs = [
    ProcessingOutput(
        source = "/opt/ml/processing/output/full_data",
        destination = f"{output_data_uri}full_data",
        output_name = "full_data"
    ),
    ProcessingOutput(
        source = "/opt/ml/processing/output/train",
        destination = f"{output_data_uri}train",
        output_name = "train_data"
    ),
    ProcessingOutput(
        source = "/opt/ml/processing/output/test",
        destination = f"{output_data_uri}test",
        output_name = "test_data"
    )
    ]

    preprocessing_processor = ScriptProcessor(
    command = ['python3'],
    image_uri = preprocessing_image_uri,
    role = role,
    instance_count = 1,
    instance_type = 'ml.m5.xlarge',
    max_runtime_in_seconds = 1200,
      base_job_name = f"{base_job_prefix}/deepar-weather-preprocess",
      sagemaker_session=pipeline_session
    )

    processing_step = ProcessingStep(
    name = "DataPreprocessingStep",
    processor = preprocessing_processor,
    outputs = outputs,
    job_arguments = ["--split-days","24",
                     "--region", region, 
                     "--bucket", read_bucket, 
                     "--prefix", read_prefix, 
                    "--target-feature", "properties.relativeHumidity.value"], 
        code = BASE_DIR + '/preprocess.py'
   # code = os.path.join(BASE_DIR, "preprocess.py")
    )

    ####
    

    print('***** Training *****')
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/WeatherTrain"
    print('model_path: ',model_path)
    freq = "H"
    prediction_length = 24
    context_length = 72
    
    hyperparameters = {
        "time_freq": freq,
        "context_length": str(context_length),
        "prediction_length": str(prediction_length),
        "num_cells": "40",
        "num_layers": "5",
        "likelihood": "gaussian",
        "epochs": "10",
        "mini_batch_size": "36",
        "learning_rate": "0.001",
        "dropout_rate": "0.05",
        "early_stopping_patience": "10",
    }
    
    constants = {
    "bucket": "cw-sagemaker-domain-1",
    "key_prefix_train": "deep_ar/data/train",
    "key_prefix_test": "deep_ar/data/test",
    "key_prefix_raw": "deep_ar/data/raw/",
        "image_name": "forecasting-deepar",
        "region": "us-east-1"
    }
    
    s3_output_path = "cw-sagemaker-domain-2/deep_ar/output/"
    
    # set deepar estimator
    sagemaker_session = sagemaker.Session()
    
    estimator = sagemaker.estimator.Estimator(
        sagemaker_session=sagemaker_session,
        image_uri=training_image_uri,
        hyperparameters = hyperparameters,
        role=role,
        instance_count=1,
        instance_type="ml.c4.xlarge",
        output_path=f"s3://{s3_output_path}",
        enable_sagemaker_metrics = True
    )
    print('**** Set Hyperparameters for Algorithm *****')
    estimator.set_hyperparameters(**hyperparameters)
    
    training_data_uri = processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri.to_string()
    testing_data_uri = processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri.to_string()
    # Set pipeline training step
    train_step = TrainingStep(
        name="ModelTrainingStep",
        estimator=estimator,
        inputs={
            "train": training_data_uri ,
            "test": testing_data_uri }
    )
    # Model Creation
    # Create a SageMaker model
    model = sagemaker.model.Model(
    image_uri=training_image_uri,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=sagemaker_session,
    role=role
    )

    print('*** Create a model for the Batch Transform ***')
    # Create a SageMaker model
    model = sagemaker.model.Model(
        image_uri=training_image_uri,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role
    )
    
    # Specify model deployment instance type
    inputs = sagemaker.inputs.CreateModelInput(instance_type=deploy_instance_type_param)
    
    create_model_step = CreateModelStep(name=pipeline_model_name, model=model, inputs=inputs)

    # Batch Transform
    print('*** Batch transform Step ***')
    output_path="s3://cw-sagemaker-domain-2/deep_ar/data/predictions/"
    
    model_name = create_model_step.properties.ModelName
    
    transformer = Transformer(model_name = model_name,
                              instance_count = 1,
                              instance_type = 'ml.m5.2xlarge',
                              output_path = output_path,
                              base_transform_job_name = "deepar-batch-transform-124",
                              sagemaker_session=sagemaker_session)
    input_data = sagemaker.inputs.TransformInput(data = "s3://cw-sagemaker-domain-2/deep_ar/data/test/test.json")

    step_batch_transform = TransformStep(
        name="GetForecastsForEvaluationStep",
        transformer = transformer,
        inputs = input_data
    )

    # Evaluate Model Predictions
    output_path="s3://cw-sagemaker-domain-2/deep_ar/data/"
    base_job_prefix = "Weather"
    
    script_eval = ScriptProcessor(
            image_uri=preprocessing_image_uri,
            command=["python3"],
            instance_type='ml.m5.xlarge',
            instance_count=1,
            base_job_name=f"{base_job_prefix}/script-weather-eval",
            sagemaker_session=sagemaker_session,
            role=role,
    )
    
    
    inputs=[ProcessingInput(
        source=step_batch_transform.properties.TransformInput.DataSource.S3DataSource.S3Uri,
        destination="/opt/ml/processing/actuals"),
            ProcessingInput(#source=output_path + "test.json.out",
                        source=step_batch_transform.properties.TransformOutput.S3OutputPath,
                        destination="/opt/ml/processing/predictions",
            )
           ]
    
    
    outputs = [
    ProcessingOutput(
        source = "/opt/ml/processing/evaluation",
        destination = f"{output_path}evaluation",
        output_name = "evaluation")
    ]
    code=BASE_DIR + "/evaluate.py"

    evaluation_report = PropertyFile(
    name="WeatherForecastEvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
    )
    
    evaluation_step = ProcessingStep(
        name="EvaluateTrainedModelStep",
        #step_args=step_args,
        processor = script_eval,
        inputs = inputs,
        outputs = outputs,
        property_files=[evaluation_report],
        code = code
    
    )
    
    print('***** Creating pipeline from steps. *****')
    # Create the Pipeline with all component steps and parameters
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[process_instance_type_param, 
                    train_instance_type_param, 
                    train_instance_count_param]
        ,
        pipeline_experiment_config=PipelineExperimentConfig(
          experiment_name,
          ExecutionVariables.PIPELINE_EXECUTION_ID
        ),
        steps=[
            processing_step,
            #train_step,
            #create_model_step,
            #step_batch_transform,
            #evaluation_step
        ],
        sagemaker_session=sess
        
    )
    
    # Create a new or update existing Pipeline
    pipeline.upsert(role_arn=role)
    
    # Full Pipeline description
    pipeline_definition = json.loads(pipeline.describe()['PipelineDefinition'])
    pipeline_definition
    return pipeline
