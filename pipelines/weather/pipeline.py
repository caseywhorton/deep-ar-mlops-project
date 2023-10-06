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
    training_instance_type="ml.m5.large",
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
    sess = sagemaker.Session()
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    write_bucket = "cw-sagemaker-domain-2"
    write_prefix = "deep_ar"
    
    region = sess.boto_region_name
    region = 'us-east-1'
    s3_client = boto3.client("s3", region_name=region)
    sm_client = boto3.client("sagemaker", region_name=region)
    sm_runtime_client = boto3.client("sagemaker-runtime")
    
    # Fetch SageMaker execution role
    sagemaker_role = sagemaker.get_execution_role()
    preprocessing_image_uri = "536826985609.dkr.ecr.us-east-1.amazonaws.com/cw-sagemaker:latest"
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
    pipeline_name = "HumidityDeepARPipeline"
    pipeline_model_name = "humidity-deep-ar-pipeline"
    model_package_group_name = "humidity-deep-ar-model-group"
    base_job_name_prefix = "humidity-deep-ar"
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
    #####

    input_data_uri = f"s3://{write_bucket}/{write_prefix}/data/train" #ok
    
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

    inputs = []
    
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
    role = sagemaker_role,
    instance_count = 1,
    instance_type = 'ml.m5.xlarge',
    max_runtime_in_seconds = 1200,
      base_job_name = f"{base_job_prefix}/deepar-weather-preprocess",
      sagemaker_session=pipeline_session
    )

    processing_step = ProcessingStep(
    name = "WeatherForecastingPreprocessingStep",
    processor = preprocessing_processor,
    outputs = outputs,
    job_arguments = ["--split-days","24",
                     "--region", region, 
                     "--bucket", read_bucket, 
                     "--prefix", read_prefix, 
                    "--target-feature", "properties.relativeHumidity.value"], 
    code = os.path.join(BASE_DIR, "preprocess.py")
    )

    ####
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/WeatherTrain"

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
        role=sagemaker_role,
        instance_count=1,
        instance_type="ml.c4.xlarge",
        output_path=f"s3://{s3_output_path}",
        enable_sagemaker_metrics = True
    )
    
    estimator.set_hyperparameters(**hyperparameters)
    
    training_data_uri = processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
    testing_data_uri = processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri
    # Set pipeline training step
    train_step = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={
            "train": training_data_uri ,
            "test": testing_data_uri }
    )

    """
    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-weather-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="WeatherEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="WeatherEvalModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name="RegisterWeatherModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckRMSEweatherEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    """

    # Create the Pipeline with all component steps and parameters
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[process_instance_type_param, 
                    train_instance_type_param, 
                    train_instance_count_param, 
                    deploy_instance_type_param,
                    deploy_instance_count_param,
                    clarify_instance_type_param,
                    skip_check_model_bias_param,
                    register_new_baseline_model_bias_param,
                    supplied_baseline_constraints_model_bias_param,
                    skip_check_model_explainability_param,
                    register_new_baseline_model_explainability_param,
                    supplied_baseline_constraints_model_explainability_param,
                    model_approval_status_param],
        steps=[
            processing_step,
            train_step,
            #lambda_eval_step
            # evaluate the model
            #create_model_step,
            #step_transform,
            # evaluate_step
        ],
        sagemaker_session=sess    
    )
    
    return pipeline
