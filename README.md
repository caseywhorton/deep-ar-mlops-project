# deep-ar-mlops-project

This repository contains code for an AWS Sagemaker pipeline that automates the training of a time-series forecasting model. The algorithm used is the DeepAR algorithm, and the use-case is described in the [electric-weather](https://github.com/caseywhorton/electric-weather) repository.

This README is separated into sections:
+ [AWS Services and Tools](aws_services_and_tools)
+ [Directory Structure](directory_structure)

# AWS Services and Tools

+ AWS Cloudwatch: Watches for errors and communicates to user.
+ AWS Elastic Container Registry (ECR)
+ AWS Sagemaker: Pipelines for Machine Learning
+ AWS Identity and Access Manager (IAM)
+ AWS Code pipeline: CI/CD services

## AWS Sagemaker

Sagemaker is used for model definition, training, versioning and monitoring. Within Sagemaker, there are several tools used in this project, including:

+ [Sagemaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html)
+ [Sagemaker Studio](https://aws.amazon.com/sagemaker/studio/)
+ [Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)

Amazon SageMaker Pipelines is a machine learning (ML) model development and deployment workflow orchestration tool within Amazon SageMaker. It provides a managed platform for creating, managing, and automating end-to-end ML workflows, from data preparation and model training to deployment and monitoring.

SageMaker Pipelines allow data scientists and ML engineers to define, automate, and scale ML workflows using a visual interface or code, making it easier to manage the complex lifecycle of ML models. These pipelines offer versioned and reusable components, enabling collaboration, governance, and reproducibility in machine learning projects. They help streamline the development process by providing a consistent and structured approach to building, testing, and deploying ML models.

# Directory Structure
```
|-- codebuild-buildspec.yml
|-- CONTRIBUTING.md
|-- pipelines
|   |-- weather
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

Your codebuild execution instructions. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CICD system (via CodePipeline). You will see that this file has the fields definined for naming the Pipeline, ModelPackageGroup etc. You can customize them as required.

```
|-- codebuild-buildspec.yml
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.


<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```

## Usage
### Pipeline Workflow
CodePipeline monitors the GitHub repository for changes. Upon detecting a change, CodePipeline triggers the pipeline.  The defined Sagemaker pipeline is ran in sagemaker on the instances and instance types configured for each step in the pipeline.

### Committing Changes
When committing changes to this repository:

Make modifications to the necessary files, such as files in the _pipelines_ directory.
Use Git commands (git add ., git commit -m "Your commit message", git push origin main) to push changes to your repository.
Ensure meaningful commit messages that describe the changes made for better tracking.
