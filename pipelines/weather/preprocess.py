import argparse
import pathlib
import boto3
import os
import logging
import json
import pandas as pd
import numpy as np
import urllib.parse
import datetime
from typing import List, Dict
from datetime import datetime, timezone, timedelta, date

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def columnNameReformat(column_name: str, target_name: str) -> str:
    """Reformats a column name for easier use in applications."""
    if column_name == target_name:
        return "target"
    else:
        column_name = column_name.lower()
        column_name = column_name.replace("properties", "")
        column_name = column_name.replace(".", "_")
        column_name = column_name.strip()
        # return column_name
        return "target"


def preprocessQuant(feature) -> np.array:
    """Processes a quantiative feature for input to an ML algorithm"""
    # scaler = StandardScaler()
    x = np.array(feature)
    np.nan_to_num(x, copy=False)
    # x_tran = scaler.fit_transform(x.reshape(-1,1))
    return x


def featureDict(start: datetime, feature_name: str, feature_data: List) -> Dict:
    """Creates a python Dict object for a feature with a given start date."""

    return dict(
        {
            "start": start,
            feature_name: feature_data.reshape(1, len(feature_data))[0].tolist(),
        }
    )


def getStart(df: pd.DataFrame) -> str:
    """Gets the start date for the input data from the dataframe."""
    # sort the dataframe by the measurement time
    return df.index[0]


def getStartString(df: pd.DataFrame) -> str:
    """Gets the start date in a string format"""
    stp = datetime.strptime(df["properties.timestamp"][0], "%Y-%m-%dT%H:%M:%S%z")
    return datetime.strftime(stp, "%Y-%m-%d %H:%M:%S")


def preprocessDataFrame(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the dataframe."""

    stp = df["properties.timestamp"].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z")
    )

    # set the index
    df.index = stp

    # sort the dataframe by the index
    df.sort_index(inplace=True)

    # rename index
    df.index.name = "timestamp"

    return df


def write_dicts_to_file(path, data):
    with open(path, "wb") as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode("utf-8"))


def series_to_obj(ts, feature_name=None, cat=None):
    obj = {"start": str(ts.index[0]), feature_name: list(ts)}
    if cat is not None:
        obj["cat"] = cat
    return obj


def series_to_jsonline(ts, feature_name=None, cat=None):
    return json.dumps(series_to_obj(ts, feature_name, cat))


def dict_to_series(time_dict: dict) -> pd.Series:
    """Translates a dictionary to a time series using a pandas series data type."""
    time_index = pd.date_range(
        start=time_dict["start"], periods=len(list(time_dict.values())[1]), freq="H"
    )
    time_index = [x.strftime("%Y-%m-%d %H:%M:%S") for x in time_index]

    return pd.Series(data=list(time_dict.values())[1], index=time_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-days", type=int, default=30)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--target-feature", type=str)
    args, _ = parser.parse_known_args()
    logger.info("Received arguments {}".format(args))

    # Get arguments passed in job_arguments
    split_days = args.split_days
    region = args.region
    bucket = args.bucket
    prefix = args.prefix
    target_feature = args.target_feature

    # Set local path prefix in the processing container
    local_dir = "/opt/ml/processing"
    encoding = "utf-8"
    os.environ["AWS_DEFAULT_REGION"] = region
    
    logger.debug("Getting CSV data from S3 and Cleaning")

    # Get S3 raw data location
    s3 = boto3.client("s3")

    # get raw data into container
    today = datetime.now(timezone.utc)
    lag_365 = datetime.now(timezone.utc) + timedelta(days=-365)
    print('bucket:',bucket)
    print('prefix:',prefix)
    objects = s3.list_objects(Bucket=bucket, Prefix=prefix)

    df_list = []
    for o in objects["Contents"]:
        if o["LastModified"] <= today and o["LastModified"] >= lag_365:
            obj = s3.get_object(Bucket=bucket, Key=o["Key"])
            df_list.append(pd.read_csv(obj["Body"]))

    # drop the duplicates from the dataframe
    preprocessed_df = pd.concat(df_list).drop_duplicates().reset_index()

    # process the features in the columns of the dataframe
    start = getStart(preprocessed_df)
    start_str = getStartString(preprocessed_df)
    features = ["properties.temperature.value","properties.relativeHumidity.value"]

    # preprocess the feature data
    mylist = list()

    for feature in features:
        mylist.append(
            featureDict(
                start_str,
                columnNameReformat(feature, target_feature),
                # feature,
                preprocessQuant(preprocessed_df[feature]),
            )
        )

    # create the individual time series for each feature
    time_series = []
    for i in mylist:
        time_series.append(dict_to_series(i))

    # Split into train and test data
    logger.info("Splitting data into train, validation, and test sets")

    # Create testing data
    time_series_test = []
    for ts in time_series:
        time_series_test.append(ts[-24:])

    test_features_output_path = os.path.join(
        "/opt/ml/processing/output/test", "test.json"
    )

    logger.info("Saving test features to {}".format(test_features_output_path))

    i = 0
    with open(test_features_output_path, "wb") as f:
        for ts in time_series:
            f.write(
                series_to_jsonline(ts, feature_name=list(mylist[i].keys())[1]).encode(
                    "utf-8"
                )
            )
            f.write("\n".encode(encoding))
            i += 1

    logger.info(f"Testing data saved to {test_features_output_path}")

    # Create training data
    time_series_training = []
    for ts in time_series:
        time_series_training.append(ts[:-24])

    train_features_output_path = os.path.join(
        "/opt/ml/processing/output/train", "train.json"
    )

    logger.info("Saving train features to {}".format(train_features_output_path))

    i = 0
    with open(train_features_output_path, "wb") as f:
        for ts in time_series:
            f.write(
                series_to_jsonline(ts, feature_name=list(mylist[i].keys())[1]).encode(
                    "utf-8"
                )
            )
            f.write("\n".encode(encoding))
            i += 1

    logger.info(f"Training data saved to {train_features_output_path}")

    # Create full dataset
    time_series_full = []
    for ts in time_series:
        time_series_full.append(ts)

    full_data_output_path = os.path.join(
        "/opt/ml/processing/output/full_data", "full_data.csv"
    )

    logger.info("Saving full data to {}".format(full_data_output_path))

    preprocessed_df.to_csv(full_data_output_path, index = False)
    
    logger.info(f"Full data saved to {full_data_output_path}")
