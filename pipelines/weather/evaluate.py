"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")

    logger.debug("Geting model predictions...")

    logger.debug("Reading test data.")
    #test_path = "/opt/ml/processing/test/test.csv"
    #df = pd.read_csv(test_path, header=None)

    logger.debug("Reading predictions data....")

    logger.debug("Calculating root mean squared error.")
    #mse = mean_squared_error(y_test, predictions)

    logger.debug("Calculation standard deviation of errors.")
    #std = np.std(y_test - predictions)

    logger.debug("Creating report dictionary...")
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", rmse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
