
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

"""Evaluation script for measuring mean squared error."""

import os 


# output_dir = "/opt/ml/processing/evaluation"
# test_path = "/opt/ml/processing/test/test.json"
# actuals_path = "/opt/ml/processing/actuals/test.json.out"

def interpolate_zero(x):
    ind = 0
    for i in x:
        if i == 0.0:
            x[ind] = (x[ind-1]+x[ind+1])/2.0
        ind += 1
    return(x)

def rmse_deepar(preds, actuals, split_days):

    num_ts = len(preds)
    assert len(preds) == len(actuals), "Predictions and actuals should have same number of time series."

    squares = []
    for i in range(len(preds)):
        squares.append(np.subtract(preds[i]['mean'], interpolate_zero(actuals[i]['target'][-split_days:]))**2)
        
    sum_squares = np.sum(np.reshape(squares, num_ts*split_days))
    rmse = np.sqrt((1.0/(num_ts*split_days))*sum_squares)
    return(rmse)

def std_deepar(preds, actuals, split_days):

    num_ts = len(preds)

    preds_list = []
    for i in preds:
        preds_list.append(i['mean'])

    yhat = np.reshape(preds_list, num_ts*split_days)

    actuals_list = []
    for i in actuals:
        actuals_list.append(i['target'][-split_days:])

    y = np.reshape(actuals_list, num_ts*split_days)    

    return(np.std(y - yhat))


if __name__ == "__main__":
    logger.info("Starting evaluation.")
    
    print(os.listdir("/opt/ml/processing/predictions/"))
    print(os.listdir("/opt/ml/processing/actuals/"))
    

    logger.info("Reading predictions data...")
    test_path = "/opt/ml/processing/predictions/test.json.out"
    preds = []
    with open(test_path) as f:
        for line in f:
            preds.append(json.loads(line))

    logger.debug("Reading actual data....")
    actuals_path = "/opt/ml/processing/actuals/test.json"
    actuals = []
    with open(actuals_path) as f:
        for line in f:
            actuals.append(json.loads(line))
            
    
    logger.debug("Calculating root mean squared error.")
    rmse_deepar_value = rmse_deepar(preds, actuals, split_days = 24)

    logger.debug("Calculation standard deviation of errors.")
    standard_deviation = std_deepar(preds, actuals, 24)

    logger.debug("Creating report dictionary...")
    report_dict = {
    "regression_metrics": {
    "rmse" : {
      "value" : rmse_deepar_value,
      "standard_deviation" : standard_deviation
      }
    }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", rmse_deepar_value)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
