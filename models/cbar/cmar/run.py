import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import sys
import os
from spinner import Spinner
from models.utils import *
from termcolor import colored, cprint
import logging

def exec_cmar(path_to_r_file, train, test, s, c):
    logger.info("Starting CBAR CMAR in R.")
    prediction_list = []
    try:
        r = ro.r
        r['options'](warn = -1)
        r.source(path_to_r_file)
        logger.info("Converting Data to R Format.")
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_train = ro.conversion.py2rpy(train)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_test = ro.conversion.py2rpy(test)

        logger.info("Executing CMAR Model.")
        p = r.cmar(df_train, df_test, s, c)
        to_np = np.array(p)
        to_list = to_np.tolist()
        prediction_list = [x for x in to_list]
        len_before = len(prediction_list)
        prediction_list = list(filter(lambda p: p == 0 or p == 1, prediction_list))
        if len(prediction_list) < len_before:
            prediction_list = []
    except BaseException as e:
        msg = colored('Exception When Running CBAR CMAR (exec_cmar): {}'.format(e), 'red')
        logger.error(msg)
    return prediction_list

#if __name__=="__main__":
def run(dataset, dataset_file, args):
    global logger
    logger = logging.getLogger('CMAR')
    if args.verbose:
        logger.setLevel(logging.INFO)

    path_to_r_file = os.path.dirname(os.path.realpath(__file__))
    path_to_r_file = os.path.join(path_to_r_file, "cmar.r")

    dataset_class = dataset['class']

    skf = StratifiedKFold(n_splits = 5)
    fold_no = 1
    general_class = []
    general_prediction = []
    for train_index, test_index in skf.split(dataset, dataset_class):
        train = dataset.loc[train_index,:]
        test = dataset.loc[test_index,:]
        logger.info("Executing Fold {}".format(fold_no))
        if not args.verbose:
            spn = Spinner("Executing Fold {}".format(fold_no))
            spn.start()
        prediction_result = exec_cmar(path_to_r_file, train, test, args.min_support, args.min_confidence)
        if not args.verbose:
            spn.stop()
        if prediction_result:
            general_class += list(test['class'])
            general_prediction += prediction_result
        fold_no += 1

    return general_class, general_prediction
