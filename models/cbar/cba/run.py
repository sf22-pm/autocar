import sys
import os
from pyarc import CBA, TransactionDB
import pandas as pd
from spinner import Spinner
import logging
from sklearn.model_selection import StratifiedKFold

def run(dataset, dataset_file, args):
    global logger
    logger = logging.getLogger('CBA')
    if args.verbose:
        logger.setLevel(logging.INFO)

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
        logger.info("Converting Data to Transactions.")
        txns_train = TransactionDB.from_DataFrame(train)
        txns_test = TransactionDB.from_DataFrame(test)

        cba = CBA(support = args.min_support, confidence = args.min_confidence, algorithm="m1", maxlen = args.max_length)
        top_rules_args = {'total_timeout':1200, 'max_iterations':500, 'init_support': args.min_support, 'init_conf':1.0, 'target_rule_count':1000}
        logger.info("Model Fit.")
        cba.fit(txns_train, top_rules_args = top_rules_args)

        if not args.verbose:
            spn.stop()
        logger.info("Making Predictions.")
        prediction_result = cba.predict(txns_test)
        prediction_result =  list(map(int, prediction_result))

        if prediction_result:
            general_class += list(test['class'])
            general_prediction += prediction_result
        fold_no += 1

    return general_class, general_prediction
