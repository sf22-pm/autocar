import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from functools import partial
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from termcolor import colored, cprint
import shutil
import os
import logging

def parallelize_func(func, fixed_parameter, const_parameters = None, cores = cpu_count()):
    logger = logging.getLogger('EQAR')
    tqdm_disable = logger.getEffectiveLevel() > logging.INFO
    x = len(fixed_parameter)
    with Pool(cores) as pool:
        if const_parameters == None:
            result = list(tqdm(pool.imap(func, fixed_parameter), total = x, disable = tqdm_disable))
        else:
            result = list(tqdm(pool.imap(partial(func, const_parameters = const_parameters), fixed_parameter), total=x, disable = tqdm_disable))
    return result

def file_content(dir_path, fold_no, file):
    f_name = os.path.join(dir_path, str(fold_no) + "_" + file)
    ct = []
    try:
        with open(f_name) as f:
            if file == "log":
                ct = [l for l in f]
                ct = ct[0]
            elif file in ["mw_rules", "bw_rules", "mw_qualify_parameters", "bw_qualify_parameters"]:
                ct = [list(map(int, l.split(','))) for l in f]
            elif file in ["rules_intersection", "rules_subset", "rules_superset", "rules_qualify"]:# or file.startswith("rules_qualify"):
                ct = [tuple(map(int, l.split(','))) for l in f]
            elif file == "test_apps":
                ct = [list(map(float, l.split(','))) for l in f]
                ct = ct[0]
            elif file == "times":
                ct = [(l.split(',')[0], float(l.split(',')[1])) for l in f]
        return ct
    except BaseException as e:
        #print(e)
        return ct

def update_log(dir_path, fold_no, step):
    f_name = os.path.join(dir_path, str(fold_no) + "_log")
    with open(f_name, 'w') as f:
        f.write(step)

def save_to_file(obj_list, dir_path, fold_no, step):
    f_name = os.path.join(dir_path, str(fold_no) + "_" + step)
    with open(f_name,"w", newline='') as f:
        f_writer = csv.writer(f)
        for obj in obj_list:
            f_writer.writerow(obj)

def load_data(location, fold_no, stopped_step):
    mw_rules = file_content(location, fold_no, "mw_rules")
    bw_rules = file_content(location, fold_no, "bw_rules")
    time_l = file_content(location, fold_no, "times")
    time_l = time_l[:stopped_step + 1]
    return mw_rules, bw_rules, time_l

def save_data(mw_rules, bw_rules, time_l, location, fold_no):
    save_to_file(mw_rules, location, fold_no, "mw_rules")
    save_to_file(bw_rules, location, fold_no, "bw_rules")
    save_to_file(time_l, location, fold_no, "times")

def check_directories(dataset_file, args):
    # - ->> directories to save data <<- -#
    root_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = (dataset_file).split('/')[-1]
    dir_name = dir_name.split('.')[0]
    dir_name += "_S" + str(int(args.min_support * 100.0))
    dir_name += "_C" + str(int(args.min_confidence * 100.0))

    path = os.path.join(root_path, "run_files", dir_name)
    DIR_BASE = path
    if args.overwrite and os.path.exists(path):
        shutil.rmtree(path)

    if args.qualify:
        path = os.path.join(path, args.qualify)
        DIR_QFY = path

    th_int = int(args.threshold * 100.0)
    th_str = "0" + str(th_int) if th_int < 10 else str(th_int)
    th_dir = "T" + th_str

    DIR_TH = os.path.join(path, th_dir)
    if not os.path.exists(DIR_TH):
        os.makedirs(DIR_TH)

    return DIR_BASE, DIR_QFY, DIR_TH

def dataset_transaction(transaction):
    num_features = len(transaction)
    l = []
    for i in range(0, num_features):
        if transaction[i] != 0:
            l.append(i)
    return l

#Dictionary For Column Names
report_colnames = {
    'a': 'support_itemset_absolute',
    's': 'support_itemset_relative',
    'S': 'support_itemset_relative_pct',
    'b': 'support_antecedent_absolute',
    'x': 'support_antecedent_relative',
    'X': 'support_antecedent_relative_pct',
    'h': 'support_consequent_absolute',
    'y': 'support_consequent_relative',
    'Y': 'support_consequent_relative_pct',
    'c': 'confidence',
    'C': 'confidence_pct',
    'l': 'lift',
    'L': 'lift_pct',
    'e': 'evaluation',
    'E': 'evaluation_pct',
    'Q': 'xx',
    'S': 'support_emptyset',
}

def to_fim_format(dataset):
    result = parallelize_func(dataset_transaction, dataset)
    return result

def to_pandas_dataframe(data, report):
    global r_count
    colnames = ['consequent', 'antecedent'] + [report_colnames.get(r, r) for r in list(report)]
    dataset_df = pd.DataFrame(data, columns = colnames)
    s = dataset_df[['consequent', 'antecedent']].values.tolist()
    size_list = [len({i[0]} | set(i[1])) for i in s]
    dataset_df['size'] = size_list
    dataset_df.rename(columns = {'support_itemset_relative':'support'}, inplace = True)
    #dataset_df = dataset_df.sort_values(['support_itemset_relative','confidence', 'size'], ascending = [False, False, True])
    #dataset_df = dataset_df.sort_values('size', ascending = True)
    return dataset_df

def generate_unique_rules(dataset_df, args):
    rules = []
    supp_step = 0.05
    if not dataset_df.empty:
        ds_df = dataset_df[(dataset_df['lift'] >= args.min_lift)]
        ds_df = ds_df.sort_values('support', ascending = False)
        flag = True
        supp = args.min_support
        while flag:
            s = ds_df[['consequent', 'antecedent']].values.tolist()
            r_list = [sorted({i[0]} | set(i[1])) for i in s]
            rules = list(map(list, set(map(lambda i: tuple(i), r_list))))
            #print(len(rules), supp)
            flag = len(rules) >= 100000
            supp += supp_step
            ds_df = ds_df[(ds_df['support'] >= supp)]
    return rules

def result_dataframe(classification, prediction, num_rules = -1):
    tn, fp, fn, tp = confusion_matrix(classification, prediction).ravel()
    accuracy = metrics.accuracy_score(classification, prediction)
    precision = metrics.precision_score(classification, prediction, zero_division = 0)
    recall = metrics.recall_score(classification, prediction, zero_division = 0)
    f1_score = metrics.f1_score(classification, prediction, zero_division = 0)
    mcc = metrics.matthews_corrcoef(classification, prediction)
    roc_auc = metrics.roc_auc_score(classification, prediction)
    result_dict = {
        "num_rules": [num_rules],
        "tp": [tp],
        "tn": [tn],
        "fp": [fp],
        "fn": [fn],
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1_score],
        "mcc": [mcc],
        "roc_auc": [roc_auc]
    }
    result_df = pd.DataFrame(result_dict)
    return result_df
