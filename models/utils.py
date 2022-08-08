from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import shutil

def format_result(result_df):
    result = result_df.to_dict('records')[0]
    if not result:
        return "No Results."
    result_str = "TP: {} TN: {} FP: {} FN: {}".format(result.get("tp"),result.get("tn"), result.get("fp"), result.get("fn"))
    precision = result.get("precision") * 100.0
    accuracy = result.get("accuracy") * 100.0
    recall = result.get("recall") * 100.0
    f1_score = result.get("f1_score") * 100.0
    mcc = result.get("mcc")
    roc_auc = result.get("roc_auc")
    result_str += "\nAccuracy: {:.3f}".format(accuracy)
    result_str += "\nPrecision: {:.3f}".format(precision) #Ability to Correctly Detect Malware
    result_str += "\nRecall: {:.3f}".format(recall)
    result_str += "\nF1 Score: {:.3f}".format(f1_score)
    result_str += "\nMCC: {:.3f}".format(mcc)
    result_str += "\nROC AuC: {:.3f}\n".format(roc_auc)
    return result_str

def result_dataframe(classification, prediction, num_rules = 0):
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

def balanced_dataset(dataset):
    B = dataset[(dataset['class'] == 0)]
    M = dataset[(dataset['class'] == 1)]

    lenB = len(B)
    lenM = len(M)
    b_dataset = None
    if lenB > lenM:
        random_select = B.sample(n = lenM, random_state = 0)
        b_dataset = pd.concat([random_select, M], ignore_index = True)
    else:
        random_select = M.sample(n = lenB, random_state = 0)
        b_dataset = pd.concat([B, random_select], ignore_index = True)
    #b_dataset.to_csv("balanced_" + args.dataset, index = False)
    return b_dataset

def check_directory(root_path, dir_name):
    dir_path = os.path.join(root_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_name)
