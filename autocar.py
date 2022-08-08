#!/usr/bin/python3

import sys
import os
import argparse
import pandas as pd
import numpy as np
from termcolor import colored, cprint
from importlib import import_module
from models.utils import *
from spinner import Spinner
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def float_range(mini,maxi):
    # Define the function with default arguments
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a Floating Point Number")
        if f <= mini or f >= maxi:
            raise argparse.ArgumentTypeError("Must be > " + str(mini) + " and < " + str(maxi))
        return f
    return float_range_checker

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        global logger
        self.print_help()
        msg = colored(message, 'red')
        logger.error(msg)
        sys.exit(2)

def parse_args(argv):
    global models_type
    argv_lower = [x.lower() for x in argv]
    parser = DefaultHelpParser(formatter_class = argparse.RawTextHelpFormatter)
    list_group = parser.add_mutually_exclusive_group(required = False)
    list_group.add_argument(
        '--list-models', nargs = '+', metavar = 'MODEL_TYPE',
        help = "Show List of Models and Exit. Choices: " + str(models_type),
        choices = models_type, type = str.lower)
    list_group.add_argument(
        '--list-models-all', help = 'Show List of All Models and Exit.',
        action = 'store_true')
    list_mdls =  any([x in ('--list-models', '--list-models-all') for x in argv_lower])

    if list_mdls:
        args = parser.parse_args(argv)
        return args

    parser.add_argument(
        '-d', '--datasets', nargs = '+', metavar = 'DATASET',
        help = 'One or More Datasets (csv Files). For All Datasets in Directory Use: [DIR_PATH]/*.csv',
        type = str,  required = not list_mdls)
    parser.add_argument(
        '--use-balanced-datasets', help = "Use Balanced Datasets.",
        action = 'store_true')
    parser.add_argument(
        '--verbose', help = "Show More Run Info.",
        action = 'store_true')

    cbar_group = parser.add_mutually_exclusive_group(required = not list_mdls)
    cbar_group.add_argument(
        '--run-cbar', nargs = '+', metavar = 'CBAR_MODEL',
        help = "Run Selected CBAR Models. Choices: " + str(models_dict['cbar']),
        choices = models_dict['cbar'], type = str.lower)
    cbar_group.add_argument(
        '--run-cbar-all', help = "Run All CBAR Models.",
        action = 'store_true')
    ml_group = parser.add_mutually_exclusive_group(required = False)
    ml_group.add_argument(
        '--run-ml', nargs = '+', metavar = 'ML_MODEL',
        help = "Run Selected Machine Learning (ML) Models. Choices: " + str(models_dict['ml']),
        choices = models_dict['ml'], type = str.lower)
    ml_group.add_argument(
        '--run-ml-all', help = "Run All Machine Learning (ML) Models.",
        action = 'store_true')
    graphs_group = parser.add_argument_group('Parameters for Graphics')
    graphs_group.add_argument(
        '--plot-graph', nargs = '+', metavar = 'GRAPH_TYPE',
        help = "Run Selected Graphics. Choices: " + str(graphics_type),
        choices = graphics_type, type = str.lower)

    cbar_complementar_args =  any([x in ('cba', 'cmar', 'eqar', '--run-cbar-all') for x in argv_lower])
    group_cbar = parser.add_argument_group('Additional Parameters for CBA / CMAR / EQAR')
    if cbar_complementar_args:
        group_cbar.add_argument(
            '-s', '--min-support', metavar = 'float', required = False,
            help = 'Minimum Support (must be > 0.0 and < 1.0). Default: 0.1',
            type = float_range(0.0, 1.0), default = 0.1)
        group_cbar.add_argument(
            '-c', '--min-confidence', metavar = 'float', required = False,
            help = 'Minimum Confidence (must be > 0.0 and < 1.0). Default: 0.95',
            type = float_range(0.0, 1.0), default = 0.95)

    cbar_complementar_args =  any([x in ('cba', 'eqar', '--run-cbar-all') for x in argv_lower])
    if cbar_complementar_args:
        group_cbar.add_argument(
            '-m', '--max-length', metavar = 'int', required = False,
            help = 'Max Length of Rules (Antecedent + Consequent) (must be > 2). Default: 5',
            type = int, default = 5)

    cbar_complementar_args =  any([x in ('eqar', '--run-cbar-all') for x in argv_lower])
    group_eqar = parser.add_argument_group('Additional Parameters for EQAR')
    if cbar_complementar_args:
        group_eqar.add_argument(
            '-a', '--algorithm', metavar = 'AR_ALGORITHM',
            choices = ['apriori', 'fpgrowth', 'eclat'],
            help = "Algorithm Used to Generate Association Rules. Default: eclat",
            type = str, default = 'eclat')
        group_eqar.add_argument(
            '-l', '--min-lift', metavar = 'float',
            help = 'Minimum lift (must be > 0.0). Default: 1.0',
            type = float, default = 1.0)
        group_eqar.add_argument(
            '-t', '--threshold', metavar = 'float',
            help = 'Percentage of Rules to be Used for Testing Samples (must be > 0.0 and < 1.0). Default: 0.2',
            type = float_range(0.0, 1.0), default = 0.2)
        q_list = ['acc', 'c1', 'c2', 'bc', 'kap', 'corr', 'cov', 'prec']
        group_eqar.add_argument(
            '-q', '--qualify', metavar = 'QUALIFY', required = not list_mdls,
            help = 'Metric for Rules Qualification. Choices: ' + str(q_list),
            choices = q_list, type = str.lower)
        group_eqar.add_argument(
            '-o', '--overwrite', help = "Delete All Previous Data.",
            action = "store_true")
        if '--use-balanced-datasets' not in argv_lower:
            group_eqar.add_argument(
                '-u', '--use-proportional-values',
                help = "Use Proportional Values to Support and Confidence.",
                action = "store_true")

    group_output = parser.add_argument_group('Parameters for Output')
    for type in models_type:
        list_models = models_dict[type]
        for model in list_models:
            all = '--run-' + type + '-all'
            output_args =  any([x in (model, all) for x in argv_lower])
            if output_args:
                argument = '--prefix-output-' + type + '-' + model
                default_prefix = type + '_' + model + '_'
                help_txt = "Prefix Output to " + type.upper() + " " + model.upper() + " Results. Default: " + default_prefix
                group_output.add_argument(
                    argument, metavar = 'PREFIX',
                    help = help_txt, type = str, default = default_prefix)
    group_output.add_argument(
        '--output-dir', metavar = 'DIRECTORY',
        help = 'Directory For Output Data. Default: outputs',
        type = str, default = "outputs")

    args = parser.parse_args(argv)

    return args

def list_models(selected_models_type):
    for m in selected_models_type:
        dir_path = os.path.join(models_path, m)
        models_in_dir = get_dir_list(dir_path)
        f = open(os.path.join(dir_path, "about.desc"), "r")
        model_desc = f.read()
        print(colored("\n>>> " + model_desc, 'green'))
        for i in models_in_dir:
            f = open(os.path.join(dir_path, i, "about.desc"), "r")
            model_desc = f.read()
            print(colored("\t" + model_desc, 'yellow'))
    exit(1)

def get_dir_list(dir_path):
    l = []
    for it in os.scandir(dir_path):
        if it.is_dir():
            l.append(it.name)
    if '__pycache__' in l:
        l.remove('__pycache__')
    return l

def run_models(dataset, dataset_file, model_type, selected_models_list, args):
    global all_results
    global predict_results
    global models_path
    global logger

    for model in selected_models_list:
        print("Running Model", colored(model.upper(), 'green'),
                "to Dataset", colored(dataset_file, 'green'))
        try:
            module = '.'.join([models_path, model_type, model, 'run'])
            model_instance = import_module(module)
            c, p = model_instance.run(dataset, dataset_file, args)
            if p:
                r_df = result_dataframe(c, p)
                print(colored(model.upper() + ' Results to Dataset ' + dataset_file, 'blue', attrs = ['bold']))
                r_str = format_result(r_df)
                print(colored(r_str, 'blue', attrs = ['bold']))
                prefix_output_file = getattr(args, 'prefix_output_' + model_type + '_' + model)
                output_file = prefix_output_file + dataset_file.replace("/", "_")
                output_file_path = os.path.join(args.output_dir, output_file)
                r_df.to_csv(output_file_path, index = False)
                r_df['model'] = model
                r_df['dataset'] = dataset_file
                all_results = pd.concat([all_results, r_df], ignore_index = True)
                predict_results.append([c, p, model, dataset_file])
            else:
                msg = colored('No ' + model.upper() + ' Results to Dataset ' + dataset_file + '.', 'yellow')
                logger.warning(msg)
        except BaseException as e:
            msg = colored('Exception in Execution of ' + model_type.upper() + " " + model.upper() + ': {}'.format(e), 'red')
            logger.error(msg)

def get_models():
    d = {}
    for model in models_type:
        models_type_path = os.path.join(models_path, model)
        dir_list = get_dir_list(models_type_path)
        d[model] = dir_list
    return d

def graph_metrics(dataset_file, output_dir):
    global all_results
    if all_results.empty:
        return

    dataset_results = all_results[(all_results['dataset'] == dataset_file)]
    models_index = list(dataset_results['model'].str.upper())
    metrics_dict = dict()
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']
    for metric in metrics_list:
        metrics_dict[metric] = list(dataset_results[metric] * 100.0)

    df = pd.DataFrame(metrics_dict, index = models_index)
    df.columns = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC']
    ax = df.plot.bar(rot = 0, edgecolor='white', linewidth = 1)
    ax.set_xlabel('Models')
    ax.set_ylabel('Values (%)')
    ax.legend(ncol = 1, loc = 'lower left')
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, len(models_index))
    ax.set_title('Metrics to ' + dataset_file)
    graph_file = 'gm_' + os.path.splitext(dataset_file.replace("/", "_"))[0] + '.pdf'
    path_graph_file = os.path.join(output_dir, graph_file)
    ax.figure.savefig(path_graph_file)

def graph_class(dataset_file, output_dir):
    global all_results
    if all_results.empty:
        return

    dataset_results = all_results[(all_results['dataset'] == dataset_file)]
    models_index = list(dataset_results['model'].str.upper())
    classification_dict = dict()
    classification_list = ['TP', 'FP', 'TN', 'FN']
    for classification in classification_list:
        classification_dict[classification] = list(dataset_results[classification.lower()])

    df = pd.DataFrame(classification_dict, index = models_index)
    stacked_data = df.apply(lambda x: x*100/sum(x), axis = 1)
    ax = stacked_data.plot.barh(rot = 0, stacked = True)
    ax.set_xlabel('Values (%)')
    ax.set_ylabel('Models')
    ax.set_ylim(-1, len(models_index))
    ax.legend(ncol = len(classification_list), loc = 'upper center')
    for container in ax.containers:
        if container.datavalues[0] > 1.0:
            ax.bar_label(container, label_type = 'center', color = 'black', weight='bold', fmt = '%.2f')
    ax.set_title('Classification to ' + dataset_file)
    graph_file = 'gc_' + os.path.splitext(dataset_file.replace("/", "_"))[0] + '.pdf'
    path_graph_file = os.path.join(output_dir, graph_file)
    ax.figure.savefig(path_graph_file)

def graph_roc_auc(dataset_file, output_dir):
    global all_results
    if all_results.empty:
        return

    dataset_results = all_results[(all_results['dataset'] == dataset_file)]
    models_list = list(dataset_results['model'].str.upper())
    values_list = list(dataset_results['roc_auc'] * 100.0)
    markers = ['o', '^', 's', 'P', 'X', 'd', '*', 'x', '+', 'D', 'v']
    plt.clf()
    i = 0.0
    for px, py, m in zip(models_list, values_list, markers):
        plt.scatter(px, py, marker = m, s = py)
        plt.annotate("{:.2f}".format(py), (i - 0.2, py - 6.0))
        i += 1.0

    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.xlim(-1, len(models_list))
    plt.ylim(0, 100)
    plt.title('ROC AuC to ' + dataset_file)

    graph_file = 'gr_' + os.path.splitext(dataset_file.replace("/", "_"))[0] + '.pdf'
    path_graph_file = os.path.join(output_dir, graph_file)
    plt.savefig(path_graph_file)

def graph_roc_curve(dataset_file, output_dir):
    global roc_results
    if roc_results.empty:
        return
    dataset_results = roc_results[(roc_results['dataset'] == dataset_file)]
    models_list = set(dataset_results['model'])
    plt.clf()
    plt.plot([0, 1], [0, 1], color = 'black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve to ' + dataset_file)

    for m in models_list:
        df = dataset_results[(dataset_results['model'] == m)]
        fper = df['fper'].to_numpy()
        tper = df['tper'].to_numpy()
        plt.plot(fper, tper, label = m.upper())

    plt.legend(loc = 'lower right')
    graph_file = 'grc_' + os.path.splitext(dataset_file.replace("/", "_"))[0] + '.pdf'
    path_graph_file = os.path.join(output_dir, graph_file)
    plt.savefig(path_graph_file)

if __name__=="__main__":
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global models_path
    global models_type
    global models_dict
    global models_output_files
    global logger
    global all_results
    global predict_results
    global roc_results
    logger = logging.getLogger('AutoCAR')
    models_path = 'models'
    models_type = get_dir_list(models_path)
    models_dict = get_models()
    all_results = pd.DataFrame()
    roc_results = pd.DataFrame()
    predict_results = list()

    global graphics_type
    graphics_type = ['metrics', 'roc_auc', 'class', 'roc_curve']

    args = parse_args(sys.argv[1:])
    #print(args)

    if args.verbose:
        logger.setLevel(logging.INFO)

    if args.list_models:
        list_models(args.list_models)
    elif args.list_models_all:
        list_models(models_type)

    output_dir = args.output_dir
    if not output_dir.startswith('.'):
        root_path = os.path.dirname(os.path.realpath(__file__))
        check_directory(root_path, output_dir)
    else:
        msg = colored("Exception: Directory Name Not Allowed.", 'red')
        logger.error(msg)
        exit(1)

    dataset_file_list = args.datasets
    for dataset_file in dataset_file_list:
        try:
            dataset = pd.read_csv(dataset_file)
        except BaseException as e:
            msg = colored("Exception: {}".format(e), 'red')
            logger.error(msg)
            exit(1)

        if args.use_balanced_datasets:
            logger.info('Using Balanced Dataset')
            dataset = balanced_dataset(dataset)

        for model_type in models_type:
            selected_models_list = getattr(args, 'run_' + model_type)
            if selected_models_list:
                run_models(dataset, dataset_file, model_type, selected_models_list, args)
            elif getattr(args, 'run_' + model_type + '_all'):
                selected_models_list = models_dict[model_type]
                run_models(dataset, dataset_file, model_type, selected_models_list, args)

        logger.info('Saving Result to %s' % dataset_file)

        file_path = os.path.join(args.output_dir, 'general_results.csv')
        all_results.to_csv(file_path, index = False)

        for p in predict_results:
            fper, tper, thresholds = roc_curve(p[0], p[1])
            roc_df = pd.DataFrame()
            roc_df['fper'] = fper
            roc_df['tper'] = tper
            roc_df['model'] = p[2]
            roc_df['dataset'] = p[3]
            roc_results = pd.concat([roc_results, roc_df], ignore_index = True)

        file_path = os.path.join(args.output_dir,'roc_curve_results.csv')
        roc_results.to_csv(file_path, index = False)

        logger.info('Ploting Graphics to %s' % dataset_file)
        if args.plot_graph:
            for g in args.plot_graph:
                (globals()['graph_' + g])(dataset_file, args.output_dir)
        else:
            graph_class(dataset_file, args.output_dir)
            graph_metrics(dataset_file, args.output_dir)
            graph_roc_auc(dataset_file, args.output_dir)
            graph_roc_curve(dataset_file, args.output_dir)
