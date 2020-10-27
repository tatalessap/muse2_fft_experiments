import weka
from weka.classifiers import Classifier, PredictionOutput, KernelClassifier, Kernel, Evaluation
import weka.core.converters as converters
from weka.core.classes import Random
import weka.core.jvm as jvm
import os
import pandas as pd
import numpy as np
from numpy import load

def java_start(path_packages):
    jvm.start(packages=path_packages)

def java_stop():
    jvm.stop()


def experiment_sequential_file(path_indices, path_features, path_folder_save_results, options, classifier, name):
    ind_f = load(path_indices)

    lst = ind_f.files

    for item in lst:
        ind = ind_f[item] + 1

    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))

    data = converters.load_any_file(path_features)

    ind = np.append(ind, len(data))

    data.class_is_last()

    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")

    d_results = {'index': [], 'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}

    for j in range(len(ind) - 1):
        print(j)

        print(str(ind[j]) + '-' + str(ind[j + 1]))

        d_test = data.subset(row_range=str(ind[j]) + '-' + str(ind[j + 1]))

        if j == 0:  # first
            d_train = data.subset(row_range=str(ind[j + 1] + 1) + '-' + str(ind[-1]))  # last element
        elif j == len(ind) - 2:  # last
            d_train = data.subset(row_range='1-' + str(ind[j] - 1))  # last element
        else:  # central
            s = '1-' + str(ind[j] - 1) + ',' + str(ind[j + 1] + 1) + '-' + str(ind[-1])
            d_train = data.subset(row_range=s)

        cls.build_classifier(d_train)

        evl = Evaluation(data)
        evl.test_model(cls, d_test, pout)

        save = pout.buffer_content()

        with open(path_folder_save_results + '/' + '/prediction/' + name + str(j) + 'pred_data.csv', 'w') as f:
            f.write(save)

        d_results['index'].append(str(ind[j]))
        d_results['percent_correct'].append(evl.percent_correct)
        d_results['percent_incorrect'].append(evl.percent_incorrect)
        d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder_save_results + '/' + name + 'results.csv', index=False)


def experiment_file_random(path_features, path_folder_save_results, options, classifier, fold, random, name):
    print("start weka")
    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))
    d_results = {'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}
    data = converters.load_any_file(path_features)
    data.class_is_last()
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, fold, Random(random), pout)
    d_results['percent_correct'].append(evl.percent_correct)
    d_results['percent_incorrect'].append(evl.percent_incorrect)
    d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder_save_results + '/' + str(name) + '.csv', index=False)

    save = pout.buffer_content()

    with open(path_folder_save_results + '/' + 'prediction/' + str(name) + '.csv', 'w') as f:
        f.write(save)


def experiment_more_file(path_files, path_folder_save_results, fold, options, classifier, random, name):
    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))

    file_list = os.listdir(path_files)

    for file in file_list:
        if ".csv" not in file:
            file_list.remove(file)

    d_results = {'name_file': [], 'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}

    print(file_list)

    for file in file_list:
        print(str(file))
        data = converters.load_any_file(path_files + "/" + file)

        data.class_is_last()

        pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")

        evl = Evaluation(data)

        evl.crossvalidate_model(cls, data, fold, Random(random), pout)

        d_results['name_file'].append(str(file))
        d_results['percent_correct'].append(evl.percent_correct)
        d_results['percent_incorrect'].append(evl.percent_incorrect)
        d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

        save = pout.buffer_content()

        with open(path_folder_save_results + '/' + 'prediction/' + str(name) + str(file)[:-4] + 'pred_data.csv', 'w') as f:
            f.write(save)

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder_save_results + '/' + str(name) + ".csv", index=False)