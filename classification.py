import weka
import math
from weka.classifiers import Classifier, PredictionOutput, KernelClassifier, Kernel, Evaluation
import weka.core.converters as converters
from weka.core.classes import Random
import weka.core.jvm as jvm
import os
import pandas as pd
import numpy as np
from numpy import load
import datetime
from utils import *
from collections import Counter


def java_start(path_packages):
    jvm.start(packages=path_packages)


def java_stop():
    jvm.stop()


def experiment_sequential_file(path_indices, path_features, path_folder_save_results, options, classifier, name,
                               indicator_col, images):
    print(name + "  Start: " + str(datetime.datetime.now()))
    time = datetime.datetime.now()

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
        print(ind)

        first = ind[j]

        if j == len(ind) - 2:
            last = ind[j + 1]
        else:
            last = ind[j + 1] - 1

        print(str(first) + '-' + str(last))

        d_test = data.subset(row_range=str(first) + '-' + str(last))

        if j == 0:  # first
            d_train = data.subset(row_range=str(last + 1) + '-' + str(ind[-1]))  # last element
            print(str(last + 1) + '-' + str(ind[-1]))
        elif j == len(ind) - 2:  # last
            d_train = data.subset(row_range='1-' + str(first - 1))  # last element
            print('1-' + str(first - 1))
        else:  # central
            s = '1-' + str(first - 1) + ',' + str(last + 1) + '-' + str(ind[-1])
            print(s)
            d_train = data.subset(row_range=s)

        print("Lunghezza d_test:" + str(len(d_test)))
        cls.build_classifier(d_train)

        evl = Evaluation(data)
        evl.test_model(cls, d_test, pout)

        d_results['index'].append(str(ind[j]))
        d_results['percent_correct'].append(evl.percent_correct)
        d_results['percent_incorrect'].append(evl.percent_incorrect)
        d_results['confusion_matrix'].append(evl.matrix())  # Generates the confusion matrix.

    save = pout.buffer_content()

    check_folder_or_create(path_folder_save_results + '/' + 'prediction')

    with open(path_folder_save_results + '/' + 'prediction/' + name + 'pred_data.csv', 'w') as f:
        f.write(save)

    if len(indicator_col) != 0:
        save_csv = pd.read_csv(path_folder_save_results + '/' + 'prediction/' + name + 'pred_data.csv', header=None,
                               index_col=False)
        print(len(save_csv))
        print(len(indicator_col))
        save_csv['indicator'] = indicator_col
        save_csv['image'] = images
        save_csv = save_csv.rename(
            columns={0: "instance", 1: "label", 2: "prediction", 4: "perc_error", 3: "different"})
        print(save_csv.columns)
        save_csv['prediction'] = list(map(lambda x: (x[2:]), save_csv['prediction']))
        save_csv['label'] = list(map(lambda x: (x[2:]), save_csv['label']))
        save_csv.to_csv(path_folder_save_results + '/' + 'prediction/' + name + 'pred_data.csv', index=False)

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder_save_results + '/' + name + 'results.csv', index=False)

    print(name + "  End: " + str(datetime.datetime.now() - time))


def experiment_file_random(path_features, path_folder_save_results, options, classifier, fold, random, name):
    print(name + "  Start: " + str(datetime.datetime.now()))
    time = datetime.datetime.now()
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

    check_folder_or_create(path_folder_save_results + '/' + 'prediction')

    with open(path_folder_save_results + '/' + 'prediction/' + str(name) + '.csv', 'w') as f:
        f.write(save)
    print(name + "  End: " + str(datetime.datetime.now() - time))


def experiment_more_file(path_files, path_folder_save_results, fold, options, classifier, random, name):
    print(name + "  Start: " + str(datetime.datetime.now()))
    time = datetime.datetime.now()

    cls = Classifier(classname=classifier, options=weka.core.classes.split_options(options))

    file_list = os.listdir(path_files)

    for file in file_list:
        if ".csv" not in file:
            file_list.remove(file)

    d_results = {'name_file': [], 'percent_correct': [], 'percent_incorrect': [], 'confusion_matrix': []}

    # print(file_list)

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

        check_folder_or_create(path_folder_save_results + '/' + 'prediction')

        with open(path_folder_save_results + '/' + 'prediction/' + str(name) + str(file)[:-4] + 'pred_data.csv',
                  'w') as f:
            f.write(save)

    d_results = pd.DataFrame(data=d_results)

    d_results.to_csv(path_folder_save_results + '/' + str(name) + ".csv", index=False)
    print(name + "  End: " + str(datetime.datetime.now() - time))


def check(indicator, image_labels, details_indicator):
    flag = False
    if len(image_labels) % 2 == 0:
        most_c = Counter(image_labels).most_common()
        if len(most_c) == 2:
            a = Counter(image_labels).most_common()[0]
            b = Counter(image_labels).most_common()[1]
            if a[1] == b[1]:
                flag = True
                details_indicator[indicator]['tot'] = details_indicator[indicator]['tot'] + 1
                if a[0] == 'attentive':
                    details_indicator[indicator]['attentive first'] = details_indicator[indicator][
                                                                          'attentive first'] + 1
                else:
                    details_indicator[indicator]['distracted first'] = details_indicator[indicator][
                                                                           'distracted first'] + 1
    return details_indicator, flag


def voting_version2(path_file_prediction, folder_to_save_results):
    predictions = [pd.read_csv(path_file_prediction + "/" + str(el), index_col=False) for el in
                   sorted(os.listdir(path_file_prediction))]
    name_files = sorted(os.listdir(path_file_prediction))

    for i in range(len(predictions)):
        image_labels = list()
        prediction = list()
        sign = ''

        details_indicator = {'over time':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0, 'label assigned attentive': 0, 'label assigned distracted': 0},
                             'attentive':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0, 'label assigned attentive': 0, 'label assigned distracted': 0},
                             'error-no over time':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0, 'label assigned attentive': 0, 'label assigned distracted': 0},
                             'error-over time':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0, 'label assigned attentive': 0, 'label assigned distracted': 0},
                             }

        p = predictions[i]
        for index in range(p.shape[0]):
            if len(image_labels) > 0:
                if p['image'][index] != p['image'][index - 1]:

                    details_indicator, flag = check(p['indicator'][index - 1], image_labels, details_indicator)

                    if flag:
                        if len(image_labels) > 4:
                            vote = 'distracted'
                            details_indicator[p['indicator'][index - 1]]['label assigned distracted']=\
                                details_indicator[p['indicator'][index - 1]]['label assigned distracted']+1
                        else:
                            vote = 'attentive'
                            details_indicator[p['indicator'][index - 1]]['label assigned attentive'] = \
                            details_indicator[p['indicator'][index - 1]]['label assigned attentive'] + 1
                    else:
                        vote = Counter(image_labels).most_common()[0][0]

                    if vote != p['label'][index - 1]:
                        sign = 'incorrect'
                    else:
                        sign = 'correct'

                    prediction.append(
                        (p['image'][index - 1], p['indicator'][index - 1], p['label'][index - 1], vote, sign))
                    image_labels = list()

            image_labels.append(p['prediction'][index])

        df = pd.DataFrame(prediction, columns=['image', 'indicator', 'original label', 'prediction', 'status'])
        check_folder_or_create(folder_to_save_results + '/' + 'details')

        sys.stdout = open(folder_to_save_results + '/details/' + name_files[i][:-4] + '_' + "test.csv", "w")
        print(name_files[i])
        print('\n')
        print('perc incorrect: ' + str((list(df['status']).count('incorrect') * 100) / df.shape[0]))
        print('perc correct: ' + str((list(df['status']).count('correct') * 100) / df.shape[0]))
        print('\n')
        df.to_csv(folder_to_save_results + '/' + str(i) + 'prediction_voting.csv')
        c = [x for _, x in df.groupby(['status'])]
        print("Numero totale di immagini divise per indicatore:\n" + str(
            (df[['indicator', 'image']].groupby(['indicator']).count())))
        print('\n\n')
        print("Numero di immagini classificate scorrettamente con indicatore:\n" + str(
            (c[0][['indicator', 'status']].groupby(['indicator']).count())))
        print('\n')
        print("Numero di immagini classificate scorrettamente con label:\n" + str(
            (c[0][['original label', 'status']].groupby(['original label']).count())))
        print('\n\n')
        print("Caso feature pari e nessuna classe in numero maggiore: \n")
        print('attentive')
        print(details_indicator['attentive'])
        print("over time")
        print(details_indicator['over time'])
        print("error-no over time")
        print(details_indicator['error-no over time'])
        print("error-over time")
        print(details_indicator['error-over time'])
        sys.stdout.close()


def voting(path_file_prediction, folder_to_save_results):
    predictions = [pd.read_csv(path_file_prediction + "/" + str(el), index_col=False) for el in
                   sorted(os.listdir(path_file_prediction))]
    name_files = sorted(os.listdir(path_file_prediction))

    for i in range(len(predictions)):
        image_labels = list()
        prediction = list()
        sign = ''

        details_indicator = {'over time':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0},
                             'attentive':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0},
                             'error-no over time':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0},
                             'error-over time':
                                 {"tot": 0, 'attentive first': 0, 'distracted first': 0},
                             }

        p = predictions[i]
        for index in range(p.shape[0]):
            if len(image_labels) > 0:
                if p['image'][index] != p['image'][index - 1]:
                    vote = Counter(image_labels).most_common()[0][0]

                    details_indicator, flag = check(p['indicator'][index - 1], image_labels, details_indicator)

                    if vote != p['label'][index - 1]:
                        sign = 'incorrect'
                    else:
                        sign = 'correct'

                    prediction.append(
                        (p['image'][index - 1], p['indicator'][index - 1], p['label'][index - 1], vote, sign))
                    image_labels = list()

            image_labels.append(p['prediction'][index])

        df = pd.DataFrame(prediction, columns=['image', 'indicator', 'original label', 'prediction', 'status'])
        check_folder_or_create(folder_to_save_results + '/' + 'details')

        sys.stdout = open(folder_to_save_results + '/details/' + name_files[i][:-4] + '_' + "test.csv", "w")
        print(name_files[i])
        print('\n')
        print('perc incorrect: ' + str((list(df['status']).count('incorrect') * 100) / df.shape[0]))
        print('perc correct: ' + str((list(df['status']).count('correct') * 100) / df.shape[0]))
        print('\n')
        df.to_csv(folder_to_save_results + '/' + str(i) + 'prediction_voting.csv')
        c = [x for _, x in df.groupby(['status'])]
        print("Numero totale di immagini divise per indicatore:\n" + str(
            (df[['indicator', 'image']].groupby(['indicator']).count())))
        print('\n\n')
        print("Numero di immagini classificate scorrettamente con indicatore:\n" + str(
            (c[0][['indicator', 'status']].groupby(['indicator']).count())))
        print('\n')
        print("Numero di immagini classificate scorrettamente con label:\n" + str(
            (c[0][['original label', 'status']].groupby(['original label']).count())))
        print('\n\n')
        print("Caso feature pari e nessuna classe in numero maggiore: \n")
        print('attentive')
        print(details_indicator['attentive'])
        print("over time")
        print(details_indicator['over time'])
        print("error-no over time")
        print(details_indicator['error-no over time'])
        print("error-over time")
        print(details_indicator['error-over time'])
        sys.stdout.close()
