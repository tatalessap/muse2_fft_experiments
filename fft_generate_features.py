import os
import numpy as np
import pandas as pd
from utils import *


def signal_sub(signal):
    """
    Generate signals divided into subsignals in equal measure, if the last subsignal does not reach this measure is discarded.
    :param signal: signal to be divided into sub-windows
    :return: sub windows
    """
    windows_size = 128
    windows = [signal[windows_size * k:windows_size * (k + 1)] for k in range(int(signal.size / windows_size))]
    return windows


def verify(answer):
    """
    verify that the answer has a series of properties
    :param answer: the answer to verify
    :return: True or False
    """
    number_RC = pd.Timedelta(answer['Response Time'].iloc[0]).seconds * 256
    number_RC = number_RC + ((pd.Timedelta(answer['Response Time'].iloc[0]).microseconds * 256) / 1000000)
    number_RC = int(number_RC)

    return (answer.shape[0] * 100) / number_RC >= 95 and answer.shape[0] >= 128


def get_k_greater_freq(sub_signal, k):
    """
    Blackman and fft are applied to the sub signal. The largest k values are extracted and the frequencies corresponding to it are identified
    :param sub_signal:
    :param k:
    :return: feature vector by k frequency
    """
    # calculate seconds, apply black man and calculate fft
    seconds = sub_signal.size / 256
    window = np.blackman(sub_signal.size)
    fft_vector = np.fft.rfft(sub_signal * window)

    # creates frequency vector, finds the largest fft values and
    # indicates which frequencies they correspond to
    fft_vector = np.abs(fft_vector)

    freq_vector = np.array(np.arange(0, fft_vector.size)) * (1 / seconds)

    freq_vector = freq_vector[3:]
    fft_vector = fft_vector[3:]

    indices, values = numbers_max(k, fft_vector)

    freq_feature = [freq_vector[index] for index in indices]  # feature vector

    return freq_feature


def generate_feature_vectors_by_data_greater_freq(data, sensors, columns):
    """
    Feature vector generator from a dataset. For each given response,
    create sub-windows and extract for each sensor the k frequencies corresponding to the greater values of fft.
    :param columns:
    :param data: the data (by one file)
    :param sensors: set of sensors
    :return: features by one data
    """
    k = 10  # length of feature vector single
    feature_set = list()
    class_column = list()
    indicator_column = list()
    windows_size = 128
    print("Size of data: " + str(data.shape))
    annotations = [x for _, x in data.groupby(['Time Check Button'])]  # set of response
    for answer in annotations:  # for each annotated image
        if verify(answer):  # check if the image can be evaluated
            windows = [answer[windows_size * k:windows_size * (k + 1)] for k in range(int(len(answer) / windows_size))] # Subdivide the data into sub-windows
            feature_vectors = list()
            for index_sub in range(len(windows)):
                for sensor in sensors:
                    signal = np.array(windows[index_sub][sensor], dtype=float)  # the signal of sub window

                    if len(feature_vectors) == 0:
                        feature_vectors = [np.array([]) for i in range(len(windows))]  # initialize vectors features

                    feature_raw = get_k_greater_freq(signal, k)

                    # add the vector feature to the main vector feature, corresponding to the sub-window feature
                    feature_vectors[index_sub] = np.append(feature_vectors[index_sub], feature_raw)

                feature_waves = list(windows[index_sub][columns].mean())

                feature_vectors[index_sub] = list(feature_waves) + list(feature_vectors[index_sub])

            # for each feature vector created, add the class and update the indicator list
            for f_vector in feature_vectors:
                class_column.append(answer['label'].iloc[0])
                if 'indicator' in answer.columns:
                    indicator_column.append((answer['indicator'].iloc[0], answer['Image'].iloc[0]))
                # and add vectors to the main feature set vectors
                if len(feature_set) == 0:  # initialize
                    feature_set = np.empty((0, (len(f_vector))))
                feature_set = np.vstack((feature_set, f_vector))

    return feature_set, class_column, indicator_column


def generate_feature_vectors_by_data_pad_fft(data, sensors, columns):
    """
    Feature vector generator from a dataset. For each given response, create a feature vector fft
    :param data: the data (by one file)
    :param sensors: set of sensors
    :return: features by one data
    """
    feature_set = list()
    class_column = list()
    indicator_column = list()
    annotations = [x for _, x in data.groupby(['Time Check Button'])]  # set of response
    for answer in annotations:
        class_column.append(answer['label'].iloc[0])
        if 'indicator' in answer.columns:
            indicator_column.append(answer['indicator'].iloc[0])

        fft_all_sensors = np.array([])

        for sensor in sensors:
            signal = np.array(answer[sensor], dtype=float)
            window = np.blackman(signal.size)
            fft_vector = np.fft.rfft(pad_to(signal * window, seconds=8, fs=256))
            fft_all_sensors = np.append(fft_all_sensors, fft_vector)  # add to feature vector of the answer

        if len(feature_set) == 0:  # initial
            feature_set = np.empty((0, len(fft_all_sensors)))
        feature_set = np.vstack((feature_set, np.abs(fft_all_sensors)))

    return feature_set, class_column, indicator_column


def generate_feature_sets_by_one_file(path_to_file, name_to_save, sensors, columns, generator_feature_vectors_by_data):
    """
    creates a feature vector file for each dataset
    :param columns:
    :param path_to_file: path from where to take files to extract vector features from
    :param name_to_save: name to save file in features
    :param sensors: list of sensors
    :param generator_feature_vectors_by_data: function with which to create vector features
    """
    datasets = [pd.read_csv(path_to_file + str(el)) for el in os.listdir(path_to_file)]  # sets of file raw
    for index in range(len(datasets)):
        print("\n")
        print("Dataset: " + str(index))
        if index == 2:
            i = 0
        feature_set, class_column, indicator_column = generator_feature_vectors_by_data(datasets[index], sensors, columns)
        print("Size of list of feature vectors: " + str(len(feature_set)))
        print("Size of list of class vectors: " + str(len(class_column)))
        print("Size of list of indicator vectors: " + str(len(indicator_column)))
        features = pd.DataFrame(feature_set)
        features['classes'] = np.array(class_column)
        features.to_csv(name_to_save + str(index) + '.csv', index=False)
        if len(indicator_column) != 0:
            indicator = pd.DataFrame()
            indicator['indicator'] = np.array([indicator[0] for indicator in indicator_column])
            indicator['image'] = np.array([indicator[1] for indicator in indicator_column])
            indicator.to_csv(name_to_save + str(index) + '_indicator' + '.csv', index=False)


def generate_feature_set_by_all_files(path_to_file, name_to_save, sensors, columns, generator_feature_vectors_by_data):
    """
    creation of a single file where all vector features are included
    :param columns:
    :param path_to_file: path from where to take files to extract vector features from
    :param name_to_save: name to save file in features
    :param sensors: list of sensors
    :param generator_feature_vectors_by_data: function with which to create vector features
    """
    datasets = [pd.read_csv(path_to_file + str(el)) for el in os.listdir(path_to_file)]  # sets of file raw
    class_column_final = list()
    indicator_column_final = list()
    indexes = []
    features = pd.DataFrame()
    for index in range(len(datasets)):
        print("\n")
        print("Dataset: " + str(index))
        feature_set, class_column, indicator_column = generator_feature_vectors_by_data(datasets[index], sensors, columns)
        if index == 0:
            features = pd.DataFrame(feature_set)
        else:
            features = pd.concat([features, pd.DataFrame(feature_set)], ignore_index=True)
        indexes.append(len(class_column_final))
        class_column_final = class_column_final + class_column
        indicator_column_final = indicator_column_final + indicator_column
        print("Size of list of feature vectors: " + str(len(features)))
        print("Size of list of class vectors: " + str(len(class_column_final)))
        print("Size of list of indicator vectors: " + str(len(indicator_column_final)))

    features['classes'] = np.array(class_column_final)
    if len(indicator_column_final) != 0:
        indicator = pd.DataFrame()
        indicator['indicator'] = np.array([indicator[0] for indicator in indicator_column_final])
        indicator['image'] = np.array([indicator[1] for indicator in indicator_column_final])
        indicator.to_csv(name_to_save + '_indicator' + '.csv', index=False)

    np.savez('index', np.array(indexes))
    features.to_csv(name_to_save + '.csv', index=False)
