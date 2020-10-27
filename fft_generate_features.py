import numpy as np
import os
from sklearn.decomposition import FastICA, PCA
import datetime
import pandas as pd
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt


def pad_to(sub_signal, seconds, fs):
    i = 0
    new_signal = []
    size_signal = seconds*fs
    for j in range(size_signal):
        if i >= len(sub_signal):
            new_signal.append(0)
        else:
            new_signal.append(sub_signal[i])
        i = i+1
    signal = np.array(new_signal)
    return signal

def generate_feature_set(data, sensors):
    """
    :param data: the data (by one file)
    :param sensors: set of sensors
    :return: features by one data
    """
    feature_set = list()
    class_column = list()
    annotations = [x for _, x in data.groupby(['Time Check Button'])]  # set of response

    for answer in annotations:
        class_column.append(answer['label'].iloc[0])
        fft_all_sensors = np.array([])
        for sensor in sensors:
            signal = np.array(answer[sensor], dtype=float)
            window = np.blackman(signal.size)
            fft_vector = np.fft.rfft(pad_to(signal * window, seconds=8, fs=256))
            fft_all_sensors = np.append(fft_all_sensors, fft_vector)  # add to feature vector of the answer
        if len(feature_set) == 0:  # initial
            feature_set = np.empty((0, len(fft_all_sensors)))
        feature_set = np.vstack((feature_set, np.abs(fft_all_sensors)))

    return feature_set, class_column

def generate_feature_sets_by_one_file(path_to_file, name_to_save, sensors):
    datasets = [pd.read_csv(path_to_file + str(el)) for el in os.listdir(path_to_file)]  # sets of file raw
    for index in range(len(datasets)):
        feature_set, class_column = generate_feature_set(datasets[index], sensors)
        features = pd.DataFrame(feature_set)
        features['classes'] = np.array(class_column)
        features.to_csv(name_to_save+str(index)+'.csv', index=False)

def generate_feature_set_by_all_files(path_to_file, name_to_save, sensors):
    datasets = [pd.read_csv(path_to_file + str(el)) for el in os.listdir(path_to_file)]  # sets of file raw
    class_column_final = list()
    indexes = []
    features = pd.DataFrame()
    for index in range(len(datasets)):
        feature_set, class_column = generate_feature_set(datasets[index], sensors)
        if index == 0:
            features = pd.DataFrame(feature_set)
        else:
            features = pd.concat([features, pd.DataFrame(feature_set)], ignore_index=True)
        indexes.append(len(class_column_final))
        class_column_final = class_column_final+class_column

    features['classes'] = np.array(class_column_final)
    np.savez('index', np.array(indexes))

    i = 0
    features.to_csv(name_to_save+'.csv', index=False)











