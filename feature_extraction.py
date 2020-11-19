import os
import numpy as np
import pandas as pd
from utils import *


def verify(answer):
    """
    verify that the answer has a series of properties
    :param answer: the answer to verify
    :return: True or False
    """
    percent_fills = pd.Timedelta(answer['Response Time'].iloc[0]).seconds * 256
    percent_fills = percent_fills + ((pd.Timedelta(answer['Response Time'].iloc[0]).microseconds * 256) / 1000000)
    percent_fills = int(percent_fills)

    return (answer.shape[0] * 100) / percent_fills >= 95 and answer.shape[0] >= 128


def generate_feature_set(dataset, sensors, columns=None, len_sub_window=0, fs=256, last_window=False, window=False,
                         pad=False, greater_freq=False, num_freq=10, drop=1):
    """
    Generator of a feature set vector per experiment
    :param dataset: data collected with an experiment
    :param sensors: list of channels from which to keeps signals
    :param columns: list of other data to average with (optional)
    :param len_sub_window: length of the sub window (optional)
    :param fs: sampling frequency
    :param last_window: if you want to apply a pad to the last sub window (optional with len_sub_window)
    :param window: apply Blackman (optional)
    :param pad: apply pad (optional)
    :param greater_freq: keep for feature vector the then greater frequencies (optional)
    :param num_freq: number of frequency to keep (optional with greater_freq)
    :param drop: the number of initial frequency to drop out (optional with greater_freq)
    :return: set of feature vector for dataset
    """
    feature_set = list()
    class_answer = list()
    info_answer = list()

    annotations = [x for _, x in dataset.groupby(['Time Check Button'])]  # set of response

    if len_sub_window > 1:
        for answer in annotations:
            if verify(answer):
                feature_set, class_answer, info_answer = divide_the_signal_in_sub_windows(len_sub_window, answer,
                                                                                          class_answer, info_answer,
                                                                                          sensors, feature_set,
                                                                                          last_window, fs, window, pad,
                                                                                          greater_freq, num_freq, drop,
                                                                                          columns)

    else:
        for answer in annotations:
            if verify(answer):
                feature_set, class_answer, info_answer = generate_feature_vector(answer, class_answer, info_answer,
                                                                                 sensors, feature_set, fs, columns,
                                                                                 window, pad, greater_freq, num_freq,
                                                                                 drop)

    return feature_set, class_answer, info_answer


def greater_frequencies(fft, seconds, numb_freq, drop=1):
    """
    :param fft: the vector fft
    :param seconds: seconds of the window
    :param numb_freq: number of frequencies to keep for the feature vector
    :param drop: number of initial frequencies to drop out
    :return:
    """
    freq = np.array(np.arange(0, fft.size)) * (1 / seconds)  # create the frequency vector

    freq = freq[drop:]
    fft = fft[drop:]

    indices, values = numbers_max(numb_freq, fft)  # select the numb_freq greater frequency indeces
    freq = [freq[index] for index in indices]  # select frequencies

    return freq


def divide_the_signal_in_sub_windows(len_sub_window, answer, class_answer, info_answer, sensors, feature_set,
                                     last_window=False, fs=256, window=False, pad=False, greater_freq=False,
                                     num_freq=10, drop=1, columns=None):
    """
    The answer is subdivided into several parts and feature vector is produced as in how many parts is subdivided
    :param len_sub_window: length of the sub-window
    :param answer: part of the dataset coinciding with the response 
    :param class_answer: the list of classes (by feature vector)
    :param info_answer: the list of info (by feature vector)
    :param sensors: the list of channel
    :param feature_set: the main set of feature vector
    :param fs: sampling frequency
    :param last_window: if you want to apply a pad to the last sub window (optional with len_sub_window)
    :param window: apply Blackman (optional)
    :param pad: apply pad (optional)
    :param greater_freq: keep for feature vector the then greater frequencies (optional)
    :param num_freq: number of frequency to keep (optional with greater_freq)
    :param drop: the number of initial frequency to drop out (optional with greater_freq)
    :param columns: list of other data to average with (optional)
    :return: feature vectors by sub answer
    """
    number_windows = len(answer) / len_sub_window

    if last_window and number_windows > int(number_windows):
        number_windows = number_windows + 1

    windows = [answer[len_sub_window * k:len_sub_window * (k + 1)] for k in range(int(number_windows))]

    feature_vectors = list()

    for index_sub in range(len(windows)):
        feature_vectors[index_sub], class_answer, info_answer = generate_feature_vector(windows[index_sub],
                                                                                        class_answer, info_answer,
                                                                                        sensors, feature_set, fs,
                                                                                        columns, window, pad,
                                                                                        greater_freq, num_freq, drop)
        for f_vector in feature_vectors:
            if len(feature_set) == 0:  # initialize
                feature_set = np.empty((0, (len(f_vector))))
            feature_set = np.vstack((feature_set, f_vector))

    return feature_set, class_answer, info_answer


def generate_feature_vector(answer, class_answer, info_answer, sensors, feature_set, fs=256, columns=None, window=False,
                            pad=False, greater_freq=False, num_freq=10, drop=1):
    """
    Generate a feature vector of one signal
    :param answer: part of the dataset coinciding with the response
    :param class_answer: the list of classes (by feature vector)
    :param info_answer: the list of info (by feature vector)
    :param sensors: the list of channel
    :param feature_set: the main set of feature vector
    :param fs: sampling frequency
    :param window: apply Blackman (optional)
    :param pad: apply pad (optional)
    :param greater_freq: keep for feature vector the then greater frequencies (optional)
    :param num_freq: number of frequency to keep (optional with greater_freq)
    :param drop: the number of initial frequency to drop out (optional with greater_freq)
    :param columns: list of other data to average with (optional)
    :return: feature vectors by answer
    """
    feature_vector_all_sensors = np.array([])

    class_answer.append(answer['label'].iloc[0])

    if 'indicator' in answer.columns:
        info_answer.append((answer['indicator'].iloc[0], answer['Image'].iloc[0]))

    for sensor in sensors:
        signal = np.array(answer[sensor], dtype=float)

        if window:
            window = np.blackman(signal.size)
            signal = signal * window

        if pad:
            signal = pad_to(signal, 8, fs)

        fft = np.abs(np.fft.rfft(signal))

        if greater_freq:
            feature_vector = greater_frequencies(fft, signal.size / 256, num_freq, drop)
        else:
            feature_vector = fft

        feature_vector_all_sensors = np.append(feature_vector_all_sensors, feature_vector)

    if columns is not None:
        feature_vector_final = list(list(answer[columns].mean())) + list(feature_vector_all_sensors)
    else:
        feature_vector_final = feature_vector_all_sensors

    if len(feature_set) == 0:  # initial
        feature_set = np.empty((0, len(feature_vector_final)))

    feature_set = np.vstack((feature_set, feature_vector_final))

    return feature_set, class_answer, info_answer


