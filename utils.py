import os
import sys

import numpy as np

def signal_sub(signal):
    """
    Generate signals divided into subsignals in equal measure, if the last subsignal does not reach this measure is discarded.
    :param signal: signal to be divided into sub-windows
    :return: sub windows
    """
    windows_size = 128
    windows = [signal[windows_size * k:windows_size * (k + 1)] for k in range(int(signal.size / windows_size))]
    return windows


def check_folder_or_create(name_folder):
    path = ''
    for folder in name_folder.split('/'):
        path = path + folder + '/'
        if not os.path.isdir(path):
            os.mkdir(path)


def numbers_max(k, x):
    indices = []
    values = []
    while k != 0:
        max_el = np.amax(x)
        max_ind = int(np.where(x == max_el)[0][0])
        indices.append(max_ind)
        values.append(max_el)
        x[max_ind] = np.amin(x)
        k = k - 1
    return indices, values


def pad_to(sub_signal, seconds, fs):
    i = 0
    new_signal = []
    size_signal = seconds * fs
    for j in range(size_signal):
        if i >= len(sub_signal):
            new_signal.append(0)
        else:
            new_signal.append(sub_signal[i])
        i = i + 1
    signal = np.array(new_signal)
    return signal


def print_info(csv, name):
    sys.stdout = open(name+'_'+".txt", "w")
    c = [x for _, x in csv.groupby(['status'])]
    print("Totale:\n" + str((csv[['indicator', 'image']].groupby(['indicator']).count())))
    print('\n')
    print("Scorrette:\n" + str((c[0][['indicator', 'status']].groupby(['indicator']).count())))
    print("Scorrette:\n" + str((c[0][['original label', 'status']].groupby(['original label']).count())))
    print('\n')
    print("Corrette:\n" + str((c[1][['indicator', 'status']].groupby(['indicator']).count())))
    print("Corrette:\n" + str((c[1][['original label', 'status']].groupby(['original label']).count())))
    print("Hello World")
    sys.stdout.close()
