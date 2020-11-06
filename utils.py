import os
import numpy as np
import sys


def check_folder_or_create(name_folder):
    if not os.path.isdir(name_folder):
        os.mkdir(name_folder)


def numbers_max(k, x):
    indices = []
    values = []
    while k != 0:
        max_el = np.amax(x)
        max_ind = int(np.where(x == max_el)[0])
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
    sys.stdout = open(name+'_'+"test.txt", "w")
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