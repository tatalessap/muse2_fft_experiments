import pandas as pd
import os
import datetime


def alignment(path_muse, path_note, path_save, columns):
    files_muse = sorted(os.listdir(path_muse))
    files_notes = sorted(os.listdir(path_note))

    for index in range(len(files_muse)):
        print(index)
        # read csv
        muse = pd.read_csv(path_muse + '/' + files_muse[index], index_col=False)

        muse = muse[['TimeStamp'] + columns]
        if len(muse['TimeStamp'][0]) >= 12:  # if there is the date
            muse['TimeStamp'] = list(map(lambda x: x[11:], muse['TimeStamp']))
        note = pd.read_csv(path_note + '/' + files_notes[index], index_col=False)

        # fix tables
        first_check = note['check time'][1][:-3]
        note = note[2:-1]
        note['check time'] = list(map(lambda x: (x[:-3]), note['check time']))
        muse['TimeStamp'] = list(map(lambda x: str(x), muse['TimeStamp']))

        # fix dimension of csv muse
        print('Dimensioni Muse prima del taglio:' + str(muse.shape))
        old_shape = muse.shape[0]
        muse = muse.dropna()
        muse = muse[(muse[columns] != 0).all(1)]
        print('Dimensioni Muse dopo taglio:' + str(muse.shape))
        print('Differenza: ' + str(old_shape - muse.shape[0]))

        # muse csv must be start with the same time of note csv
        first_index = 0
        while pd.to_datetime(first_check) > pd.to_datetime(list(muse['TimeStamp'])[first_index]):
            first_index = first_index + 1
        muse = muse[first_index:]

        muse = create_new_muse_table(note, muse, first_check)

        muse.to_csv(path_save + "/" + str(index) + "alignment.csv", index=False)


def create_new_muse_table(note, muse, first_check):
    #
    eti = list(note['check time'])
    muse_t = list(muse['TimeStamp'])
    class_user = list(note['class by user'])
    class_ori = list(note['original class'])

    class_by_user = list()
    original_class = list()
    time_check = list()
    time_diff = list()
    image = list()

    # support variables
    keep_going = True
    i = j = 0

    print('start')

    while i < len(muse_t) and keep_going:
        # if time of muse is major, change answer
        if pd.to_datetime(muse_t[i]) >= pd.to_datetime(eti[j]):
            j = j + 1
        # if there is an answer in eti[j]
        if j < len(eti):
            if j == 0:
                time_diff.append(str(pd.to_datetime(eti[j]) - pd.to_datetime(first_check)))
            else:
                time_diff.append(str(pd.to_datetime(eti[j]) - pd.to_datetime(eti[j - 1])))
            class_by_user.append(class_user[j])
            original_class.append(class_ori[j])
            time_check.append(pd.to_datetime(eti[j]).time())
            image.append(j)
        # the answers are finished
        else:
            keep_going = False
        i = i + 1

    # create the new table / csv
    muse = muse[:len(class_by_user)]
    muse["Class By User"] = class_by_user
    muse["Original Class"] = original_class
    muse["Image"] = image
    muse["Time Check Button"] = time_check
    muse["Response Time"] = time_diff

    return muse


def labeling(path_file, path_save):
    files = sorted(os.listdir(path_file))
    print("File da etichettare:" + str(files))
    data_sets = [pd.read_csv(path_file + '/' + str(el)) for el in files]
    threshold = calculate_threshold(data_sets)
    print(threshold)
    for i in range(len(data_sets)):
        print('Dataset da etichettare:' + str(i))
        add_label(threshold, data_sets[i])
        data_sets[i].to_csv(path_save + "/" + str(i) + "raw_f.csv", index=False)


def rc_single(df, minutes):
    # print('\n')
    time = df['TimeStamp']
    RT = df['Response Time']
    class_us = df['Class By User']
    change = 0
    index_start = 1
    while change <= 10:
        if RT[index_start] != RT[index_start - 1] and class_us[index_start] != class_us[index_start - 1]:
            change = change + 1
        index_start = index_start + 1

    time_start = pd.to_timedelta(time[index_start])
    minutes_added = datetime.timedelta(minutes=minutes)
    time_end = time_start + minutes_added

    i = index_start

    average_RT = datetime.timedelta(hours=0, minutes=0, seconds=0)

    while pd.to_timedelta(time[i]) < time_end:
        average_RT = average_RT + pd.to_timedelta(RT[i])
        i = i + 1

    return average_RT / (i - index_start)


def add_label(threshold, df):
    labels = list()
    col_indicator = list()

    annotations = [x for _, x in df.groupby(['Image'])]  # set of response

    for answer in annotations:
        if answer["Class By User"].iloc[0] != answer["Original Class"].iloc[0] \
                and pd.to_timedelta(answer["Response Time"].iloc[0]) >= threshold:
            for i in range(len(answer)):
                labels.append('distracted')
                col_indicator.append('error-over time')
        elif answer["Class By User"].iloc[0] != answer["Original Class"].iloc[0] \
                and pd.to_timedelta(answer["Response Time"].iloc[0]) < threshold:
            for i in range(len(answer)):
                labels.append('distracted')
                col_indicator.append('error-no over time')
        elif pd.to_timedelta(answer["Response Time"].iloc[0]) >= threshold:
            for i in range(len(answer)):
                labels.append('distracted')
                col_indicator.append('over time')
        else:
            for i in range(len(answer)):
                labels.append('attentive')
                col_indicator.append('attentive')

    df['label'] = labels
    df['indicator'] = col_indicator


def calculate_threshold(data_sets):
    sum_c = datetime.timedelta(hours=0, minutes=0, seconds=0)
    for x in [rc_single(df, 5) for df in data_sets]:
        sum_c = sum_c + x
    return sum_c / len(data_sets)
