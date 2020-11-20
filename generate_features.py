from feature_extraction import *
from utils import *

def generate_feature_sets_by_one_file(path_to_file, to_save, sensors, columns=None, len_sub_window=0, fs=256,
                                      last_window=False, window=False, pad=False, greater_freq=False, num_freq=10,
                                      drop=1):
    """
    file by file feature generator
    :param path_to_file: the path of files
    :param to_save: path to save file
    :param sensors: list of sensors
    :param columns: list of another columns
    :param len_sub_window: length of the sub-window
    :param fs: sampling frequency
    :param last_window: if you want to apply a pad to the last sub window (optional with len_sub_window)
    :param window: apply Blackman (optional)
    :param pad: apply pad (optional)
    :param greater_freq: keep for feature vector the then greater frequencies (optional)
    :param num_freq: number of frequency to keep (optional with greater_freq)
    :param drop: the number of initial frequency to drop out (optional with greater_freq)
    :return:
    """

    path_to_save_info = to_save + "/indicator/"

    check_folder_or_create(path_to_save_info)

    datasets = [pd.read_csv(path_to_file + str(el)) for el in os.listdir(path_to_file)]  # sets of file raw

    for index in range(len(datasets)):
        print("Dataset number: " + str(index))
        feature_set, class_column, indicator_column = generate_feature_set(datasets[index], sensors, columns, len_sub_window, fs,
                                                                           last_window, window, pad, greater_freq,
                                                                           num_freq, drop)
        features = pd.DataFrame(feature_set)
        features['classes'] = np.array(class_column)
        features.to_csv(to_save + str(index) + 'features.csv', index=False)

        indicator = pd.DataFrame()
        indicator['indicator'] = np.array([indicator[0] for indicator in indicator_column])
        indicator['image'] = np.array([indicator[1] for indicator in indicator_column])
        indicator.to_csv(path_to_save_info + str(index) + '_indicator' + '.csv', index=False)


def generate_feature_set_by_all_files(path_to_file, to_save, sensors, columns=None, len_sub_window=0, fs=256,
                                      last_window=False, window=False, pad=False, greater_freq=False, num_freq=10,
                                      drop=1):
    """
    :param path_to_file: the path of files
    :param to_save: path to save file
    :param sensors: list of sensors
    :param columns: list of another columns
    :param len_sub_window: length of the sub-window
    :param fs: sampling frequency
    :param last_window: if you want to apply a pad to the last sub window (optional with len_sub_window)
    :param window: apply Blackman (optional)
    :param pad: apply pad (optional)
    :param greater_freq: keep for feature vector the then greater frequencies (optional)
    :param num_freq: number of frequency to keep (optional with greater_freq)
    :param drop: the number of initial frequency to drop out (optional with greater_freq)
    :return:
    """

    path_to_save_info = to_save + "/indicator/"

    check_folder_or_create(path_to_save_info)

    datasets = [pd.read_csv(path_to_file + str(el)) for el in os.listdir(path_to_file)]  # sets of file raw
    class_column_final = list()
    indicator_column_final = list()
    indexes = []
    features = pd.DataFrame()

    for index in range(len(datasets)):
        print("Dataset: " + str(index))
        feature_set, class_column, indicator_column = generate_feature_set(datasets[index], sensors, columns,
                                                                           len_sub_window, fs, last_window, window,
                                                                           pad, greater_freq, num_freq, drop)
        if index == 0:
            features = pd.DataFrame(feature_set)
        else:
            features = pd.concat([features, pd.DataFrame(feature_set)], ignore_index=True)

        indexes.append(len(class_column_final))

        class_column_final = class_column_final + class_column
        indicator_column_final = indicator_column_final + indicator_column

    features['classes'] = np.array(class_column_final)

    indicator = pd.DataFrame()
    indicator['indicator'] = np.array([indicator[0] for indicator in indicator_column_final])
    indicator['image'] = np.array([indicator[1] for indicator in indicator_column_final])
    indicator.to_csv(path_to_save_info + '_indicator' + '.csv', index=False)

    np.savez('index', np.array(indexes))
    features.to_csv(to_save + 'features.csv', index=False)
