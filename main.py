from generate_features import *
from prepocessing import *
from experiments_classification import *

sensors = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

columns = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
       'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 'Alpha_TP9',
       'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10', 'Beta_TP9', 'Beta_AF7',
       'Beta_AF8', 'Beta_TP10', 'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8',
       'Gamma_TP10']

"""
Pre-processing
"""
# alignment('muse2_file/muse', 'muse2_file/note', 'muse2_file/alignment', columns+sensors)

# labeling('muse2_file/alignment', 'muse2_file/labeled')

path_to_file = 'muse2_file/labeled/'

root = 'feature'

name_experiment = 'prova'

folder_experiments = root + '/' + name_experiment

to_save = folder_experiments + '/feature_single/'

check_folder_or_create(to_save)

"""
(path_to_file, to_save, sensors, columns=None, len_sub_window=0, fs=256, last_window=False, window=False, pad=False, 
greater_freq=False, num_freq=10, drop=1):
"""

"""
Feature extraction
"""
generate_feature_sets_by_one_file(path_to_file, to_save, sensors, columns, pad=True, greater_freq=True, num_freq=10, drop=1)

to_save = folder_experiments + '/feature_all/'

check_folder_or_create(to_save)

# generate_feature_set_by_all_files(path_to_file, to_save, sensors, columns, pad=True, greater_freq=True, num_freq=10, drop=1)

"""
Classification
"""
root_folder = 'results/'+name_experiment
check_folder_or_create(root_folder)

path_features = folder_experiments + '/feature_all/' + 'features.csv'
path_info = folder_experiments + '/feature_all/indicator/' + '_indicator.csv'
path_indices = 'index.npz'

path_packages = '/home/tatalessap/wekafiles/packages'

java_start(path_packages)
#experiment_B(root_folder, path_features, path_info, path_indices)


path_features = folder_experiments + '/feature_single'
experiment_D(root_folder, path_features)
java_stop()
