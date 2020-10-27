from fft_generate_features import *

sensors = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

path_to_file = 'res_eti_1/'
name_to_save = 'features_fft_single/feature_by_file'
generate_feature_sets_by_one_file(path_to_file, name_to_save, sensors)

path_to_file = 'res_eti_1/'
name_to_save = 'features_fft/feature'
generate_feature_set_by_all_files(path_to_file, name_to_save, sensors)
