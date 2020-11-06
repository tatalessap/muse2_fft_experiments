from fft_generate_features import *
from prepocessing import *
from utils import *

sensors = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

columns = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
       'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 'Alpha_TP9',
       'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10', 'Beta_TP9', 'Beta_AF7',
       'Beta_AF8', 'Beta_TP10', 'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8',
       'Gamma_TP10']

# alignment('muse2_file/muse', 'muse2_file/note', 'muse2_file/alignment', columns+sensors)

# labeling('muse2_file/alignment', 'muse2_file/labeled')


path_to_file = 'muse2_file/labeled/'

name = 'sub_window_fft_freq_waves'

root_folder = 'feature_single' + name

check_folder_or_create(root_folder)
name_to_save = root_folder+'/feature_by_file'
generate_feature_sets_by_one_file(path_to_file, name_to_save, sensors, columns, generate_feature_vectors_by_data_greater_freq)

root_folder = 'feature' + name

check_folder_or_create(root_folder)
name_to_save = root_folder + '/feature'
generate_feature_set_by_all_files(path_to_file, name_to_save, sensors, columns, generate_feature_vectors_by_data_greater_freq)



