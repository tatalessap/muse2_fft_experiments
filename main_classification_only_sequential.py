from classification import *
from utils import *

path_packages = '/home/tatalessap/wekafiles/packages'

path_indices = 'index.npz'

name = 'sub_window_fft_freq_waves'

path_features = 'feature' + name + '/feature.csv'

root_folder = 'results_'+name

check_folder_or_create(root_folder)

"""
B Utilizzo di tutti i dataset insieme, con 5 fold cross validation sequenziale ( ogni dataset corrisponde ad un fold)
"""

"""

try:
    indicator_table = pd.read_csv('feature' + name + '/feature_indicator.csv')
    indicator = list(indicator_table['indicator'])
    images = list(indicator_table['image'])
except:
    print('indicator not exist')
    indicator = images = []

path_folder_save_results = root_folder + '/B_one_sequently'
sub = 'OneFileSeq'
check_folder_or_create(path_folder_save_results)

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"

experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'RandomForest' + sub,
                           indicator, images)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'SMO' + sub,
                           indicator, images)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'NaiveBayes' + sub,
                           indicator, images)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'RandomTree' + sub,
                           indicator, images)


path_folder_save_results = 'results_'+name+'/voting2'
check_folder_or_create(path_folder_save_results)
voting_version2(root_folder+'/B_one_sequently/prediction', path_folder_save_results)
"""

"""
D Utilizzo dei dataset in modo separato, con 5 fold cross-validation sequenziale
"""

java_start(path_packages)

path_files = 'feature_single' + name

path_folder_save_results = root_folder+'/D_more_sequently'

check_folder_or_create(path_folder_save_results)

sub = 'MoreFileSeq'

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'RandomForest' + sub)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'SMO' + sub)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'NaiveBayes' + sub)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'RandomTree' + sub)

java_stop()

dir_res = os.listdir(path_folder_save_results)

dir_to_check = []

for d in dir_res:
    if '.csv' not in d and 'Seq' in d:
        dir_to_check.append(d)

print(dir_to_check)

for d in dir_to_check:
    path_folder_s = path_folder_save_results + '/voting2/' + d
    check_folder_or_create(path_folder_s)

    prediction = path_folder_save_results + '/' + d + '/' + 'prediction'

    voting_version_check_len(prediction, path_folder_s)






