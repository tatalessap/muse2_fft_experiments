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

java_start(path_packages)
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

java_stop()

path_folder_save_results = 'results_'+name+'/voting'
check_folder_or_create(path_folder_save_results)
voting(root_folder+'/B_one_sequently/prediction', path_folder_save_results)

