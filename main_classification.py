from classification import *
from utils import *

path_packages = '/home/tatalessap/wekafiles/packages'

path_indices = 'index.npz'

path_features = 'feature_fft/feature.csv'

java_start(path_packages)

root_folder = 'results'

check_folder_or_create(root_folder)

"""
A Utilizzo di tutti i dataset insieme, con 5 fold cross validation random
"""
path_folder_save_results = root_folder+'/A_one_random'

check_folder_or_create(path_folder_save_results)

sub = 'OneFileRandom'

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"
experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'RandomForest' + sub)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
#experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'SMO' + sub)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'NaiveBayes' + sub)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'RandomTree' + sub)

"""
B Utilizzo di tutti i dataset insieme, con 5 fold cross validation sequenziale ( ogni dataset corrisponde ad un fold)
"""
try:
    d = pd.read_csv('features/feature_indicator.csv')
    col = list(d['indicator'])
except:
    print('indicator not exist')
    col = []

path_folder_save_results = root_folder+'/B_one_sequently'
sub = 'OneFileSeq'
check_folder_or_create(path_folder_save_results)

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"

experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'RandomForest' + sub, col)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
#experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'SMO' + sub, col)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'NaiveBayes' + sub, col)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'RandomTree' + sub, col)

"""
C Utilizzo dei dataset in modo separato, con 5 fold cross-validation random
"""
path_folder_save_results = root_folder+'/C_more_random'
sub = 'MoreFileRandom'
path_files = 'feature_fft_single'
check_folder_or_create(path_folder_save_results)

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'RandomForest' + sub)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'SMO' + sub)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'NaiveBayes' + sub)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'RandomTree' + sub)

"""
D Utilizzo dei dataset in modo separato, con 5 fold cross-validation sequenziale
"""
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
