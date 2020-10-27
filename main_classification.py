from classification import *

path_packages = '/home/tatalessap/wekafiles/packages'

path_indices = 'index.npz'

path_features = 'features_fft/feature.csv'

java_start(path_packages)

"""
A Utilizzo di tutti i dataset insieme, con 5 fold cross validation random
"""

path_folder_save_results = 'results/A_one_random'
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
#experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'NaiveBayes' + sub)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'RandomTree' + sub)

"""
B Utilizzo di tutti i dataset insieme, con 5 fold cross validation sequenziale ( ogni dataset corrisponde ad un fold)
"""

path_folder_save_results = 'results/B_one_sequently'
sub = 'OneFileSeq'

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"

experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'RandomForest' + sub)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
#experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'SMO' + sub)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
#experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'NaiveBayes' + sub)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'RandomTree' + sub)

"""
C Utilizzo dei dataset in modo separato, con 5 fold cross-validation random
"""
path_folder_save_results = 'results/C_more_random'
sub = 'MoreFileRandom'
path_files = 'features_fft_single'

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'RandomForest' + sub)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
#experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'SMO' + sub)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
#experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'NaiveBayes' + sub)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 3, 'RandomTree' + sub)

"""
D Utilizzo dei dataset in modo separato, con 5 fold cross-validation sequenziale
"""
path_folder_save_results = 'results/D_more_sequently'
sub = 'MoreFileSeq'
path_files = 'features_fft_single'

# first experiment
op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
classifier = "weka.classifiers.trees.RandomForest"
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'RandomForest' + sub)

# second experiment
op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
classifier = "weka.classifiers.functions.SMO"
#experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'SMO' + sub)

# third experiment
op = ''
classifier = "weka.classifiers.bayes.NaiveBayes"
#experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'NaiveBayes' + sub)

# fourth experiment
classifier = 'weka.classifiers.trees.RandomTree'
op = '-K 0 -M 1.0 -V 0.001 -S 1'
experiment_more_file(path_files, path_folder_save_results, 5, op, classifier, 0, 'RandomTree' + sub)

java_stop()
