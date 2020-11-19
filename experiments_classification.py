from classification import *


def experiment_A(root_folder, path_features):
    """
    A Utilizzo di tutti i dataset insieme, con 5 fold cross validation random
    """
    path_folder_save_results = root_folder + '/A_one_random'

    check_folder_or_create(path_folder_save_results)

    sub = 'OneFileRandom'

    # first experiment
    op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
    classifier = "weka.classifiers.trees.RandomForest"
    experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'RandomForest' + sub)

    # second experiment
    op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
    classifier = "weka.classifiers.functions.SMO"
    experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'SMO' + sub)

    # third experiment
    op = ''
    classifier = "weka.classifiers.bayes.NaiveBayes"
    experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'NaiveBayes' + sub)

    # fourth experiment
    classifier = 'weka.classifiers.trees.RandomTree'
    op = '-K 0 -M 1.0 -V 0.001 -S 1'
    experiment_file_random(path_features, path_folder_save_results, op, classifier, 5, 3, 'RandomTree' + sub)


def experiment_B(root_folder, path_features, path_info, path_indices, voting_check=False):
    """
    B Utilizzo di tutti i dataset insieme, con 5 fold cross validation sequenziale ( ogni dataset corrisponde ad un fold)
    """
    info_table = pd.read_csv(path_info)
    indicator = list(info_table['indicator'])
    images = list(info_table['image'])

    path_folder_save_results = root_folder + '/B_one_sequently'
    sub = 'OneFileSeq'
    check_folder_or_create(path_folder_save_results)

    # first experiment
    op = "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
    classifier = "weka.classifiers.trees.RandomForest"

    experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier,
                               'RandomForest' + sub,
                               indicator, images)

    # second experiment
    op = "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
    classifier = "weka.classifiers.functions.SMO"
    experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier, 'SMO' + sub,
                               indicator, images)

    # third experiment
    op = ''
    classifier = "weka.classifiers.bayes.NaiveBayes"
    experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier,
                               'NaiveBayes' + sub,
                               indicator, images)

    # fourth experiment
    classifier = 'weka.classifiers.trees.RandomTree'
    op = '-K 0 -M 1.0 -V 0.001 -S 1'
    experiment_sequential_file(path_indices, path_features, path_folder_save_results, op, classifier,
                               'RandomTree' + sub,
                               indicator, images)

    if voting_check:
        path_folder_save_results_voting = path_folder_save_results + '/voting2'
        check_folder_or_create(path_folder_save_results_voting)
        voting_version_check_len(path_folder_save_results + '/prediction', path_folder_save_results_voting)

        path_folder_save_results_voting = path_folder_save_results + '/votingk'
        check_folder_or_create(path_folder_save_results_voting)
        voting(path_folder_save_results + '/prediction', path_folder_save_results_voting)


def experiment_C(root_folder, path_files):
    """
    C Utilizzo dei dataset in modo separato, con 5 fold cross-validation random
    """
    path_folder_save_results = root_folder + '/C_more_random'
    sub = 'MoreFileRandom'
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


def experiment_D(root_folder, path_files, voting_check=False):
    path_folder_save_results = root_folder + '/D_more_sequently'
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

    if voting_check:
        path_folder_save_results = root_folder + '/D_more_sequently'

        dir_res = os.listdir(path_folder_save_results)

        dir_to_check = []

        for d in dir_res:
            if '.csv' not in d and 'Seq' in d:
                dir_to_check.append(d)

        for d in dir_to_check:
            path_folder_s = path_folder_save_results + '/voting2/' + d
            check_folder_or_create(path_folder_s)

            prediction = path_folder_save_results + '/' + d + '/' + 'prediction'

            voting_version_check_len(prediction, path_folder_s)

        for d in dir_to_check:
            path_folder_s = path_folder_save_results + '/voting_check/' + d
            check_folder_or_create(path_folder_s)

            prediction = path_folder_save_results + '/' + d + '/' + 'prediction'
            voting(prediction, path_folder_s)
