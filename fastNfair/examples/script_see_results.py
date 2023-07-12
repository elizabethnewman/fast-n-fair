import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dir_name = 'unfair_2d_results/'

file_name = 'robust--trust--r_'

2d_average_times = {}
2d_average_times['trust'] = []
2d_average_times['pgd'] = []
2d_average_times['rand'] = []
lsat_average_times = {}
adult_average_times = {}
lsat_average_times['trust'] = []
lsat_average_times['pgd'] = []
lsat_average_times['rand'] = []
adult_average_times['trust'] = []
adult_average_times['pgd'] = []
adult_average_times['rand'] = []

# Variation of fairness metrics and accuracy
# with radius for unfair 2d example with
# trust region subproblem and PGD optimizers
unfair_2d_data = []
train_indy0_2d = []
train_indy1_2d = []
test_indy0_2d = []
test_indy1_2d = []
train_sepy0_2d = []
train_sepy1_2d = []
test_sepy0_2d = []
test_sepy1_2d = []
train_sufy0_2d = []
train_sufy1_2d = []
test_sufy0_2d = []
test_sufy1_2d = []
radii_2d = []
train_accuracies_2d = []
test_accuracies_2d = []
for n in range(11):
    r = 0.1 + 0.01 * n
    radii_2d.append(r)

    obj = pickle.load(open(dir_name + file_name + ('%0.2e' % r) + '.pkl', "rb"))

    train_indy0_2d.append(obj['results_eval']['train']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    train_indy1_2d.append(obj['results_eval']['train']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    test_indy0_2d.append(obj['results_eval']['test']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    test_indy1_2d.append(obj['results_eval']['test']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    train_sepy0_2d.append(obj['results_eval']['train']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    train_sepy1_2d.append(obj['results_eval']['train']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    test_sepy0_2d.append(obj['results_eval']['test']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    test_sepy1_2d.append(obj['results_eval']['test']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    train_sufy0_2d.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    train_sufy1_2d.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    test_sufy0_2d.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    test_sufy1_2d.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    train_accuracies_2d.append(obj['results_train']['train']['accuracy'][-1])
    test_accuracies_2d.append(obj['results_train']['test']['accuracy'])
    2d_average_times['trust'].append(np.mean(np.array(obj['results_train']['history']['values'])[:, 1]))
unfair_2d_data.append(train_indy0_2d)
unfair_2d_data.append(train_indy1_2d)
unfair_2d_data.append(test_indy0_2d)
unfair_2d_data.append(test_indy1_2d)
unfair_2d_data.append(train_sepy0_2d)
unfair_2d_data.append(train_sepy1_2d)
unfair_2d_data.append(test_sepy0_2d)
unfair_2d_data.append(test_sepy1_2d)
unfair_2d_data.append(train_sufy0_2d)
unfair_2d_data.append(train_sufy1_2d)
unfair_2d_data.append(test_sufy0_2d)
unfair_2d_data.append(test_sufy1_2d)


dir_name_2 = 'lsat_results/'

file_name = 'robust--trust--r_'

# Variation of fairness metrics and accuracy
# with radius for lsat example with
# trust region subproblem and PGD optimizers
lsat_data = []
train_indy0_lsat = []
train_indy1_lsat = []
test_indy0_lsat = []
test_indy1_lsat = []
train_sepy0_lsat = []
train_sepy1_lsat = []
test_sepy0_lsat = []
test_sepy1_lsat = []
train_sufy0_lsat = []
train_sufy1_lsat = []
test_sufy0_lsat = []
test_sufy1_lsat = []
radii_lsat = []
train_accuracies_lsat = []
test_accuracies_lsat = []
for n in range(11):
    r = 0.1 + 0.01 * n
    radii_lsat.append(r)

    obj = pickle.load(open(dir_name_2 + file_name + ('%0.2e' % r) + '.pkl', "rb"))

    train_indy0_lsat.append(obj['results_eval']['train']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    train_indy1_lsat.append(obj['results_eval']['train']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    test_indy0_lsat.append(obj['results_eval']['test']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    test_indy1_lsat.append(obj['results_eval']['test']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    train_sepy0_lsat.append(obj['results_eval']['train']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    train_sepy1_lsat.append(obj['results_eval']['train']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    test_sepy0_lsat.append(obj['results_eval']['test']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    test_sepy1_lsat.append(obj['results_eval']['test']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    train_sufy0_lsat.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    train_sufy1_lsat.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    test_sufy0_lsat.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    test_sufy1_lsat.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    train_accuracies_lsat.append(obj['results_train']['train']['accuracy'][-1])
    test_accuracies_lsat.append(obj['results_train']['test']['accuracy'])
    lsat_average_times['trust'].append(np.mean(np.array(obj['results_train']['history']['values'])[:, 1]))
lsat_data.append(train_indy0_lsat)
lsat_data.append(train_indy1_lsat)
lsat_data.append(test_indy0_lsat)
lsat_data.append(test_indy1_lsat)
lsat_data.append(train_sepy0_lsat)
lsat_data.append(train_sepy1_lsat)
lsat_data.append(test_sepy0_lsat)
lsat_data.append(test_sepy1_lsat)
lsat_data.append(train_sufy0_lsat)
lsat_data.append(train_sufy1_lsat)
lsat_data.append(test_sufy0_lsat)
lsat_data.append(test_sufy1_lsat)

dir_name_3 = 'adult_results/'

file_name = 'robust--trust--r_'

# Variation of fairness metrics and accuracy
# with radius for adult example with
# trust region subproblem and PGD optimizers
adult_data = []
train_indy0_adult = []
train_indy1_adult = []
test_indy0_adult = []
test_indy1_adult = []
train_sepy0_adult = []
train_sepy1_adult = []
test_sepy0_adult = []
test_sepy1_adult = []
train_sufy0_adult = []
train_sufy1_adult = []
test_sufy0_adult = []
test_sufy1_adult = []
radii_adult= []
train_accuracies_adult = []
test_accuracies_adult = []
average_times_adult = []
for n in range(6):
    r = 0.1 + 0.02 * n
    radii_adult.append(r)

    obj = pickle.load(open(dir_name_3 + file_name + ('%0.2e' % r) + '.pkl', "rb"))

    train_indy0_adult.append(obj['results_eval']['train']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    train_indy1_adult.append(obj['results_eval']['train']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    test_indy0_adult.append(obj['results_eval']['test']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    test_indy1_adult.append(obj['results_eval']['test']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    train_sepy0_adult.append(obj['results_eval']['train']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    train_sepy1_adult.append(obj['results_eval']['train']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    test_sepy0_adult.append(obj['results_eval']['test']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    test_sepy1_adult.append(obj['results_eval']['test']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    train_sufy0_adult.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    train_sufy1_adult.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    test_sufy0_adult.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    test_sufy1_adult.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    train_accuracies_adult.append(obj['results_train']['train']['accuracy'][-1])
    test_accuracies_adult.append(obj['results_train']['test']['accuracy'])
    adult_average_times['trust'].append(np.mean(np.array(obj['results_train']['history']['values'])[:, 1]))
adult_data.append(train_indy0_adult)
adult_data.append(train_indy1_adult)
adult_data.append(test_indy0_adult)
adult_data.append(test_indy1_adult)
adult_data.append(train_sepy0_adult)
adult_data.append(train_sepy1_adult)
adult_data.append(test_sepy0_adult)
adult_data.append(test_sepy1_adult)
adult_data.append(train_sufy0_adult)
adult_data.append(train_sufy1_adult)
adult_data.append(test_sufy0_adult)
adult_data.append(test_sufy1_adult)

file_name2 = 'robust--rand--r_'

# Variation of fairness metrics and accuracy
# with radius for unfair 2d example with
# random perturbation
unfair_2d_data_rand = []
train_indy0_2d_rand = []
train_indy1_2d_rand = []
test_indy0_2d_rand = []
test_indy1_2d_rand = []
train_sepy0_2d_rand = []
train_sepy1_2d_rand = []
test_sepy0_2d_rand = []
test_sepy1_2d_rand = []
train_sufy0_2d_rand = []
train_sufy1_2d_rand = []
test_sufy0_2d_rand = []
test_sufy1_2d_rand = []
train_accuracies_2d_rand = []
test_accuracies_2d_rand = []
for n in range(11):
    r = 0.1 + 0.01 * n

    obj = pickle.load(open(dir_name + file_name2 + ('%0.2e' % r) + '.pkl', "rb"))

    train_indy0_2d_rand.append(obj['results_eval']['train']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    train_indy1_2d_rand.append(obj['results_eval']['train']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    test_indy0_2d_rand.append(obj['results_eval']['test']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    test_indy1_2d_rand.append(obj['results_eval']['test']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    train_sepy0_2d_rand.append(obj['results_eval']['train']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    train_sepy1_2d_rand.append(obj['results_eval']['train']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    test_sepy0_2d_rand.append(obj['results_eval']['test']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    test_sepy1_2d_rand.append(obj['results_eval']['test']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    train_sufy0_2d_rand.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    train_sufy1_2d_rand.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    test_sufy0_2d_rand.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    test_sufy1_2d_rand.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    train_accuracies_2d_rand.append(obj['results_train']['train']['accuracy'][-1])
    test_accuracies_2d_rand.append(obj['results_train']['test']['accuracy'])
    2d_average_times['rand'].append(np.mean(np.array(obj['results_train']['history']['values'])[:, 1]))
unfair_2d_data_rand.append(train_indy0_2d_rand)
unfair_2d_data_rand.append(train_indy1_2d_rand)
unfair_2d_data_rand.append(test_indy0_2d_rand)
unfair_2d_data_rand.append(test_indy1_2d_rand)
unfair_2d_data_rand.append(train_sepy0_2d_rand)
unfair_2d_data_rand.append(train_sepy1_2d_rand)
unfair_2d_data_rand.append(test_sepy0_2d_rand)
unfair_2d_data_rand.append(test_sepy1_2d_rand)
unfair_2d_data_rand.append(train_sufy0_2d_rand)
unfair_2d_data_rand.append(train_sufy1_2d_rand)
unfair_2d_data_rand.append(test_sufy0_2d_rand)
unfair_2d_data_rand.append(test_sufy1_2d_rand)

# Variation of fairness metrics and accuracy
# with radius for lsat example with
# random perturbation
lsat_data_rand = []
train_indy0_lsat_rand = []
train_indy1_lsat_rand = []
test_indy0_lsat_rand = []
test_indy1_lsat_rand = []
train_sepy0_lsat_rand = []
train_sepy1_lsat_rand = []
test_sepy0_lsat_rand = []
test_sepy1_lsat_rand = []
train_sufy0_lsat_rand = []
train_sufy1_lsat_rand = []
test_sufy0_lsat_rand = []
test_sufy1_lsat_rand = []
train_accuracies_lsat_rand = []
test_accuracies_lsat_rand = []
for n in range(11):
    r = 0.1 + 0.01 * n

    obj = pickle.load(open(dir_name_2 + file_name2 + ('%0.2e' % r) + '.pkl', "rb"))
    print(dir_name_2 + file_name2 + ('%0.2e' % r) + '.pkl')
    train_indy0_lsat_rand.append(obj['results_eval']['train']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    train_indy1_lsat_rand.append(obj['results_eval']['train']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    test_indy0_lsat_rand.append(obj['results_eval']['test']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    test_indy1_lsat_rand.append(obj['results_eval']['test']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    train_sepy0_lsat_rand.append(obj['results_eval']['train']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    train_sepy1_lsat_rand.append(obj['results_eval']['train']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    test_sepy0_lsat_rand.append(obj['results_eval']['test']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    test_sepy1_lsat_rand.append(obj['results_eval']['test']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    train_sufy0_lsat_rand.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    train_sufy1_lsat_rand.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    test_sufy0_lsat_rand.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    test_sufy1_lsat_rand.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    train_accuracies_lsat_rand.append(obj['results_train']['train']['accuracy'][-1])
    test_accuracies_lsat_rand.append(obj['results_train']['test']['accuracy'])
    lsat_average_times['rand'].append(np.mean(np.array(obj['results_train']['history']['values'])[:, 1]))
lsat_data_rand.append(train_indy0_lsat_rand)
lsat_data_rand.append(train_indy1_lsat_rand)
lsat_data_rand.append(test_indy0_lsat_rand)
lsat_data_rand.append(test_indy1_lsat_rand)
lsat_data_rand.append(train_sepy0_lsat_rand)
lsat_data_rand.append(train_sepy1_lsat_rand)
lsat_data_rand.append(test_sepy0_lsat_rand)
lsat_data_rand.append(test_sepy1_lsat_rand)
lsat_data_rand.append(train_sufy0_lsat_rand)
lsat_data_rand.append(train_sufy1_lsat_rand)
lsat_data_rand.append(test_sufy0_lsat_rand)
lsat_data_rand.append(test_sufy1_lsat_rand)

# Variation of fairness metrics and accuracy
# with radius for adult example with
# random perturbation
adult_data_rand = []
train_indy0_adult_rand = []
train_indy1_adult_rand = []
test_indy0_adult_rand = []
test_indy1_adult_rand = []
train_sepy0_adult_rand = []
train_sepy1_adult_rand = []
test_sepy0_adult_rand = []
test_sepy1_adult_rand = []
train_sufy0_adult_rand = []
train_sufy1_adult_rand = []
test_sufy0_adult_rand = []
test_sufy1_adult_rand = []
train_accuracies_adult_rand = []
test_accuracies_adult_rand = []
for n in range(6):
    r = 0.1 + 0.02 * n

    obj = pickle.load(open(dir_name_3 + file_name2 + ('%0.2e' % r) + '.pkl', "rb"))

    train_indy0_adult_rand.append(obj['results_eval']['train']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    train_indy1_adult_rand.append(obj['results_eval']['train']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    test_indy0_adult_rand.append(obj['results_eval']['test']['fairness']['independence']['y = 0']['Difference y=0 (s1-s0)'])
    test_indy1_adult_rand.append(obj['results_eval']['test']['fairness']['independence']['y = 1']['Difference y=1 (s1-s0)'])
    train_sepy0_adult_rand.append(obj['results_eval']['train']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    train_sepy1_adult_rand.append(obj['results_eval']['train']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    test_sepy0_adult_rand.append(obj['results_eval']['test']['fairness']['separation']['y = 0']['Difference y=0 (s1-s0)'])
    test_sepy1_adult_rand.append(obj['results_eval']['test']['fairness']['separation']['y = 1']['Difference y=1 (s1-s0)'])
    train_sufy0_adult_rand.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    train_sufy1_adult_rand.append(obj['results_eval']['train']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    test_sufy0_adult_rand.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 0']['Difference y_pred = 0 (s1-s0)'])
    test_sufy1_adult_rand.append(obj['results_eval']['test']['fairness']['sufficiency']['y_pred = 1']['Difference y_pred = 1 (s1-s0)'])
    train_accuracies_adult_rand.append(obj['results_train']['train']['accuracy'][-1])
    test_accuracies_adult_rand.append(obj['results_train']['test']['accuracy'])
    adult_average_times['rand'].append(np.mean(np.array(obj['results_train']['history']['values'])[:, 1]))
adult_data_rand.append(train_indy0_adult_rand)
adult_data_rand.append(train_indy1_adult_rand)
adult_data_rand.append(test_indy0_adult_rand)
adult_data_rand.append(test_indy1_adult_rand)
adult_data_rand.append(train_sepy0_adult_rand)
adult_data_rand.append(train_sepy1_adult_rand)
adult_data_rand.append(test_sepy0_adult_rand)
adult_data_rand.append(test_sepy1_adult_rand)
adult_data_rand.append(train_sufy0_adult_rand)
adult_data_rand.append(train_sufy1_adult_rand)
adult_data_rand.append(test_sufy0_adult_rand)
adult_data_rand.append(test_sufy1_adult_rand)

new_filename = 'robust--pgd--r_'

for n in range(11):
    r = 0.1 + 0.01 * n
    obj = pickle.load(open(dir_name + new_filename + ('%0.2e' % r) + '.pkl', "rb"))
    2d_average_times['pgd'].append(np.mean(np.array(obj['results_train']['history']['values'])[:,1]))

for n in range(11):
    r = 0.1 + 0.01 * n
    obj = pickle.load(open(dir_name_2 + new_filename + ('%0.2e' % r) + '.pkl', "rb"))
    lsat_average_times['pgd'].append(np.mean(np.array(obj['results_train']['history']['values'])[:,1]))

for n in range(6):
    r = 0.1 + 0.01 * n
    obj = pickle.load(open(dir_name_3 + new_filename + ('%0.2e' % r) + '.pkl', "rb"))
    adult_average_times['pgd'].append(np.mean(np.array(obj['results_train']['history']['values'])[:,1]))

# plt.show()
