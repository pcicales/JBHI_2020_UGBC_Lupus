import h5py
import os
from config import options
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.eval_utils import predictive_entropy, mutual_info
from utils.plot_utils import normalized_uncertainty_toleration_removal, uncertainty_fraction_removal


def normalize(x):
    max_ = x.max()
    min_ = x.min()
    return (x - min_) / (max_ - min_)


# load all images and normal predictions
normal_files = glob.glob('/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/3class/normal' + '/*.h5')

normal_files.sort()

names, y_true, y_pred_normal, all_acc = np.array([]), np.array([]), np.array([]), np.array([])
imgs = np.zeros((0, 256, 256, 3))
for i, file_ in enumerate(normal_files):
    h5f = h5py.File(file_, 'r')
    # x = h5f['x'][:]
    name = h5f['name'][:]
    y = h5f['y'][:]
    y_pred = h5f['y_pred'][:]
    h5f.close()

    # imgs = np.concatenate((imgs, x), axis=0)
    y_true = np.append(y_true, y)
    y_pred_normal = np.append(y_pred_normal, y_pred)
    names = np.append(names, name)

    acc = 100 * (np.sum(np.equal(y, y_pred)) / y.shape[0])
    all_acc = np.append(all_acc, acc)

# non-Bayesian accuracy
# all_acc = np.array([68.42596917, 81.54210215, 85.5570161 , 83.54772114, 83.39292099])
acc_mean = np.repeat(all_acc.mean(), 100)
acc_std = all_acc.std() / 2



# Bayesian Network
all_files = glob.glob('/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/3class/0.2' + '/*.h5')

all_y = []
all_mc_probs = []
all_mc_acc = [[] for _ in range(len(all_files))]

# compute the accuracy of each fold for various mc_iter
for fold_num, file_ in enumerate(all_files):
    h5f = h5py.File(file_, 'r')
    y = h5f['y'][:]
    mc_probs = h5f['mc_probs'][:]
    h5f.close()

    for iter_num in range(100):
        mean_prob = mc_probs[0:iter_num+1].mean(axis=0)
        all_mc_acc[fold_num].append(1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), y.argmax(axis=1))) / mean_prob.shape[0])

    all_y.append(y)
    all_mc_probs.append(mc_probs)

# compute acc for MC sampling
all_mc_acc = 100*np.array(all_mc_acc)
mc_acc_mean = all_mc_acc.mean(axis=0)
mc_acc_std = all_mc_acc.std(axis=0) / 2

# acc_plot
# mc_iters = np.arange(1, 101)
# plt.figure()
# plt.plot(mc_iters, acc_mean, lw=1.5, color='orange')
# plt.fill_between(mc_iters, acc_mean - acc_std/2, acc_mean + acc_std/2, color='orange', alpha=0.3)
# # plt.ylim([10, 20])
# plt.plot(mc_iters, mc_acc_mean, lw=1.5, color='royalblue', marker='o', markersize=4)
# plt.fill_between(mc_iters, mc_acc_mean - mc_acc_std/2, mc_acc_mean + mc_acc_std/2, color='royalblue', alpha=0.3)
# plt.xlabel('MC iterations')
# plt.ylabel('Prediction error (%)')
# plt.show()

opt_iter = mc_acc_mean.argmax()
y = np.concatenate(all_y)   # [N, C]
y_probs = np.concatenate(all_mc_probs, axis=1)[:opt_iter]    # [opt_iter, N, C]
mean_probs = np.mean(y_probs, axis=0)
acc_opt_iter = np.sum(np.equal(mean_probs.argmax(axis=1), y.argmax(axis=1))) / y.shape[0]

# compute uncertainties
var_pred_entropy = predictive_entropy(mean_probs)  # [N, C]
var_pred_MI = mutual_info(y_probs)

# normalize the uncertainty values
var_pred_entropy = normalize(var_pred_entropy)
var_pred_MI = normalize(var_pred_MI)

y_var = var_pred_entropy

normalized_uncertainty_toleration_removal(y.argmax(axis=1), mean_probs.argmax(axis=1), y_var, num_points=20)
uncertainty_fraction_removal(y.argmax(axis=1), mean_probs.argmax(axis=1), y_var, mean_probs.max(axis=1), num_fracs=20, num_random_reps=20)

print()
