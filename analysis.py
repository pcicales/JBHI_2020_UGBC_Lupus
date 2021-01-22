import glob
import h5py
import numpy as np
from utils.eval_utils import predictive_entropy, mutual_info
from utils.plot_utils import normalized_uncertainty_toleration_removal, uncertainty_fraction_removal
import matplotlib.pyplot as plt

normal_files = glob.glob(
    '/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/3class/normal' + '/*.h5')
normal_files.sort()

mc_files = glob.glob('/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/3class/0.2' + '/*.h5')
mc_files.sort()


def normalize(x):
    max_ = x.max()
    min_ = x.min()
    return (x - min_) / (max_ - min_)


# regular: array([68.42, 81.54, 85.56 , 83.55, 83.39])

# array[(29, 48, 67, 23, 28)]
# array([72.02, 79.33, 86.49, 83.75, 82.79])

# read the images, targets, names and predictions for this fold

y_var_all = []
y_all = []
y_pred_all = []
name_all = []
x_all = np.zeros((0, 256, 256, 3))

acc_unc_t, per_class_acc_t = [], []
acc_unc_f, per_class_acc_f = [], []

num_points = 20

for fold_num in range(5):
    h5f = h5py.File(mc_files[fold_num], 'r')
    y = h5f['y'][:]  # [N, C]
    mc_probs = h5f['mc_probs'][:]  # [100, N, C]
    h5f.close()

    h5f = h5py.File(normal_files[fold_num], 'r')
    # x = h5f['x'][:]
    name = h5f['name'][:]  # [N]
    y_pred = h5f['y_pred'][:]  # [N]   Regular prediction, not Bayesian
    h5f.close()


    idx = None
    if fold_num == 0:
        idx = np.where(name == 100720190523)[0]
    if fold_num == 1:
        idx = np.where(name == 100720190833)[0]
    if fold_num == 3:
        idx = np.where(name == 100720190705)[0]
    print()

    if idx is not None:
        name = np.delete(name, idx)
        y = np.delete(y, idx, axis=0)
        y_pred = np.delete(y_pred, idx, axis=0)
        mc_probs = np.delete(mc_probs, idx, axis=1)


    all_mc_acc = []
    for iter_num in range(100):
        mean_prob = mc_probs[0:iter_num + 1].mean(axis=0)
        all_mc_acc.append(
            1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), y.argmax(axis=1))) / mean_prob.shape[0])

    opt_iter = np.argmax(np.array(all_mc_acc))
    # print('Maximum accuracy is {0:.02%} for fold {1} happens at T={2}'.format(all_mc_acc[opt_iter], fold_num, opt_iter))

    # get the values on T
    y_probs = mc_probs[:opt_iter]  # [opt_iter, N, C]
    mean_probs = y_probs.mean(axis=0)

    # compute uncertainties
    var_pred_entropy = predictive_entropy(mean_probs)  # [N, C]
    # var_pred_MI = mutual_info(y_probs)

    # normalize the uncertainty values
    y_var = normalize(var_pred_entropy)

    # patient_acc = patient_level_accuracy(y.argmax(axis=1), mean_probs.argmax(axis=1), name)
    # patient_acc = patient_level_accuracy(y.argmax(axis=1), y_pred, name)

    # x_all = np.concatenate((x_all, x), axis=0)
    y_all = np.append(y_all, y.argmax(axis=1))
    y_var_all = np.append(y_var_all, y_var)
    y_pred_all = np.append(y_pred_all, mean_probs.argmax(axis=1))
    name_all = np.append(name_all, name)


    acc_uncertainty, per_class_acc, thresholds = normalized_uncertainty_toleration_removal(
        y.argmax(axis=1), mean_probs.argmax(axis=1), y_var,
        num_points=num_points, name=name)
    #
    acc_unc_t.append(acc_uncertainty)
    per_class_acc_t.append(per_class_acc)
    #
    acc_unc, per_class_acc, acc_random_m, acc_random_s, fractions = uncertainty_fraction_removal(
        y.argmax(axis=1), mean_probs.argmax(axis=1), y_var,
        num_fracs=num_points, num_random_reps=40, name=name)
    #
    acc_unc_f.append(acc_unc)
    per_class_acc_f.append(per_class_acc)

print()
#
# concatenations for uncertainty toleration removal
fold_t_acc = np.reshape(np.concatenate(acc_unc_t), (5, num_points))
mean_t_acc = np.mean(fold_t_acc, axis=0)
std_t_acc = np.std(fold_t_acc, axis=0)

fold_t_acc_percls = np.concatenate(per_class_acc_t)
mean_t_acc_percls = np.mean(fold_t_acc_percls, axis=0)
std_t_acc_percls = np.std(fold_t_acc_percls, axis=0) / 2


# concatenations for uncertainty fraction removal
fold_f_acc = np.reshape(np.concatenate(acc_unc_f), (5, num_points))
mean_f_acc = np.mean(fold_f_acc, axis=0)
std_f_acc = np.std(fold_f_acc, axis=0)

fold_f_acc_percls = np.concatenate(per_class_acc_f)
mean_f_acc_percls = np.mean(fold_f_acc_percls, axis=0)
std_f_acc_percls = np.std(fold_f_acc_percls, axis=0) / 2

fig, axs = plt.subplots(nrows=1, ncols=2)
ax = axs[0]

ax.grid(color='black', alpha=0.1, linestyle='-', linewidth=1)

ax.plot(fractions, mean_f_acc_percls[:, 0], '--', lw=1, color='magenta', markersize=2, alpha=0.7, label='control')
ax.fill_between(fractions, mean_f_acc_percls[:, 0] - std_f_acc_percls[:, 0] / 4,
                mean_f_acc_percls[:, 0] + std_f_acc_percls[:, 0] / 2,
                color='magenta', alpha=0.1)

ax.plot(fractions, mean_f_acc_percls[:, 1], '--', lw=1, color='mediumorchid', markersize=2, alpha=0.7, label='mild')
ax.fill_between(fractions, mean_f_acc_percls[:, 1] - std_f_acc_percls[:, 1] / 2,
                mean_f_acc_percls[:, 1] + std_f_acc_percls[:, 1] / 2,
                color='mediumorchid', alpha=0.1)

ax.plot(fractions, mean_f_acc_percls[:, 2], '--', lw=1, color='darkviolet', markersize=2, alpha=0.7, label='severe')
ax.fill_between(fractions, mean_f_acc_percls[:, 2] - std_f_acc_percls[:, 2] / 2,
                mean_f_acc_percls[:, 2] + std_f_acc_percls[:, 2] / 2,
                color='darkviolet', alpha=0.1)

ax.plot(fractions, mean_f_acc, lw=2, color='darkslateblue', marker='o', markersize=5, label='average')
ax.fill_between(fractions, mean_f_acc - std_f_acc / 2, mean_f_acc + std_f_acc / 2, color='darkslateblue', alpha=0.5)


line1, = ax.plot(fractions, acc_random_m, 'o', lw=1, label='Random', markersize=3, color='black')
ax.fill_between(fractions,
                acc_random_m - acc_random_s,
                acc_random_m + acc_random_s,
                color='black', alpha=0.3)
line1.set_dashes([1, 1, 1, 1])  # 2pt line, 2pt break, 10pt line, 2pt break

ax.legend()

ax.set_xlabel('Fraction of Retained Data')
ax.set_ylabel('Prediction Accuracy')

ax = axs[1]

ax.grid(color='black', alpha=0.1, linestyle='-', linewidth=1)

ax.plot(thresholds, mean_t_acc_percls[:, 0], '--', lw=1, color='magenta', markersize=2, alpha=0.7, label='control')
ax.fill_between(thresholds, mean_t_acc_percls[:, 0] - std_t_acc_percls[:, 0] / 4,
                mean_t_acc_percls[:, 0] + std_t_acc_percls[:, 0] / 2,
                color='magenta', alpha=0.1)

ax.plot(thresholds, mean_t_acc_percls[:, 1], '--', lw=1, color='mediumorchid', markersize=2, alpha=0.7, label='mild')
ax.fill_between(thresholds, mean_t_acc_percls[:, 1] - std_t_acc_percls[:, 1] / 2,
                mean_t_acc_percls[:, 1] + std_t_acc_percls[:, 1] / 2,
                color='mediumorchid', alpha=0.1)

ax.plot(thresholds, mean_t_acc_percls[:, 2], '--', lw=1, color='darkviolet', markersize=2, alpha=0.7, label='severe')
ax.fill_between(thresholds, mean_t_acc_percls[:, 2] - std_t_acc_percls[:, 2] / 2,
                mean_t_acc_percls[:, 2] + std_t_acc_percls[:, 2] / 2,
                color='darkviolet', alpha=0.1)

ax.plot(thresholds, mean_t_acc, lw=2, color='darkslateblue', marker='o', markersize=5, label='average')
ax.fill_between(thresholds, mean_t_acc - std_t_acc / 2, mean_t_acc + std_t_acc / 2, color='darkslateblue', alpha=0.5)

ax.legend()

ax.set_xlabel('Normalized Tolerated Model Uncertainty')
ax.set_ylabel('Prediction Accuracy')

width = 10
height = 4
fig.set_size_inches(width, height)
# plt.show()
plt.savefig('removal_fig.png')
plt.savefig('removal_fig.svg')








# uncertainty_fraction_removal(y.argmax(axis=1), mean_probs.argmax(axis=1), y_var,
#                              num_fracs=20, num_random_reps=20, name=name)



# normalized_uncertainty_toleration_removal(y_all, y_pred_all, y_var_all, num_points=50, name=name_all)


# uncertainty_fraction_removal(y_all, y_pred_all, y_var_all, num_fracs=50, num_random_reps=20, name=name_all)



# h5f = h5py.File('/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/uncertainty_values.h5', 'w')
# for i in range(5):
#     h5f.create_dataset('y_var_fold_{}'.format(i), data=np.array(y_var[i]))
# h5f.close()



print()
