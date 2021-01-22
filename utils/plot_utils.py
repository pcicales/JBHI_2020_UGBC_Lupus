import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def uncertainty_density_plot(y, y_pred, y_var, save_name):
    plt.hist(y_var, bins=50, color='gray')
    plt.hist(y_var[y == y_pred], bins=50, color='forestgreen')
    plt.hist(y_var[y != y_pred], bins=50, color='tomato')

    # sns.kdeplot(y_var[y == y_pred], shade=True, color='forestgreen')
    # sns.kdeplot(y_var[y != y_pred], shade=True, color='tomato')
    plt.savefig(save_name + '.png')
    plt.savefig(save_name + '.svg')
    plt.savefig(save_name + '.pdf')
    plt.close()


def class_based_density_plot(y, y_pred, y_var, save_name, num_cls=7):
    for i in range(num_cls):
        y_c = y[y == i]
        y_var_c = y_var[y == i]
        y_pred_c = y_pred[y == i]
        sns.kdeplot(y_var_c[y_c == y_pred_c], shade=True, color='forestgreen')
        sns.kdeplot(y_var_c[y_c != y_pred_c], shade=True, color='tomato')
        plt.savefig(save_name + str(i) + '.png')
        plt.savefig(save_name + str(i) + '.svg')
        plt.savefig(save_name + str(i) + '.pdf')
        plt.close()


def patient_level_accuracy(y, y_pred, name, t=None):
    unq_name = np.unique(name)
    total_subs = unq_name.shape[0]
    correct_subs = 0
    for sub in unq_name:
        sub_idx = np.where(name == sub)
        y_sub, y_pred_sub = y[sub_idx], y_pred[sub_idx]
        patient_y, patient_pred = stats.mode(y_sub)[0], stats.mode(y_pred_sub)[0]
        if patient_pred == patient_y:
            correct_subs += 1
        # else:
        #     print('{} is misclassified at {}, total_subs={}'.format(sub, t, len(unq_name)))
    return (correct_subs * 1.0) / total_subs, len(unq_name)


def normalized_uncertainty_toleration_removal(y, y_pred, y_var, num_points, name):

    save_dir = '/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/removal_analysis_data/unc_tol_new'

    acc_uncertainty, acc_patient = np.array([]), np.array([])
    num_cls = len(np.unique(y))
    # y_var = (y_var - y_var.min()) / (y_var.max() - y_var.min())
    per_class_remain_count = np.zeros((num_points, num_cls))
    per_class_acc = np.zeros((num_points, num_cls))
    thresholds = np.linspace(0.05, 1, num_points)
    remain_samples = []
    for i, t in enumerate(thresholds):
        if t == 0:
            t += 0.01
        idx = np.argwhere(y_var >= t)
        y_temp = np.delete(y, idx)
        remain_samples.append(len(y_temp))
        y_pred_temp = np.delete(y_pred, idx)
        name_temp = np.delete(name, idx)
        y_var_temp = np.delete(y_var, idx)
        # img_temp = np.delete(img, idx, axis=0)

        acc_uncertainty = np.append(acc_uncertainty, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        if len(y_temp):
            p_acc, num_patient = patient_level_accuracy(y_temp, y_pred_temp, name_temp, t)
            acc_patient = np.append(acc_patient, p_acc)

            per_class_remain_count[i, :] = np.array([len(y_temp[y_temp == c]) for c in range(num_cls)])
            per_class_acc[i, :] = np.array(
                [np.sum(y_temp[y_temp == c] == y_pred_temp[y_temp == c]) / y_temp[y_temp == c].shape[0] for c in
                 range(num_cls)])

            # h5f = h5py.File(os.path.join(save_dir, 'data_{}_{}.h5'.format(fold_num, t)), 'w')
            # h5f.create_dataset('t', data=t)
            # # h5f.create_dataset('x', data=img_temp)
            # h5f.create_dataset('glum_acc', data=np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
            # h5f.create_dataset('patient_acc', data=p_acc)
            # h5f.create_dataset('num_patient', data=num_patient)
            # h5f.create_dataset('name', data=name_temp.astype(int))
            # h5f.create_dataset('y', data=y_temp)
            # h5f.create_dataset('y_pred', data=y_pred_temp)
            # h5f.create_dataset('y_var', data=y_var_temp)
            # h5f.close()



    return acc_uncertainty, np.reshape(per_class_acc, (1, num_points, num_cls)), thresholds

    # plt.figure()
    # plt.plot(thresholds, acc_uncertainty, lw=1.5, color='royalblue', marker='o', markersize=4)
    # plt.plot(thresholds, acc_patient, lw=1.5, color='green', marker='o', markersize=4)
    # plt.xlabel('Normalized Tolerated Model Uncertainty')
    # plt.ylabel('Prediction Accuracy')
    # plt.savefig('toleration_removal.png')


def uncertainty_fraction_removal(y, y_pred, y_var, num_fracs, num_random_reps, name):
    fractions = np.linspace(0.1, 1, num_fracs)
    num_samples = y.shape[0]
    acc_unc_sort = np.array([])
    # acc_pred_sort = np.array([])
    acc_random_frac = np.zeros((0, num_fracs))
    acc_patient = np.array([])

    save_dir = '/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/removal_analysis_data/data_frac_new'

    num_cls = len(np.unique(y))
    per_class_remain_count = np.zeros((num_fracs, num_cls))
    per_class_acc = np.zeros((num_fracs, num_cls))

    remain_samples = []
    # uncertainty-based removal
    inds = y_var.argsort()
    y_sorted = y[inds]
    y_pred_sorted = y_pred[inds]
    name_sorted = name[inds]
    y_var_sorted = y_var[inds]
    for i, frac in enumerate(fractions):
        y_temp = y_sorted[:int(num_samples * frac)]
        remain_samples.append(len(y_temp))
        y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
        name_temp = name_sorted[:int(num_samples * frac)]
        y_var_temp = y_var_sorted[:int(num_samples * frac)]

        acc_unc_sort = np.append(acc_unc_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

        p_acc, num_patient = patient_level_accuracy(y_temp, y_pred_temp, name_temp, frac)
        acc_patient = np.append(acc_patient, p_acc)

        per_class_remain_count[i, :] = np.array([len(y_temp[y_temp == c]) for c in range(num_cls)])
        per_class_acc[i, :] = np.array(
            [np.sum(y_temp[y_temp == c] == y_pred_temp[y_temp == c]) / y_temp[y_temp == c].shape[0] for c in
             range(num_cls)])

        # h5f = h5py.File(os.path.join(save_dir, 'data_{}_{}.h5'.format(fold_num, frac)), 'w')
        # h5f.create_dataset('glum_acc', data=np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        # h5f.create_dataset('f', data=frac)
        # h5f.create_dataset('patient_acc', data=p_acc)
        # h5f.create_dataset('num_patient', data=num_patient)
        # # h5f.create_dataset('x', data=img_temp)
        # h5f.create_dataset('name', data=name_temp.astype(int))
        # h5f.create_dataset('y', data=y_temp)
        # h5f.create_dataset('y_pred', data=y_pred_temp)
        # h5f.create_dataset('y_var', data=y_var_temp)
        # h5f.close()

    # prediction-based removal
    # inds = y_prob.argsort()[::-1]
    # y_sorted = y[inds]
    # y_pred_sorted = y_pred[inds]
    # for frac in fractions:
    #     y_temp = y_sorted[:int(num_samples * frac)]
    #     y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
    #     acc_pred_sort = np.append(acc_pred_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

    # random removal
    for rep in range(num_random_reps):
        acc_random_sort = np.array([])
        perm = np.random.permutation(y_var.shape[0])
        y_sorted = y[perm]
        y_pred_sorted = y_pred[perm]
        for frac in fractions:
            y_temp = y_sorted[:int(num_samples * frac)]
            y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
            acc_random_sort = np.append(acc_random_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        acc_random_frac = np.concatenate((acc_random_frac, np.reshape(acc_random_sort, [1, -1])), axis=0)
    acc_random_m = np.mean(acc_random_frac, axis=0)
    acc_random_s = np.std(acc_random_frac, axis=0)

    return acc_unc_sort, np.reshape(per_class_acc, (1, num_fracs, num_cls)), acc_random_m, acc_random_s, fractions

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(fractions, acc_unc_sort, 'o-', lw=1.5, label='uncertainty-based', markersize=3, color='royalblue')
    # ax.plot(fractions, acc_patient, 'o-', lw=1.5, label='uncertainty-based', markersize=3, color='green')
    #
    # # ax.plot(fractions, acc_pred_sort, 'o-', lw=1.5, label='prediction-based', markersize=3, color='red')
    #
    # line1, = ax.plot(fractions, acc_random_m, 'o', lw=1, label='Random', markersize=3, color='black')
    # ax.fill_between(fractions,
    #                 acc_random_m - acc_random_s,
    #                 acc_random_m + acc_random_s,
    #                 color='black', alpha=0.3)
    # line1.set_dashes([1, 1, 1, 1])  # 2pt line, 2pt break, 10pt line, 2pt break
    #
    # ax.set_xlabel('Fraction of Retained Data')
    # ax.set_ylabel('Prediction Accuracy')
    # plt.savefig('fraction_removal.png')
