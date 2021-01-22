import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.eval_utils import predictive_entropy
from utils.plot_utils import uncertainty_density_plot
import os

normal_files = glob.glob(
    '/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/3class/normal' + '/*.h5')
normal_files.sort()

mc_files = glob.glob('/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/3class/0.2' + '/*.h5')
mc_files.sort()


all_dirs = glob.glob('/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/distribution_plot/*')

done_subs = [int(dir_.split('/')[-1]) for dir_ in all_dirs]


def subject_dist_plot(x, y, y_pred, y_prob, y_var, name):

    img_save_dir = '/home/cougarnet.uh.edu/pcicales/Desktop/glomerulus_classification-master/distribution_plot'
    sub_dir = os.path.join(img_save_dir, str(name))
    sub_dir_pdf = os.path.join(img_save_dir, str(name), 'PDF')
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
        os.makedirs(sub_dir_pdf)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(y_var[y == y_pred], shade=True, color='forestgreen')
    sns.kdeplot(y_var[y != y_pred], shade=True, color='tomato')
    plt.xlim([-0.2, 1.2])
    plt.savefig(os.path.join(sub_dir, 'correct_incorrect.png'))
    plt.savefig(os.path.join(sub_dir_pdf, 'correct_incorrect.svg'))
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(y_var, shade=True, color='forestgreen')
    plt.xlim([-0.2, 1.2])
    plt.savefig(os.path.join(sub_dir, 'all.png'))
    plt.savefig(os.path.join(sub_dir_pdf, 'all.svg'))
    plt.close()

    for i in range(len(y)):

        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax = axs[0]
        img = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('T: {0}, P: {1}, H: {2:.2f}'.format(y[i], y_pred[i], y_var[i]))
        ax = axs[1]
        sns.kdeplot(y_prob[:, i, 0], shade=True, color='plum')
        sns.kdeplot(y_prob[:, i, 1], shade=True, color='mediumorchid')
        sns.kdeplot(y_prob[:, i, 2], shade=True, color='indigo')
        ax.set_xlim([-0.1, 1.1])
        fig.set_size_inches(8, 4)
        plt.savefig(os.path.join(sub_dir, str(i) + '.png'))
        plt.savefig(os.path.join(sub_dir_pdf, str(i) + '.svg'))
        plt.close()


def normalize(x):
    max_ = x.max()
    min_ = x.min()
    return (x - min_) / (max_ - min_)

y_var_all = []
y_all = []
y_pred_all = []
name_all = []
x_all = np.zeros((0, 256, 256, 3))

for fold_num in range(5):
    print(fold_num)
    h5f = h5py.File(mc_files[fold_num], 'r')
    y = h5f['y'][:]  # [N, C]
    mc_probs = h5f['mc_probs'][:]  # [100, N, C]
    h5f.close()

    h5f = h5py.File(normal_files[fold_num], 'r')
    x = h5f['x'][:]
    name = h5f['name'][:]  # [N]
    y_pred = h5f['y_pred'][:]  # [N]   Regular prediction, not Bayesian
    h5f.close()

    #
    # idx = None
    # if fold_num == 0:
    #     idx = np.where(name == 100720190523)[0]
    # if fold_num == 1:
    #     idx = np.where(name == 100720190833)[0]
    # if fold_num == 3:
    #     idx = np.where(name == 100720190705)[0]
    # print()
    #
    # if idx is not None:
    #     name = np.delete(name, idx)
    #     y = np.delete(y, idx, axis=0)
    #     y_pred = np.delete(y_pred, idx, axis=0)
    #     mc_probs = np.delete(mc_probs, idx, axis=1)
    #
    #
    #




    all_mc_acc = []
    for iter_num in range(100):
        mean_prob = mc_probs[0:iter_num + 1].mean(axis=0)
        all_mc_acc.append(
            1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), y.argmax(axis=1))) / mean_prob.shape[0])

    opt_iter = np.argmax(np.array(all_mc_acc))
    # get the values on T
    y_probs = mc_probs[:opt_iter]  # [opt_iter, N, C]
    mean_probs = y_probs.mean(axis=0)

    # compute uncertainties
    var_pred_entropy = predictive_entropy(mean_probs)  # [N, C]
    # var_pred_MI = mutual_info(y_probs)

    # normalize the uncertainty values
    y_var = normalize(var_pred_entropy)


    # plot subject level distributions
    unq_subs = np.unique(name)
    for sub in unq_subs:
        if sub not in done_subs:
            sub_idx = np.where(name == sub)
            sub_x = x[sub_idx]
            sub_y = y[sub_idx].argmax(axis=1)
            sub_y_pred = mean_probs[sub_idx].argmax(axis=1)
            sub_y_prob = np.squeeze(y_probs[:, sub_idx])
            sub_y_var = y_var[sub_idx]

            subject_dist_plot(sub_x, sub_y, sub_y_pred, sub_y_prob, sub_y_var, sub)

    x_all = np.concatenate((x_all, x), axis=0)
    y_all = np.append(y_all, y.argmax(axis=1))
    y_var_all = np.append(y_var_all, y_var)
    y_pred_all = np.append(y_pred_all, mean_probs.argmax(axis=1))
    name_all = np.append(name_all, name)

np.savez_compressed('result', y=y_all, y_var=y_var_all, y_pred=y_pred_all)
uncertainty_density_plot(y_all, y_pred_all, y_var_all, 'error_vs_correct')





