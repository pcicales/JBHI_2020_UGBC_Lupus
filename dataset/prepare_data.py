import os
import glob
import imageio
import numpy as np
import skimage.transform
import operator
import itertools
import h5py
import json
import random

imgs_dir = '/home/cougarnet.uh.edu/pcicales/Desktop/masks'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Excluding the suspect patients
temp_img_name_list = glob.glob(os.path.join(imgs_dir+"/*.jpg"))
img_name_list = list()
for i in range(0,len(temp_img_name_list)):
    # if str(100720190608) in temp_img_name_list[i]:
    #     continue
    # else:
        img_name_list.append(temp_img_name_list[i])

new_w, new_h = 256, 256
N = 10      # desired number of folds
num_cls = 5

data = np.zeros((len(img_name_list), new_h, new_w, 3))
subject_ids, labels = np.array([]), np.zeros(len(img_name_list))

subject_glom_dict = {}
for i in range(num_cls):
    subject_glom_dict[str(i)] = {}
    subject_label_dict = {}
seen_subjects = []

heights, widths = np.array([]), np.array([])

for i, img_name in enumerate(img_name_list):
    x = imageio.imread(img_name)
    heights = np.append(heights, x.shape[0])
    widths = np.append(widths, x.shape[1])

    data[i] = skimage.transform.resize(x, (new_h, new_w, 3), mode='constant', preserve_range=True)
    subject_id = img_name.split('glom')[0].split('/')[-1][:-1].split('_')[-1]
    subject_ids = np.append(subject_ids, subject_id)
    label = int(img_name.split('glom')[1].split('_')[1])

    labels[i] = label
    if subject_id not in seen_subjects:
        seen_subjects.append(subject_id)
        subject_glom_dict[str(label)][subject_id] = 0
        subject_label_dict[subject_id] = label

# add counts
subjects = np.unique(seen_subjects)
for i, subject_id in enumerate(subjects):
    label = subject_label_dict[subject_id]
    count = np.where(labels[subject_ids == subject_id] == label)[0].shape[0]
    subject_glom_dict[str(label)][subject_id] = count


def split(label_dict, num_folds):
    total_subs = len(label_dict)
    in_each = total_subs // num_folds
    fold_sizes = in_each * np.ones(num_folds).astype(int)
    remain = total_subs - (in_each*num_folds)
    idxs = random.sample(range(num_folds), remain)
    fold_sizes[idxs] += 1
    sub_list = list(label_dict)
    all_subs = []
    for num_pops in fold_sizes:
        current = []
        for j in range(num_pops):
            current.append(sub_list.pop())
        all_subs.append(current)
    return all_subs


fold_list = [[] for i in range(N)]
for key in subject_glom_dict.keys():
    key_subs = split(subject_glom_dict[key], num_folds=N)
    for i in range(N):
        fold_list[i].append(key_subs[i])

for i, fold in enumerate(fold_list):
    fold_list[i] = list(itertools.chain(*fold))
# so now fold_list is a list of size N where each element is a list of subject ids for that fold


for i, fold in enumerate(fold_list):
    fold_names, fold_data, fold_labels = np.array([]), np.zeros((0, new_w, new_h, 3)), np.array([])
    for sub_id in fold:
        idxs = np.where(subject_ids == sub_id)[0]
        fold_data = np.concatenate((fold_data, data[idxs]), axis=0)
        fold_labels = np.append(fold_labels, labels[idxs])
        fold_names = np.append(fold_names, subject_ids[idxs])

        h5f = h5py.File('data_5C_{}.h5'.format(i), 'w')
        h5f.create_dataset('x', data=fold_data)
        h5f.create_dataset('y', data=fold_labels.astype(int))
        h5f.create_dataset('name', data=fold_names.astype(int))
        h5f.close()

print()




