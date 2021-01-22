import os
import glob
import imageio
import numpy as np
import skimage.transform
import operator
import itertools
import h5py
import json

imgs_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/masks'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_name_list = glob.glob(os.path.join(imgs_dir+"/*.jpg"))
new_w, new_h = 256, 256

data = np.zeros((len(img_name_list), new_h, new_w, 3))
subject_ids, labels = np.array([]), np.zeros(len(img_name_list))
subject_glom_dict = {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}}
subject_label_dict = {}
seen_subjects = []

heights, widths = np.array([]), np.array([])

for i, img_name in enumerate(img_name_list):
    x = imageio.imread(img_name)
    heights = np.append(heights, x.shape[0])
    widths = np.append(widths, x.shape[1])

    data[i] = skimage.transform.resize(x, (new_h, new_w, 3), mode='constant', preserve_range=True)
    subject_id = img_name.split('glom')[0].split('/')[-1][:-1]
    subject_ids = np.append(subject_ids, img_name.split('glom')[0].split('/')[-1][:-1])
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


def approx_with_accounting_and_duplicates(x_list, s):
    '''
    Modified from http://en.wikipedia.org/wiki/Subset_sum_problem#Polynomial_time_approximate_algorithm
         initialize a list S to contain one element 0.
         for each i from 1 to N do
           let T be a list consisting of xi + y, for all y in S
           let U be the union of T and S
           sort U
           make S empty
           let y be the smallest element of U
           add y to S
           for each element z of U in increasing order do
              //trim the list by eliminating numbers close to one another
              //and throw out elements greater than s
             if y + cs/N < z <= s, set y = z and add z to S
         if S contains a number between (1 - c)s and s, output yes, otherwise no
    '''
    c = .01  # fraction error (constant)
    N = len(x_list)  # number of values
    S = [('', 0, [])]
    for lbl, x in sorted(x_list, key=operator.itemgetter(1)):
        T = []
        L = []
        for _, y, y_list in S:
            T.append((lbl, x + y, y_list + [(lbl, x)]))
        U = T + S
        U = sorted(U, key=operator.itemgetter(1))
        lbl, y, y_list = U[0]
        S = [(lbl, y, y_list)]

        for lbl, z, z_list in U:
            lower_bound = (float(y) + c * float(s) / float(N))
            if lower_bound < z <= s:
                y = z
                S.append((lbl, z, z_list))
    return sorted(S, key=operator.itemgetter(1))[-1]


def split(scan_count, splits=(0.8, 0.2)):
    """
    generate train, test splits by ratio and dumps a splits.json file
            @splits: list of train,test split ratios
    """

    keys = list(scan_count.keys())
    np.random.shuffle(keys)
    num_imgs = []
    total = 0.0
    for k in keys:
        v = scan_count[k]
        num_imgs.append((k, v))
        total += v

    train = approx_with_accounting_and_duplicates(num_imgs, int(splits[0] * total))[2]
    num_imgs = [a for a in num_imgs if a not in train]
    test = num_imgs

    return train, test


# subject_glom_dict.pop('1')
# subject_glom_dict.pop('2')
# subject_glom_dict.pop('3')


train_list, test_list = [], []
for key in subject_glom_dict.keys():
    train, test = split(subject_glom_dict[key])

    # add labels
    train = [train[i] + tuple(str(subject_label_dict[train[i][0]])) for i in range(len(train))]
    test = [test[i] + tuple(str(subject_label_dict[test[i][0]])) for i in range(len(test))]

    train_list.append(train)
    test_list.append(test)

with open(os.path.join(BASE_DIR, 'splits.json'), 'w') as f:
    json.dump({'train': train_list, 'test': test_list}, f)

train_list = list(itertools.chain(*train_list))
test_list = list(itertools.chain(*test_list))
train_scans = [tuple_[0] for tuple_ in train_list]
test_scans = [tuple_[0] for tuple_ in test_list]

x_train = np.array(data[[i for i, e in enumerate(subject_ids) if e in train_scans]])
y_train = np.array(labels[[i for i, e in enumerate(subject_ids) if e in train_scans]])
name_train = np.array([[tuple_[0]]*tuple_[1] for tuple_ in train_list])

x_test = np.array(data[[i for i, e in enumerate(subject_ids) if e in test_scans]])
y_test = np.array(labels[[i for i, e in enumerate(subject_ids) if e in test_scans]])
name_test = np.array([[tuple_[0]]*tuple_[1] for tuple_ in test_list])


# for i, y in enumerate(y_train):
#     if y_train[i] == 2:
#         y_train[i] = 1
#     elif y_train[i] == 3:
#         y_train[i] = 2
#     elif y_train[i] == 4:
#         y_train[i] = 3
#
# for i, y in enumerate(y_test):
#     if y_test[i] == 2:
#         y_test[i] = 1
#     elif y_test[i] == 3:
#         y_test[i] = 2
#     elif y_test[i] == 4:
#         y_test[i] = 3

h5f = h5py.File('data_5C.h5', 'w')
h5f.create_dataset('x_train', data=x_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('name_train', data=name_train)
h5f.create_dataset('x_test', data=x_test)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('name_test', data=name_test)
h5f.close()

# with open(os.path.join(BASE_DIR, 'splits.json'), 'r') as f:
#     scan_names = json.load(f)['train']





