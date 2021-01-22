import PIL
from torchvision import transforms
import h5py
import torch
import numpy as np
from config import options
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, mode='train', data_len=None):

        self.mode = mode
        if mode == 'train':
            print('Loading the training data...')
            train_idx = list(range(5))
            train_idx.remove(options.loos)
            images = np.zeros((0, options.img_c, options.img_h, options.img_w))
            labels = np.array([])
            for idx in train_idx:
                h5f = h5py.File('/home/cougarnet.uh.edu/pcicales/Desktop/'
                                'glomerulus_classification-master/dataset/5fold_data/data_5C_{}.h5'.
                                format(idx), 'r')
                x = np.transpose(h5f['x'][:], [0, 3, 1, 2]).astype(int)
                y = h5f['y'][:].astype(int)
                images = np.concatenate((images, x), axis=0)
                labels = np.append(labels, y)
                h5f.close()
            print('Training Data Label Counts (Pre-Label Condensation):')
            idx0 = np.where(labels == 0)[0]
            print('Train label 0 image count: ' + str(np.count_nonzero(idx0)))
            idx1 = np.where(labels == 1)[0]
            print('Train label 1 image count: ' + str(np.count_nonzero(idx1)))
            idx2 = np.where(labels == 2)[0]
            print('Train label 2 image count: ' + str(np.count_nonzero(idx2)))
            idx3 = np.where(labels == 3)[0]
            print('Train label 3 image count: ' + str(np.count_nonzero(idx3)))
            idx4 = np.where(labels == 4)[0]
            print('Train label 4 image count: ' + str(np.count_nonzero(idx4)))

            print('Training Data Label Counts (Post-Label Condensation, ' + str(options.num_classes) + ' class):')
            if options.num_classes == 5:
                idx0 = np.where(labels == 0)[0]
                print('Train label 0 image count: ' + str(np.count_nonzero(idx0)))
                idx1 = np.where(labels == 1)[0]
                print('Train label 1 image count: ' + str(np.count_nonzero(idx1)))
                idx2 = np.where(labels == 2)[0]
                print('Train label 2 image count: ' + str(np.count_nonzero(idx2)))
                idx3 = np.where(labels == 3)[0]
                print('Train label 3 image count: ' + str(np.count_nonzero(idx3)))
                idx4 = np.where(labels == 4)[0]
                print('Train label 4 image count: ' + str(np.count_nonzero(idx4)))
            elif options.num_classes == 4:
                labels[idx4] = 3
                idx0 = np.where(labels == 0)[0]
                print('Train label 0 image count: ' + str(np.count_nonzero(idx0)))
                idx1 = np.where(labels == 1)[0]
                print('Train label 1 image count: ' + str(np.count_nonzero(idx1)))
                idx2 = np.where(labels == 2)[0]
                print('Train label 2 image count: ' + str(np.count_nonzero(idx2)))
                idx3 = np.where(labels == 3)[0]
                print('Train label 3 image count: ' + str(np.count_nonzero(idx3)))
            elif options.num_classes == 3:
                labels[idx2] = 1
                labels[idx3] = 2
                labels[idx4] = 2
                idx0 = np.where(labels == 0)[0]
                print('Train label 0 image count: ' + str(np.count_nonzero(idx0)))
                idx1 = np.where(labels == 1)[0]
                print('Train label 1 image count: ' + str(np.count_nonzero(idx1)))
                idx2 = np.where(labels == 2)[0]
                print('Train label 2 image count: ' + str(np.count_nonzero(idx2)))

            self.images = images
            self.labels = labels

        elif mode == 'test':
            print('Loading the test data...')
            h5f = h5py.File('/home/cougarnet.uh.edu/pcicales/Desktop/'
                            'glomerulus_classification-master/dataset/5fold_data/data_5C_{}.h5'
                            .format(options.loos), 'r')
            self.images = np.transpose(h5f['x'][:], [0, 3, 1, 2]).astype(int)[:data_len]
            self.labels = h5f['y'][:].astype(int)[:data_len]
            h5f.close()

            print('Testing Data Label Counts (Pre-Label Condensation):')
            idx0 = np.where(self.labels == 0)[0]
            print('Test label 0 image count: ' + str(np.count_nonzero(idx0)))
            idx1 = np.where(self.labels == 1)[0]
            print('Test label 1 image count: ' + str(np.count_nonzero(idx1)))
            idx2 = np.where(self.labels == 2)[0]
            print('Test label 2 image count: ' + str(np.count_nonzero(idx2)))
            idx3 = np.where(self.labels == 3)[0]
            print('Test label 3 image count: ' + str(np.count_nonzero(idx3)))
            idx4 = np.where(self.labels == 4)[0]
            print('Test label 4 image count: ' + str(np.count_nonzero(idx4)))

            print('Training Data Label Counts (Post-Label Condensation, ' + str(options.num_classes) + ' class):')
            if options.num_classes == 5:
                idx0 = np.where(self.labels == 0)[0]
                print('Test label 0 image count: ' + str(np.count_nonzero(idx0)))
                idx1 = np.where(self.labels == 1)[0]
                print('Test label 1 image count: ' + str(np.count_nonzero(idx1)))
                idx2 = np.where(self.labels == 2)[0]
                print('Test label 2 image count: ' + str(np.count_nonzero(idx2)))
                idx3 = np.where(self.labels == 3)[0]
                print('Test label 3 image count: ' + str(np.count_nonzero(idx3)))
                idx4 = np.where(self.labels == 4)[0]
                print('Test label 4 image count: ' + str(np.count_nonzero(idx4)))
            elif options.num_classes == 4:
                self.labels[idx4] = 3
                idx0 = np.where(self.labels == 0)[0]
                print('Test label 0 image count: ' + str(np.count_nonzero(idx0)))
                idx1 = np.where(self.labels == 1)[0]
                print('Test label 1 image count: ' + str(np.count_nonzero(idx1)))
                idx2 = np.where(self.labels == 2)[0]
                print('Test label 2 image count: ' + str(np.count_nonzero(idx2)))
                idx3 = np.where(self.labels == 3)[0]
                print('Test label 3 image count: ' + str(np.count_nonzero(idx3)))
            elif options.num_classes == 3:
                self.labels[idx2] = 1
                self.labels[idx3] = 2
                self.labels[idx4] = 2
                idx0 = np.where(self.labels == 0)[0]
                print('Test label 0 image count: ' + str(np.count_nonzero(idx0)))
                idx1 = np.where(self.labels == 1)[0]
                print('Test label 1 image count: ' + str(np.count_nonzero(idx1)))
                idx2 = np.where(self.labels == 2)[0]
                print('Test label 2 image count: ' + str(np.count_nonzero(idx2)))

    def __getitem__(self, index):

        # img = torch.tensor(self.images[index]).div(255.).float()

        img = torch.tensor(self.images[index]).float()
        img = (img - img.min()) / (img.max() - img.min())
        target = torch.tensor(self.labels[index]).long()

        if self.mode == 'train':
            # normalization & augmentation
            img = transforms.ToPILImage()(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomVerticalFlip()(img)
            # img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05)(img)
            img = transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.))(img)
            img = transforms.RandomRotation(90, resample=PIL.Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)

        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        return len(self.labels)
