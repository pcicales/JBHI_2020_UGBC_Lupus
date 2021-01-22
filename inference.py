import os
import warnings

import h5py

warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset.dataset import Data as data
from utils.visualize_utils import visualize
from utils.eval_utils import compute_accuracy
from models import *
from config import options
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import torch.nn.functional as F
from utils.eval_utils import mutual_info, predictive_entropy
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# def apply_dropout(m):
#     if m.__class__.__name__ == '_DenseLayer':
#         m.train()


def evaluate():
    net.eval()
    test_loss = 0
    targets, outputs = [], []

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
    return outputs, targets, test_loss


@torch.no_grad()
def mc_evaluate():
    net.eval()

    # if options.MC:
    #     net.apply(apply_dropout)

    test_loss = 0
    targets, outputs, probs = [], [], []

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = net(data)
            prob = F.softmax(output)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            probs += [prob]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
    return torch.cat(probs).unsqueeze(0).cpu().numpy(), F.one_hot(torch.cat(targets), options.num_classes).cpu().numpy(), test_loss


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    mc_dir = os.path.join(save_dir, 'mc_results')
    if not os.path.exists(mc_dir):
        os.makedirs(mc_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    if options.model == 'resnet':
        net = resnet.resnet50()
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
        grad_cam_hooks = {'forward': net.layer4, 'backward': net.fc}
    elif options.model == 'vgg':
        net = vgg19_bn(pretrained=True, num_classes=options.num_classes)
        grad_cam_hooks = {'forward': net.features, 'backward': net.fc}
    elif options.model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(2048, options.num_classes)
        grad_cam_hooks = {'forward': net.Mixed_7c, 'backward': net.fc}
    elif options.model == 'densenet':
        net = densenet.densenet121()
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)
        grad_cam_hooks = {'forward': net.features.norm5, 'backward': net.classifier}

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))

    ##################################
    # Loss and Optimizer
    ##################################
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=options.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################

    # train_dataset = data(mode='train', data_len=options.data_len)
    # train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
    #                           shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start Testing')

    h5f = h5py.File('/home/cougarnet.uh.edu/pcicales/Desktop/'
                    'glomerulus_classification-master/dataset/5fold_data/data_5C_{}.h5'
                    .format(options.loos), 'r')
    x = h5f['x'][:]
    names = h5f['name'][:].astype(int)
    h5f.close()

    if options.MC:
        mc_probs = []
        for mc_iter in range(options.mc_iter):
            print('running mc iteration # {}'.format(mc_iter))
            iter_probs, iter_targets, iter_loss = mc_evaluate()
            mc_probs += [iter_probs]
        mc_probs = np.concatenate(mc_probs)  # [mc_iter, N, C]

        h5f = h5py.File(os.path.join(mc_dir, 'output_0.2_loos={}_mc={}.h5'.format(options.loos, options.mc_iter)), 'w')
        h5f.create_dataset('y', data=iter_targets)
        h5f.create_dataset('mc_probs', data=mc_probs)
        h5f.close()

        mean_prob = mc_probs.mean(axis=0)
        var_pred_entropy = predictive_entropy(mean_prob)
        var_pred_MI = mutual_info(mc_probs)
        acc = 1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), iter_targets.argmax(axis=1))) / mean_prob.shape[0]
        print('accuracy={0:.02%}'.format(acc))
    else:
        test_outputs, test_targets, test_loss_ = evaluate()
        test_acc = compute_accuracy(torch.cat(test_targets), torch.cat(test_outputs))
        targets = torch.cat(test_targets).cpu().numpy()
        outputs = np.argmax(torch.cat(test_outputs).cpu().numpy(), axis=1)
        log_string('Glomerulus Level Classification Confusion Matrix and Accuracy: ')
        log_string(str(confusion_matrix(targets, outputs)))
        # display
        log_string("validation_loss: {0:.4f}, validation_accuracy: {1:.02%}".format(test_loss_, test_acc))

        h5f = h5py.File(os.path.join(save_dir, 'prediction_{}.h5'.format(options.loos)), 'w')
        h5f.create_dataset('x', data=x)
        h5f.create_dataset('name', data=names)
        h5f.create_dataset('y', data=targets)
        h5f.create_dataset('y_pred', data=outputs)
        h5f.close()


    #################################
    # Grad Cam visualizer
    #################################
    # if options.gradcam:
    #     log_string('Generating Gradcam visualizations')
    #     iter_num = options.load_model_path.split('/')[-1].split('.')[0]
    #     img_dir = os.path.join(save_dir, 'imgs')
    #     if not os.path.exists(img_dir):
    #         os.makedirs(img_dir)
    #     viz_dir = os.path.join(img_dir, iter_num)
    #     if not os.path.exists(viz_dir):
    #         os.makedirs(viz_dir)
    #     visualize(net, test_loader, grad_cam_hooks, viz_dir)
    #     log_string('Images saved in: {}'.format(viz_dir))



