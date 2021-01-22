import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def compute_accuracy(target, output):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size]
    :param output: Tensor of model predictions of size [batch_size, num_classes]
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    num_correct = torch.sum(target == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy


def mutual_info(mc_prob):
    """
    computes the mutual information
    :param mc_prob: List MC probabilities of length mc_simulations;
                    each of shape  of shape [batch_size, num_cls]
    :return: mutual information of shape [batch_size, num_cls]
    """
    eps = 1e-5
    mean_prob = mc_prob.mean(axis=0)
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=1)
    return first_term + second_term


def predictive_entropy(prob):
    """
    Entropy of the probabilities (to measure the epistemic uncertainty)
    :param prob: probabilities of shape [batch_size, C]
    :return: Entropy of shape [batch_size]
    """
    eps = 1e-5
    return -1 * np.sum(np.log(prob+eps) * prob, axis=1)


def save_confusion_matrix(y_true, y_pred, classes, dest_path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens):
    """
    This function plots and saves the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(dest_path)

