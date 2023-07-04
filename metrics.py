import torch
import torchmetrics
from torchmetrics.classification import MulticlassSpecificity
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import precision_recall_fscore_support

def torchmetrics_accuracy(preds, labels):
    acc = torchmetrics.functional.accuracy(preds, labels)
    return acc

def torchmetrics_spef(preds, labels):
    metric = MulticlassSpecificity(num_classes=2).cuda()
    spef = metric(preds, labels)
    return spef

def torchmetrics_auc(preds, labels):
    auc = torchmetrics.functional.auroc(preds, labels, task="multiclass", num_classes=2)
    return auc

def confusion_matrix(preds, labels):
    conf_matrix = torch.zeros(2, 2)
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1 
    return conf_matrix
def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : computer the value of confusion matrix
    - normalize : True: %, False: 123
    """
    classes = ['0:ASD','1:TC']
    if normalize:
        cm = cm.numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def correct_num(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction)

def prf(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    pred_lab= np.argmax(preds, 1)
    p,r,f,s  = precision_recall_fscore_support(labels, pred_lab, average='binary')
    return [p,r,f]





