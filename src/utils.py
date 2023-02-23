from sklearn.metrics import roc_curve, auc,precision_recall_curve
import numpy as np
import pandas as pd

def load_data(args):
    labels = np.loadtxt(args.adj)
    AM = pd.read_csv(args.AM, header=None).values
    CF = pd.read_csv(args.CF, header=None).values
    MF = pd.read_csv(args.MF, header=None).values
    return labels,AM,CF,MF

def score(test_labels, scores):
    fpr, tpr, threshold = roc_curve(test_labels, scores)
    precision, recall, _thresholds = precision_recall_curve(test_labels, scores, pos_label=1)
    roc = auc(fpr, tpr)
    aupr = auc(recall, precision)
    print(f'ROC:{roc} AUPR:{aupr}')
