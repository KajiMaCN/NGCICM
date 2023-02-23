from src.train import train
from src.inits import div_list
from src.utils import score, load_data
import numpy as np
import argparse
import tensorflow as tf
from models import GAT


def get_args():
    parser = argparse.ArgumentParser(description='NGCICM')
    parser.add_argument('--adj', default='data/adj.txt', type=str, help='list of correlation matrix')
    parser.add_argument('--AM', default='data/AM.csv',type=str, help='the adjacency matrix')
    parser.add_argument('--CF', default='data/CF.csv',type=str, help='the circRNA feature matrix')
    parser.add_argument('--MF', default='data/MF.csv',type=str, help='the miRNA feature matrix')
    parser.add_argument('--T', default=10, type=int, help='number of runs')
    parser.add_argument('--cv_num', default=5, type=int,help='n-fold cross-validation')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--nb_epochs', default=1000, type=int, help='iteration number')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--l2_coef', default=0.005, type=float, help='l2 coefficient')
    parser.add_argument('--weight_decay', default=5e-4, type=float,help='weight decay')
    parser.add_argument('--delta', default=8, type=int, help='number of neurons')
    parser.add_argument('--k', default=[4, 1], type=list, help='number of interaction')
    parser.add_argument('--act', default=tf.nn.elu, help='activation function')
    parser.add_argument('--model', default=GAT, help='basic model')
    return parser


def main(args):
    labels,AM,CF,MF=load_data(args)
    reorder = np.arange(labels.shape[0])
    np.random.shuffle(reorder)
    for t in range(args.T):
        order = div_list(reorder.tolist(),args.cv_num)
        for i in range(args.cv_num):
            test_arr = order[i]
            arr = list(set(reorder).difference(set(test_arr)))
            np.random.shuffle(arr)
            train_arr = arr
            test_labels, scores,acc= train(labels,AM,CF,MF,train_arr, test_arr,args)
            score(test_labels, scores)


if __name__=='__main__':
    parser=get_args()
    args = parser.parse_args()
    main(args)