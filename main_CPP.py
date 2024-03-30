import numpy as np
import random as random
import torch
import copy
import argparse

from models.Semi_CPP import CPP
## Some recommend configs
# Cora-->results:85.82 'order': 4,'nheads': 8,'Trans_layer_num': 2,'lr': 0.0005,'MLPdim':256,'dropout_att': 0.4,'random_aug_feature': 0.2, 'beta': 0.06,tua:1,'wd': 6e-4,
# CiteSeer-->results:75.06 'order': 4,'nheads': 8,'Trans_layer_num': 2,'lr': 0.0004,'MLPdim':256,'dropout_att': 0.4,'random_aug_feature': 0.3, 'beta': 0.005, tua:1,'wd': 5e-4
# pubmed-->results:81.48 'order': 3,'nheads': 8,'Trans_layer_num': 2,'lr': 0.0008,'MLPdim':256,'dropout_att': 0.5,'random_aug_feature': 0.2, 'beta': 100, 'wd':5e-4:tua:0.15, batch:4000

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Cora', help='dataset used.')
parser.add_argument('--beta', type=float, default=0.06, help='hyparameter in loss function.')
parser.add_argument('--tau', type=float, default=1, help='hyparameter in contrastive loss.')
parser.add_argument('--order', type=int, default=4, help='number of multi-hop graph.')
parser.add_argument('--nb_epochs', type=int, default=5000, help='maximal epochs.')
parser.add_argument('--patience', type=int, default=70, help='early stop.')
parser.add_argument('--nheads', type=int, default=8, help='number of heads in self-attention.')
parser.add_argument('--Trans_layer_num', type=int, default=2, help='layers number for self-attention.')
parser.add_argument('--lr', type=float, default=0.0005, help='learning ratio.')
parser.add_argument('--MLPdim', type=int, default=256, help='hidden dimension.')
parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension in self-attention.')
parser.add_argument('--dropout_att', type=float, default=0.4, help='dropout in self-attention layers.')
parser.add_argument('--random_aug_feature', type=float, default=0.2, help='dropout in hidden layers.')
parser.add_argument('--wd', type=float, default=6e-4, help='weight delay.')
parser.add_argument('--act', type=str, default='leakyrelu', help='hidden action.')

args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    if torch.cuda.is_available():
        args.device = torch.device('cuda:4')
    else:
        args.device = torch.device('cpu')

    ACC_seed = []
    Time_seed = []
    for seed in range(2020, 2025):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        embedder = CPP(copy.deepcopy(args))
        test_acc, training_time, stop_epoch = embedder.training()
        ACC_seed.append(test_acc)
        Time_seed.append(training_time)
        torch.cuda.empty_cache()
    ACC_seed = np.array(ACC_seed)*100

    print("-->ACC %.4f  -->STD is: %.4f" %(np.mean(ACC_seed), np.std(ACC_seed)))
