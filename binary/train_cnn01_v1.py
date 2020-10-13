import torch
import os
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
sys.path.append('..')

from tools import args, load_data, ModelArgs, BalancedBatchSampler
from core.lossfunction import ZeroOneLoss
from core.cnn01 import LeNet_cifar, _01_init
from core.basic_module_v1 import *
from core.train_cnn01_basic_v1 import train_single_cnn01
import pickle
import pandas as pd
# Args assignment
scd_args = ModelArgs()

scd_args.nrows = args.nrows
scd_args.nfeatures = args.nfeatures
scd_args.w_inc = args.w_inc
scd_args.tol = 0.00000
scd_args.local_iter = args.iters
scd_args.num_iters = args.num_iters
scd_args.interval = args.interval
scd_args.rounds = args.round
scd_args.w_inc1 = args.w_inc1
scd_args.updated_features = args.updated_features
scd_args.n_jobs = args.n_jobs
scd_args.num_gpus = args.num_gpus
scd_args.adv_train = True if args.adv_train else False
scd_args.eps = args.eps
scd_args.w_inc2 = args.w_inc2
scd_args.hidden_nodes = args.hidden_nodes
scd_args.evaluation = False if args.no_eval else True
scd_args.verbose = True if args.verbose else False
scd_args.b_ratio = args.b_ratio
scd_args.cuda = True if torch.cuda.is_available() else False
scd_args.seed = args.seed
scd_args.target = args.target
scd_args.source = args.source
scd_args.save = True if args.save else False
scd_args.resume = True if args.resume else False
scd_args.criterion = ZeroOneLoss
if args.version == 'lenet_cifar':
    scd_args.structure = LeNet_cifar

scd_args.dataset = args.dataset
scd_args.num_classes = args.n_classes
scd_args.gpu = args.gpu
scd_args.fp16 = True if args.fp16 else False
scd_args.act = args.act
scd_args.updated_nodes = args.updated_nodes
scd_args.width = args.width
scd_args.normal_noise = True if args.normal_noise else False

train_data, test_data, train_label, test_label = load_data('cifar10', 2)
train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
np.random.seed(scd_args.seed)

best_model, val_acc = train_single_cnn01(scd_args, None, None, (train_data, test_data, train_label, test_label))