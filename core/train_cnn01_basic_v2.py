import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
sys.path.append('..')

from tools import args, load_data, ModelArgs, BalancedBatchSampler
from core.lossfunction import ZeroOneLoss, BCELoss
from core.cnn01 import LeNet_cifar, _01_init, Toy
from core.basic_module_v2 import *
from core.basic_function import evaluation
import pickle
import pandas as pd


def train_single_cnn01(scd_args, device=None, seed=None, data_set=None):
    if device is not None:
        scd_args.gpu = device
    if seed is not None:
        scd_args.seed = seed
    resume = False
    use_cuda = scd_args.cuda
    dtype = torch.float16 if scd_args.fp16 else torch.float32

    best_acc = 0

    # seed = 2047775

    print('Random seed: ', scd_args.seed)
    np.random.seed(scd_args.seed)
    torch.manual_seed(scd_args.seed)
    df = pd.DataFrame(columns=['epoch', 'train acc', 'test acc'])
    log_path = os.path.join('logs', scd_args.dataset)
    log_file_name = os.path.join(log_path, scd_args.target.replace('.pkl', '.csv'))

    if scd_args.save:
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    train_data, test_data, train_label, test_label = data_set
    batch_size = int(train_data.shape[0] * scd_args.nrows) \
        if scd_args.nrows < 1 else int(scd_args.nrows)
    
    trainset = TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
                            torch.from_numpy(train_label.astype(np.int64)).reshape((-1, 1)))
    testset = TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
                            torch.from_numpy(test_label.astype(np.int64)).reshape((-1, 1)))
    train_loader = DataLoader(trainset, batch_size=batch_size,  num_workers=0,
                    pin_memory=True,
        sampler=BalancedBatchSampler(trainset, torch.from_numpy(train_label)))
    # train_loader = DataLoader(trainset, batch_size=batch_size,  num_workers=0,
    #                                            pin_memory=False, shuffle=True,)
    val_loader = DataLoader(trainset, batch_size=train_data.shape[0], shuffle=False, num_workers=0,
                                             pin_memory=False)
    test_loader = DataLoader(testset, batch_size=test_data.shape[0], shuffle=False, num_workers=0,
                                              pin_memory=False)

    net = scd_args.structure(
        num_classes=1, act=scd_args.act, sigmoid=scd_args.sigmoid)
    # _01_init(net)
    best_model = scd_args.structure(
        num_classes=1, act=scd_args.act, sigmoid=scd_args.sigmoid)

    if scd_args.resume:
        with open('checkpoints/%s' % scd_args.source, 'rb') as f:
            temp = pickle.load(f).best_model.state_dict()
            net.load_state_dict(temp)

    criterion = scd_args.criterion()

    if scd_args.cuda:
        print('start move to cuda')
        torch.cuda.manual_seed_all(scd_args.seed)
        # torch.backends.cudnn.deterministic = True
        cudnn.benchmark = True
        if scd_args.fp16:
            net = net.half()

        # net = torch.nn.DataParallel(net, device_ids=[0,1])
        device = torch.device("cuda:%s" % scd_args.gpu)
        net.to(device=device)
        best_model.to(device=device)
        criterion.to(device=device, dtype=dtype)

    net.eval()

    best_acc = 0

    # Training
    for epoch in range(scd_args.num_iters):

        print(f'\nEpoch: {epoch}')
        a = time.time()

        layers = list(net._modules.keys())[::-1]  # reverse the order of layers' name
        with torch.no_grad():

            # update Final layer
            p = iter(train_loader)
            data, target = p.next()
            # for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            train_loss = 5

            # initial bias
            if epoch == 0:
                init_bias(net, data, criterion, target, dtype, scd_args)

            # update final layer
            update_final_layer_fc(net, layers, dtype, data,
                            scd_args, criterion, target, device)
            # update H2->H-1 layer

            if len(layers) > 2:
                for layer_index, layer in enumerate(layers[1:-1], start=1):
                    if 'fc' in layer:
                        update_mid_layer_fc(net, layers, layer_index, data, dtype,
                                            scd_args, criterion, target, device)
                    elif 'conv' in layer:
                        update_mid_layer_conv(net, layers, layer_index, data,
                            dtype, scd_args, criterion, target, device)

            # update H1
            layer_index = len(layers) - 1
            train_loss = update_first_layer_conv(net, layers, layer_index,
                    data, dtype, scd_args, criterion, target, device)

            # print('current loss: %.5f' % train_loss)
            print('This epoch cost %0.2f seconds' % (time.time() - a))

            val_acc = evaluation(val_loader, use_cuda, device,
                                 dtype, net, 'Train')
            if val_acc > best_acc:
                best_acc = val_acc
                best_model.load_state_dict(net.state_dict())
            test_acc = evaluation(test_loader, use_cuda, device,
                                 dtype, net, 'Test')
            temp_row = pd.Series(
                {'epoch': epoch+1,
                 'train acc': val_acc,
                 'test acc': test_acc,
                 }
            )

            if scd_args.save:
                if epoch == 0:
                    with open(log_file_name.replace('csv', 'temp'), 'w') as f:
                        f.write('epoch, train_acc, test_acc\n')
                else:
                    with open(log_file_name.replace('csv', 'temp'), 'a') as f:
                        f.write(f'{epoch}, {val_acc}, {test_acc}\n')

            df = df.append(temp_row, ignore_index=True)




    df.to_csv(log_file_name, index=False)

    # scd.set_model(best_model.cpu())
    # if scd_args.save:
    #     scd.save('checkpoints', scd_args.target)
    return best_model.cpu(), best_acc


if __name__ == '__main__':

    scd_args = ModelArgs()

    scd_args.nrows = 0.15
    scd_args.nfeatures = 1
    scd_args.w_inc = 0.17
    scd_args.tol = 0.00000
    scd_args.local_iter = 1
    scd_args.num_iters = 1000
    scd_args.interval = 20
    scd_args.rounds = 1
    scd_args.w_inc1 = 0.1
    scd_args.updated_fc_features = 128
    scd_args.updated_conv_features = 3
    scd_args.n_jobs = 1
    scd_args.num_gpus = 1
    scd_args.adv_train = False
    scd_args.eps = 0.1
    scd_args.w_inc2 = 0.1
    scd_args.hidden_nodes = 20
    scd_args.evaluation = True
    scd_args.verbose = True
    scd_args.b_ratio = 0.2
    scd_args.cuda = True
    scd_args.seed = 2018
    scd_args.target = 'toy'
    scd_args.source = None
    scd_args.save = True
    scd_args.resume = False
    scd_args.criterion = BCELoss
    scd_args.structure = Toy
    scd_args.dataset = 'cifar10'
    scd_args.num_classes = 2
    scd_args.gpu = 1
    scd_args.fp16 = True
    scd_args.act = 'sign'
    scd_args.updated_nodes = 1
    scd_args.width = 100
    scd_args.normal_noise = False
    scd_args.verbose = True
    scd_args.normalize = True
    scd_args.batch_size = 256
    scd_args.percentile = False
    scd_args.sigmoid = True

    train_data, test_data, train_label, test_label = load_data('cifar10', 2)
    train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    np.random.seed(scd_args.seed)

    best_model, val_acc = train_single_cnn01(
        scd_args, None, None, (train_data, test_data, train_label, test_label))




