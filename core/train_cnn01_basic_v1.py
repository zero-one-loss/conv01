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
import pickle
import pandas as pd
# Args assignment
# scd_args = ModelArgs()
#
# scd_args.nrows = args.nrows
# scd_args.nfeatures = args.nfeatures
# scd_args.w_inc = args.w_inc
# scd_args.tol = 0.00000
# scd_args.local_iter = args.iters
# scd_args.num_iters = args.num_iters
# scd_args.interval = args.interval
# scd_args.rounds = args.round
# scd_args.w_inc1 = args.w_inc1
# scd_args.updated_features = args.updated_features
# scd_args.n_jobs = args.n_jobs
# scd_args.num_gpus = args.num_gpus
# scd_args.adv_train = True if args.adv_train else False
# scd_args.eps = args.eps
# scd_args.w_inc2 = args.w_inc2
# scd_args.hidden_nodes = args.hidden_nodes
# scd_args.evaluation = False if args.no_eval else True
# scd_args.verbose = True if args.verbose else False
# scd_args.b_ratio = args.b_ratio
# scd_args.cuda = True if torch.cuda.is_available() else False
# scd_args.seed = args.seed
# scd_args.target = args.target
# scd_args.source = args.source
# scd_args.save = True if args.save else False
# scd_args.resume = True if args.resume else False
# scd_args.criterion = ZeroOneLoss
# if args.version == 'linear':
#     scd_args.structure = LeNet_cifar
#
# scd_args.dataset = args.dataset
# scd_args.num_classes = args.n_classes
# scd_args.gpu = args.gpu
# scd_args.fp16 = True if args.fp16 else False
# scd_args.act = args.act
# scd_args.updated_nodes = args.updated_nodes
# scd_args.width = args.width
# scd_args.normal_noise = True if args.normal_noise else False


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
    batch_size = int(train_data.shape[0] * scd_args.nrows)
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
                                              torch.from_numpy(train_label.astype(np.int64)))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
                                             torch.from_numpy(test_label.astype(np.int64)))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,  num_workers=0,
                                               pin_memory=True,
                                               sampler=BalancedBatchSampler(trainset, torch.from_numpy(train_label)))
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,  num_workers=0,
    #                                            pin_memory=False, shuffle=True,)
    val_loader = torch.utils.data.DataLoader(trainset, batch_size=train_data.shape[0], shuffle=False, num_workers=0,
                                             pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_data.shape[0], shuffle=False, num_workers=0,
                                              pin_memory=False)

    net = scd_args.structure(num_classes=1, act=scd_args.act)
    _01_init(net)
    best_model = scd_args.structure(num_classes=1, act=scd_args.act)
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

    def test():
        a = time.time()
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            outputs = net(data)

            predicted = outputs.flatten()
            correct += predicted.eq(target).sum().item()

        acc = correct / len(test_loader.dataset)
        print('Test_accuracy: %0.5f' % acc)
        print('This epoch cost %0.2f seconds' % (time.time() - a))

        return acc

    def val():
        a = time.time()
        correct = 0
        # yp = []
        for batch_idx, (data, target) in enumerate(val_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            outputs = net(data)

            predicted = outputs.flatten()
            # yp.append(predicted)
            # correct += predicted.eq(target).sum().item()
            ba_acc = balanced_accuracy_score(target.cpu().numpy(), predicted.cpu().numpy())
            ub_acc = accuracy_score(target.cpu().numpy(), predicted.cpu().numpy())
        # acc = correct / len(val_loader.dataset)
        print('Train accuracy: %0.5f, balanced accuracy: %.5f' % (ub_acc, ba_acc))
        print('This epoch cost %0.2f seconds' % (time.time() - a))

        return ub_acc

    for epoch in range(scd_args.num_iters):
        # if epoch % 200 == 0:
        #     criterion = HingeLoss(c=args.c + epoch / 200)
        print('\nEpoch: %d' % epoch)
        a = time.time()

        layers = list(net._modules.keys())
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
                for layer in layers[:-1]:
                    projection = net(data, layer=layer)
                    if 'fc' in layer:
                        net._modules[layer].bias = torch.nn.Parameter(
                            projection.mean(dim=0), requires_grad=False)
                    elif 'conv' in layer:
                        net._modules[layer].bias = torch.nn.Parameter(
                            projection.transpose(0, 1).reshape((projection.size(1), -1)).mean(dim=1)
                        , requires_grad=False)

                net._modules[layers[-1]].bias.data.zero_()

                # initial last fc layer
                init_bias_last_layer(net, data, layers, criterion,
                                     target, dtype, train_loss)

                # update h2 -> h-1
                if len(layers) > 2:
                    for layer_index, layer in enumerate(layers[1:-1]):
                        # initialize dense layer's bias
                        if 'fc' in layer:
                            init_mid_layer_fc(net, layer,  dtype, data,
                                              criterion, target, train_loss)
                        # initialize convolution layer's bias
                        elif 'conv' in layer:

                            init_mid_layer_conv(net, layer, dtype, data,
                                                criterion, target, train_loss)
                # update h1
                for layer_index, layer in enumerate(layers[:1]):
                    init_first_layer_conv(net, layer, dtype, data, criterion,
                                          target, scd_args, train_loss)

            # update final layer
            update_final_layer_fc(net, layers, dtype, data, scd_args, criterion,
                                  target)
            # update H2->H-1 layer
            train_loss = 5
            # p1 = net(data, layer='p1')
            if len(layers) > 2:
                for layer_index, layer in enumerate(layers[1:-1]):
                    if 'fc' in layer:
                        update_mid_layer_fc(net, layer, data, dtype, scd_args,
                                            criterion, target, train_loss)
                    elif 'conv' in layer:
                        update_mid_layer_conv(net, layer, data, dtype, scd_args, criterion, target,
                        train_loss)


            train_loss = 5
            # update H1
            for layer_index, layer in enumerate(layers[:1]):
                train_loss = update_first_layer_conv(net, layer, data, dtype,
                                    scd_args, criterion, target, train_loss)

            print('current loss: %.5f' % train_loss)
            print('This epoch cost %0.2f seconds' % (time.time() - a))

            val_acc = val()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model.load_state_dict(net.state_dict())
            test_acc = test()
            temp_row = pd.Series(
                {'epoch': epoch+1,
                 'train acc': val_acc,
                 'test acc':test_acc,
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

    scd_args.nrows = 0.001
    scd_args.nfeatures = 1
    scd_args.w_inc = 0.17
    scd_args.tol = 0.00000
    scd_args.local_iter = 1
    scd_args.num_iters = 100
    scd_args.interval = 20
    scd_args.rounds = 1
    scd_args.w_inc1 = 0.17
    scd_args.updated_features = 6
    scd_args.n_jobs = 1
    scd_args.num_gpus = 1
    scd_args.adv_train = False
    scd_args.eps = 0.1
    scd_args.w_inc2 = 0.2
    scd_args.hidden_nodes = 20
    scd_args.evaluation = True
    scd_args.verbose = True
    scd_args.b_ratio = 0.2
    scd_args.cuda = True
    scd_args.seed = 2018
    scd_args.target = 'sdf'
    scd_args.source = None
    scd_args.save = False
    scd_args.resume = False
    scd_args.criterion = ZeroOneLoss
    scd_args.structure = LeNet_cifar
    scd_args.dataset = 'cifar10'
    scd_args.num_classes = 2
    scd_args.gpu = 1
    scd_args.fp16 = True
    scd_args.act = 'sign'
    scd_args.updated_nodes = 10
    scd_args.width = 1000
    scd_args.normal_noise = False

    train_data, test_data, train_label, test_label = load_data('cifar10', 2)
    train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    np.random.seed(scd_args.seed)

    # scd = SCDBinary()
    # scd.set_args(scd_args)
    best_model, val_acc = train_single_cnn01(scd_args, None, None, (train_data, test_data, train_label, test_label))
    # scd.set_model(best_model)
    # scd.add_model(best_model)
    # scd.val_accs = val_acc
    # if scd_args.save:
    #     scd.save('checkpoints', scd_args.target)



