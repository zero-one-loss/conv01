import numpy as np
import pickle
import sys
from torchvision.models.resnet import resnet50, resnet18
sys.path.append('..')
from tools import args, save_checkpoint, print_title, load_data
import torch.nn as nn

import time
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from core.cnn import Cifar10CNN1, Cifar10CNN2, Toy
from tools.model_args import ModelArgs
import skorch.tests.test_net
# from core.resnet import ResNet18


if __name__ == '__main__':
    # set flag
    scd_args = ModelArgs()
    scd_args.seed = 2018
    scd_args.num_classes = 2
    scd_args.dataset = 'cifar10'
    scd_args.num_iters = 100
    scd_args.lr = 0.001
    scd_args.batch_size = 64
    scd_args.target = 'toy_cnn'
    # Set Random Seed

    # print information
    et, vc = print_title()

    np.random.seed(scd_args.seed)
    torch.random.manual_seed(scd_args.seed)
    save = True if scd_args.save else False

    train, test, train_label, test_label = load_data(scd_args.dataset, scd_args.num_classes)

    train = train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
    test = test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
    # train = train.reshape((-1, 3, 48, 48)).astype(np.float32)
    # test = test.reshape((-1, 3, 48, 48)).astype(np.float32)
    # train = train.reshape((-1, 3, 96, 96)).astype(np.float32)
    # test = test.reshape((-1, 3, 96, 96)).astype(np.float32)
    # train = train.reshape((-1, 3, 224, 224)).astype(np.float32)
    # test = test.reshape((-1, 3, 224, 224)).astype(np.float32)
    train_label = train_label.astype(np.int64)
    test_label = test_label.astype(np.int64)
    print('training data size: ')
    print(train.shape)
    print('testing data size: ')
    print(test.shape)

    valid_ds = Dataset(test, test_label)

    scd = NeuralNetClassifier(
        Toy,
        classes=scd_args.num_classes,
        max_epochs=scd_args.num_iters,
        lr=scd_args.lr,
        criterion=torch.nn.CrossEntropyLoss,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        # train_split=predefined_split(valid_ds),
        batch_size=scd_args.batch_size,
        optimizer=torch.optim.Adam,
        device='cuda',
        verbose=1,
        warm_start=False,

    )
    # # scd = CNN(scd_args.round, SimpleNet_gtsrb)
    # # scd = CNN(scd_args.round, LeNet_gtsrb)
    # # scd = CNN(scd_args.round, SimpleNet_celeba)
    # # scd = CNN(scd_args.round, LeNet_celeba)
    # # scd = CNN(scd_args.round, SimpleNet_cifar)
    # scd = CNN(scd_args.round, LeNet_cifar)
    # # scd = CNN(scd_args.round, ResNet18)
    # # models = []
    # # for i in range(scd_args.round):
    # #     model = resnet50()
    # #     model._modules['conv1'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # #     model._modules['fc'] = nn.Linear(2048, scd_args.n_classes, bias=True)
    # #
    # #     models.append(model)
    # # scd = CNN(scd_args.round, models)
    # # scd = CNN(scd_args.round, resnet50)
    # # scd = CNN(scd_args.round, resnet18)
    a = time.time()
    scd.fit(train, train_label)

    print('Cost: %.3f seconds' % (time.time() - a))

    print('Best Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    # print('Vote Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test)))
    # print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test)))

    if save:
        save_path = 'checkpoints'
        save_checkpoint(scd, save_path, scd_args.target, et, vc)
    # del scd