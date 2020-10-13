import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
           init.constant_(m.bias, 0)
           

class Cifar10CNN1(nn.Module):
    def __init__(self, num_classes=2):
        super(Cifar10CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.conv1_ds = nn.Conv2d(8, 16, 2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.conv2_ds = nn.Conv2d(16, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
        self.conv3_ds = nn.Conv2d(32, 64, 2, stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv1_ds(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2_ds(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv3_ds(out))
        out = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)))
        out = out.view((out.size(0), out.size(1)))
        out = self.fc(out)

        return out

class Cifar10CNN2(nn.Module):
    def __init__(self, num_classes=2):
        super(Cifar10CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        self.conv1_ds = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.conv2_ds = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        # self.conv3_ds = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_ds(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_ds(out)
        out = F.relu(self.conv3(out))
        # out = self.conv3_ds(out)
        out = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)))
        out = out.view((out.size(0), out.size(1)))
        out = self.fc(out)

        return out


class LeNet_cifar(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        # x = F.pad(x, 2)
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Toy(nn.Module):
    def __init__(self, num_classes=2):
        super(Toy, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.fc1 = nn.Linear(6 * 8 * 8, 20)
        self.fc2 = nn.Linear(20, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    # net = Cifar10CNN2(2)
    x = torch.randn((100, 3, 32, 32))
    net = Toy(2)
    output = net(x)