import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def sign(x):
    return torch.sign(x)

def signb(x):
    return (1 + torch.sign(x)) // 2
    

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(
                m.weight.view((m.weight.size(0), -1)),
                dim=1).view((-1, 1, 1, 1))
        )
        m.weight.requires_grad = False
        if m.bias is not None:
            init.constant_(m.bias, 0)
            m.bias.requires_grad = False

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1)
        # nn.init.zeros_(m.weight)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(m.weight, dim=1, keepdim=True))
        m.weight.requires_grad = False
        if m.bias is not None:
            m.bias.data.zero_()
            m.bias.requires_grad = False


def _01_init(model):
    for name, m in model.named_modules():
        if 'si' not in name:
            if isinstance(m, nn.Conv2d):
                m.weight = torch.nn.Parameter(
                    m.weight.sign()
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):
                m.weight = torch.nn.Parameter(
                    m.weight.sign())
                if m.bias is not None:
                    m.bias.data.zero_()


class LeNet_cifar(nn.Module):
    def __init__(self, num_classes=2, act=sign):
        super(LeNet_cifar, self).__init__()
        if act == 'sign':
            self.act = sign
        elif act == 'signb':
            self.act = signb
        self.conv1_si = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2_si = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1_si = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        previous_layer = []
        if not input_ or input_ == 'conv1_si':
            previous_layer.append(input_)
            out = self.conv1_si(x)
            if layer == 'conv1_si':
                return out
            out = self.act(out)
            out = F.avg_pool2d(out, 2)

        if input_ == 'conv2_si':
            out = x
            previous_layer.append(input_)
        if input_ in previous_layer:
            out = self.conv2_si(out)
            if layer == 'conv2_si':
                return out
            out = self.act(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
        if input_ == 'fc1_si':
            out = x
            previous_layer.append(input_)
        if input_ in previous_layer:
            out = self.fc1_si(out)
            if layer == 'fc1_si':
                return out
            out = self.act(out)
        if input_ == 'fc2':
            out = x
            previous_layer.append(input_)
        if input_ in previous_layer:
            out = self.fc2(out)
            if layer == 'fc2':
                return out
            out = self.act(out)
        if input_ == 'fc3':
            out = x
            previous_layer.append(input_)
        if input_ in previous_layer:
            out = self.fc3(out)
            if layer == 'fc3':
                return out
            out = signb(out)

        return out


class Toy(nn.Module):
    def __init__(self, num_classes=1, act=sign, sigmoid=False):
        super(Toy, self).__init__()
        if act == 'sign':
            self.act = sign
        elif act == 'signb':
            self.act = signb
        self.conv1_si = nn.Conv2d(3, 6, 5, padding=2)
        self.fc1_si = nn.Linear(6 * 8 * 8, 20)
        self.fc2 = nn.Linear(20, num_classes)
        self.signb = torch.sigmoid if sigmoid else signb
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # previous_layer = ['conv1_ap', 'fc1_si_ap', 'fc2_ap']
        #         # layers = ['conv1']
        status = -1
        layers = ['conv1_si', 'fc1_si', 'fc2']
        for items in layers:
            status += 1
            if input_ is None or items in input_:
                break

        if status < 1:
            if input_ != 'conv1_si_ap':
                # previous_layer.append(input_)
                out = self.conv1_si(x)
            if layer == 'conv1_si_projection':
                return out
            if input_ == 'conv1_si_ap':
                out = x
                # previous_layer.append(input_)
            out = self.act(out)
            out = F.avg_pool2d(out, 4)
            out = out.view((out.size(0), -1))
            if layer == 'conv1_si_output':
                return out
        if input_ == 'fc1_si':
            out = x
            # previous_layer.append(input_)
        if status < 2:
            if input_ != 'fc1_si_ap':
                out = self.fc1_si(out)
            if layer == 'fc1_si_projection':
                return out
            if input_ == 'fc1_si_ap':
                out = x
            out = self.act(out)
            if layer == 'fc1_si_output':
                return out
        if input_ == 'fc2':
            out = x
            # previous_layer.append(input_)
        if status < 3:
            if input_ != 'fc2_ap':
                out = self.fc2(out)
            if layer == 'fc2_projection':
                return out
            if input_ == 'fc2_ap':
                out = x
            out = self.signb(out)

        return out


if __name__ == '__main__':
    net = Toy(1, act='sign')
    x = torch.rand(size=(100, 3, 32, 32))
    # x = torch.rand(size=(100, 6, 16, 16))
    output = net(x)  # shape 100, 1
