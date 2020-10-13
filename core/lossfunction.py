import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ZeroOneLoss(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, kind='balanced', num_classes=2):
        super(ZeroOneLoss, self).__init__()
        self.kind = kind
        self.n_classes = num_classes

        return

    def forward(self, inputs, target):

        loss = 1 - inputs.eq(target).float().mean(dim=0)
        return loss

class BCELoss(nn.Module):
    """
    Multi dimension BCE loss
    """

    def __init__(self, ):
        super(BCELoss, self).__init__()

        return

    def forward(self, inputs, target):
        """

        :param inputs: nrows * -1
        :param target: nrows * 1
        :return:
        """
        prob_1 = inputs - 1e-20
        prob_0 = 1 - inputs + 1e-20
        loss = -1 * (target * prob_1.log() + (1 - target) * prob_0.log()).mean(dim=0)

        return loss

class ZeroOneLossMC(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, kind='balanced', num_classes=2):
        super(ZeroOneLossMC, self).__init__()
        self.kind = kind
        self.n_classes = num_classes

        return

    def forward(self, inputs, target):
        # inputs.float()
        if self.kind == 'balanced':
            inputs_onehot = inputs
            target_onehot = F.one_hot(target, num_classes=self.n_classes)
            # match = (inputs_onehot + target_onehot) // 2
            # loss = 1 - (match.sum(dim=0) / target_onehot.sum(dim=0).float()).mean()
            match = (1 - (inputs_onehot - target_onehot).abs().mean(dim=1)).mean()
            return match

        elif self.kind == 'combined':
            inputs_onehot = F.one_hot(inputs.flatten().long(), num_classes=self.n_classes)
            target_onehot = F.one_hot(target, num_classes=self.n_classes)
            match = (inputs_onehot + target_onehot) // 2
            balanced_loss = 1 - (match.sum(dim=0) / target_onehot.sum(dim=0).float()).mean()
            imbalanced_loss = 1 - inputs.flatten().long().eq(target).float().mean()
            loss = balanced_loss + imbalanced_loss

        return loss



class HingeLoss(nn.Module):
    """Hinge-Loss

    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(HingeLoss, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = 2.0 * target - 1
        loss = self.c - target * inputs
        loss[loss < 0] = 0
        loss = loss.mean()



        return loss

class ConditionalEntropy(nn.Module):
    """Hinge-Loss

    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(ConditionalEntropy, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        unique_target, target_counts = torch.unique(target, return_counts=True)
        target_probs = target_counts.float() / target.size(0)
        sum = 0
        for i, unique_value in enumerate(unique_target):
            sequence = inputs.flatten()[target == unique_value]
            sequence_prob = torch.unique(sequence, return_counts=True)[1].float() / sequence.size(0)
            sum += target_probs[i] * Categorical(probs=sequence_prob).entropy()



        return sum

class HingeLoss3(nn.Module):
    """Hinge-Loss

    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(HingeLoss3, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = 2.0 * target - 1
        loss = self.c - (target * inputs).abs()
        loss[loss < 0] = 0
        loss1 = loss.mean()

        inputs_ = inputs.sign()
        loss2 = 1 - inputs_.eq(target).float().mean()
        return loss1


class HingeLoss2(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(HingeLoss2, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = 2.0 * target - 1
        loss = -1.0 * target * inputs
        # loss[loss < 0] = 0
        loss = loss.mean()
        return loss

class CombinedLoss(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1):
        super(CombinedLoss, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c
        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = (2.0 * target - 1)
        loss = 1 - target * inputs
        loss[loss < 0] = 0
        loss  = loss
        loss_h = loss.mean()

        inputs = (torch.sign(inputs) + 1 ) // 2
        inputs_onehot = F.one_hot(inputs.flatten().long(), num_classes=self.n_classes)
        target_onehot = F.one_hot(target, num_classes=self.n_classes)
        match = (inputs_onehot + target_onehot) // 2
        loss_01 = 1 - (match.sum(dim=0) / target_onehot.sum(dim=0).float()).mean()

        return loss_h + loss_01


if __name__ == '__main__':
    import torch.nn as nn

    m = nn.Sigmoid()
    loss = nn.BCELoss()
    input = torch.randn(3, requires_grad=True)
    import torch

    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(m(input), target)
    print(output)
    loss = BCELoss()
    output = loss(m(input), target)
    print(output)

