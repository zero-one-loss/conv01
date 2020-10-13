"""
Conv01 neural networks' optimization function defined here
"""
import torch
import numpy as np

def init_bias_last_layer(net, data, layers, criterion, target, dtype, train_loss):
    p2 = net(data, layer=layers[-1]).flatten()
    # sorted_p2, sorted_index = torch.sort(p2)
    # temp_p2 = sorted_p2.clone()
    # temp_bias = (sorted_p2 + torch.cat([temp_p2[0:1] - 0.1, temp_p2[:-1]])) / 2
    # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
    unique_p2 = torch.unique(p2, sorted=True).to(dtype=dtype)
    temp_bias = -1.0 * (
            unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

    for bias in temp_bias:
        net._modules[layers[-1]].bias.data.fill_(bias)
        output = net(data)
        loss = criterion(output, target).item()
        if loss < train_loss:
            best_bias = bias.data.item()
            # print('current loss: %.5f' % loss)
            train_loss = loss

    net._modules[layers[-1]].bias.data.fill_(best_bias)


def init_mid_layer_fc(net, layer, dtype, data, criterion, target, train_loss):
    for i in np.random.permutation(net._modules[layer].weight.shape[0]):
        net._modules[layer].bias[i].zero_()
        p1 = net(data, layer=layer)
        # sorted_p1, sorted_index = torch.sort(p1[:, i])
        # temp_p1 = sorted_p1.clone()
        # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
        # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
        unique_p1 = torch.unique(p1[:, i], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2

        for bias in temp_bias:
            net._modules[layer].bias[i].fill_(bias)
            output = net(data)
            loss = criterion(output, target).item()
            # print(loss)
            if loss < train_loss:
                best_bias = bias.data.item()
                # print('current loss: %.5f' % loss)
                train_loss = loss
                # best_idx = temp_index[bias_idx]
        # print(best_bias)
        net._modules[layer].bias[i].fill_(best_bias)


def init_mid_layer_conv(net, layer, dtype, data, criterion, target, train_loss):
    for i in np.random.permutation(net._modules[layer].weight.shape[0]):
        net._modules[layer].bias[i].zero_()
        p1 = net(data, layer=layer)
        p1 = p1.transpose(0, 1).reshape((p1.size(1), -1))
        # sorted_p1, sorted_index = torch.sort(p1[:, i])
        # temp_p1 = sorted_p1.clone()
        # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
        # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)

        unique_p1 = torch.unique(p1[i], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2

        for bias in temp_bias:
            net._modules[layer].bias[i].fill_(bias)
            output = net(data)
            loss = criterion(output, target).item()
            # print(loss)
            if loss < train_loss:
                best_bias = bias.data.item()
                # print('current loss: %.5f' % loss)
                train_loss = loss
                # best_idx = temp_index[bias_idx]
        # print(best_bias)
        net._modules[layer].bias[i].fill_(best_bias)


def init_first_layer_conv(net, layer, dtype, data, criterion, target,
                          scd_args, train_loss):
    for i in np.random.permutation(net._modules[layer].weight.shape[0]):
        net._modules[layer].bias[i].zero_()
        p1 = net(data, layer=layer)
        p1 = p1.transpose(0, 1).reshape((p1.size(1), -1))
        # sorted_p1, sorted_index = torch.sort(p1[:, i])
        # temp_p1 = sorted_p1.clone()
        # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
        # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
        unique_p1 = torch.unique(p1[i], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (unique_p1 + torch.cat(
            [unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2
        # sorted_temp_bias, sorted_temp_bias_index = torch.sort(temp_bias)
        for bias in temp_bias[::scd_args.width]:
            net._modules[layer].bias[i].fill_(bias)
            output = net(data)
            loss = criterion(output, target).item()
            # print(loss)
            if loss < train_loss:
                best_bias = bias.data.item()
                # print('current loss: %.5f' % loss)
                train_loss = loss
                # best_idx = temp_index[bias_idx]
        # print(best_bias)
        net._modules[layer].bias[i].fill_(best_bias)


def update_final_layer_fc(net, layers, dtype, data, scd_args, criterion, target
                          ):
    train_loss = 5
    best_bias = net._modules[layers[-1]].bias.data.item()
    net._modules[layers[-1]].bias.data.zero_()

    p2 = net(data, layer=layers[-1]).flatten()
    # sorted_p2, sorted_index = torch.sort(p2)
    # sorted_label = target[sorted_index]
    # temp_p2 = sorted_p2.clone()
    # temp_bias = (sorted_p2 + torch.cat([temp_p2[0:1] - 0.1, temp_p2[:-1]])) / 2
    #
    # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)

    unique_p2 = torch.unique(p2, sorted=True).to(dtype=dtype)
    temp_bias = -1.0 * (
            unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

    for bias_idx, bias in enumerate(temp_bias[::scd_args.width]):
        net._modules[layers[-1]].bias.data.fill_(bias)
        output = net(data)
        loss = criterion(output, target).item()
        if loss < train_loss:
            best_bias = bias.data.item()
            # print('current loss: %.5f' % loss)
            train_loss = loss
            # best_idx = temp_index[bias_idx]

    net._modules[layers[-1]].bias.data.fill_(best_bias)
    global_bias = best_bias

    updated_features = net._modules[layers[-1]].weight.shape[1]
    # cords = np.random.choice(net.fc1.weight.size(1), updated_features, False)
    cords = np.random.permutation(updated_features)
    w = net._modules[layers[-1]].weight.clone()
    best_w = w[0].clone()
    # w_incs2 = torch.tensor([-1, 1]).type_as(w) * scd_args.w_inc2
    w_incs2 = -1

    w_ = torch.repeat_interleave(
        w.reshape((-1, 1)), updated_features, dim=1)
    w_[cords, np.arange(updated_features)] *= w_incs2

    # w_ = torch.cat([w_, -1.0 * w_], dim=1)

    for i in range(updated_features):
        net._modules[layers[-1]].weight[0] = w_[:, i]
        net._modules[layers[-1]].bias.data.zero_()
        p2 = net(data, layer=layers[-1]).flatten()
        # sorted_p2, sorted_index = torch.sort(p2)
        # sorted_label = target[sorted_index]
        # temp_p2 = sorted_p2.clone()
        # temp_bias = (sorted_p2 + torch.cat([temp_p2[0:1] - 0.1, temp_p2[:-1]])) / 2
        # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
        unique_p2 = torch.unique(p2, sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (
                unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

        intervals = (temp_bias - global_bias).abs().argsort()[:scd_args.interval]

        for bias_idx, bias in enumerate(temp_bias[intervals]):
            net._modules[layers[-1]].bias.data.fill_(bias)
            output = net(data)
            loss = criterion(output, target).item()
            # print(loss)
            if loss < train_loss:
                best_bias = bias.data.item()
                # print('current loss: %.5f' % loss)
                train_loss = loss
                # best_idx = temp_index[bias_idx]
                best_w = w_[:, i]

    net._modules[layers[-1]].bias.data.fill_(best_bias)
    net._modules[layers[-1]].weight[0] = best_w


def update_mid_layer_fc(net, layer, data, dtype, scd_args, criterion, target,
                        train_loss):
    for idx in np.random.permutation(net._modules[layer].weight.shape[0])[:scd_args.updated_nodes]:
        # train_loss = 5
        best_bias = net._modules[layer].bias[idx].data.item()
        net._modules[layer].bias[idx].zero_()
        p1 = net(data, layer=layer)
        # sorted_p1, sorted_index = torch.sort(p1[:, idx])
        # temp_p1 = sorted_p1.clone()
        # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
        # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
        unique_p1 = torch.unique(p1[:, idx], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2


        for bias_idx, bias in enumerate(temp_bias[::scd_args.width]):
            net._modules[layer].bias[idx].fill_(bias)
            output = net(data)
            loss = criterion(output, target).item()
            # print(loss)
            if loss < train_loss:
                best_bias = bias.data.item()
                # print('current loss: %.5f' % loss)
                train_loss = loss
                # best_idx = bias_idx * scd_args.interval
        # print(best_bias)
        net._modules[layer].bias[idx].fill_(best_bias)
        global_bias = best_bias

        updated_features = min(net._modules[layer].weight.size(1), scd_args.updated_features)
        cords = np.random.choice(net._modules[layer].weight.size(1), updated_features, False)
        # cords = np.random.permutation(updated_features)
        w = net._modules[layer].weight.clone()
        best_w = w[idx].clone()
        w_incs1 = torch.tensor([-1, 1]).type_as(w) * scd_args.w_inc1
        if 'si' in layer:
            inc = []
            for i in range(w_incs1.shape[0]):
                w_inc = w_incs1[i]
                w_ = torch.repeat_interleave(
                    best_w.reshape((-1, 1)), updated_features, dim=1)
                w_[cords, np.arange(updated_features)] += w_inc
                inc.append(w_)
            w_ = torch.cat(inc, dim=1)
            del inc
            w_ /= w_.norm(dim=0)
            # w_ = torch.cat([w_, -1.0 * w_], dim=1)
        else:
            w_incs2 = -1

            w_ = torch.repeat_interleave(
                best_w.reshape((-1, 1)), updated_features, dim=1)
            w_[cords, np.arange(updated_features)] *= w_incs2

        ic = updated_features * w_incs1.shape[0] if 'si' in layer else \
            updated_features
        for i in range(ic):
            net._modules[layer].weight[idx] = w_[:, i]
            net._modules[layer].bias[idx].data.zero_()
            p1 = net(data, layer=layer)
            # sorted_p1, sorted_index = torch.sort(p1[:, idx])
            # temp_p1 = sorted_p1.clone()
            # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
            # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
            unique_p1 = torch.unique(p1[:, idx], sorted=True).to(dtype=dtype)
            temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2

            # intervals = (torch.arange(temp_bias.shape[0]) - best_idx).abs().argsort()[:scd_args.interval]
            intervals = (temp_bias - global_bias).abs().argsort()[:scd_args.interval]
            for bias_idx, bias in enumerate(temp_bias[intervals]):
                net._modules[layer].bias[idx].data.fill_(bias)
                output = net(data)
                loss = criterion(output, target).item()
                # print(loss)
                if loss < train_loss:
                    best_bias = bias.data.item()
                    # print('current loss: %.5f' % loss)
                    train_loss = loss
                    # best_idx = temp_index[bias_idx]
                    best_w = w_[:, i]

        net._modules[layer].bias[idx].data.fill_(best_bias)
        net._modules[layer].weight[idx] = best_w


def update_mid_layer_conv(net, layer, data, dtype, scd_args, criterion, target,
                        train_loss):
    for idx in np.random.permutation(net._modules[layer].weight.shape[0])[:scd_args.updated_nodes]:
        # train_loss = 5
        best_bias = net._modules[layer].bias[idx].data.item()
        net._modules[layer].bias[idx].zero_()
        p1 = net(data, layer=layer)
        p1 = p1.transpose(0, 1).reshape((p1.size(1), -1))
        # sorted_p1, sorted_index = torch.sort(p1[:, idx])
        # temp_p1 = sorted_p1.clone()
        # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
        # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
        unique_p1 = torch.unique(p1[idx], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2

        for bias_idx, bias in enumerate(temp_bias):
            net._modules[layer].bias[idx].fill_(bias)
            output = net(data)
            loss = criterion(output, target).item()
            # print(loss)
            if loss < train_loss:
                best_bias = bias.data.item()
                # print('current loss: %.5f' % loss)
                train_loss = loss
                # best_idx = bias_idx * scd_args.interval
        # print(best_bias)
        net._modules[layer].bias[idx].fill_(best_bias)
        global_bias = best_bias
        weight_size = net._modules[layer].weight.size()[1:]
        n_nodes = weight_size[0] * weight_size[1] * weight_size[2]

        updated_features = min(n_nodes, scd_args.updated_features)
        cords_index = np.random.choice(n_nodes, updated_features, False)
        cords = []
        for i in range(weight_size[0]):
            for j in range(weight_size[1]):
                for k in range(weight_size[2]):
                    cords.append([i, j, k])
        cords = torch.tensor(cords)[cords_index]

        # cords = np.random.permutation(updated_features)
        w = net._modules[layer].weight.clone()
        best_w = w[idx].clone()
        w_incs1 = torch.tensor([-1, 1]).type_as(w) * scd_args.w_inc1
        if 'si' in layer:
            inc = []
            for i in range(w_incs1.shape[0]):
                w_inc = w_incs1[i]
                w_ = torch.repeat_interleave(
                    best_w.unsqueeze(dim=0), updated_features, dim=0)
                # w_[cords, np.arange(updated_features)] += w_inc
                for i in range(updated_features):
                    w_[i, cords[i][0], cords[i][1], cords[i][2]] += w_inc
                inc.append(w_)
            w_ = torch.cat(inc, dim=0)
            del inc
            w_ /= w_.view((updated_features* w_incs1.shape[0], -1)).norm(dim=1).view((-1, 1, 1, 1))
            # w_ = torch.cat([w_, -1.0 * w_], dim=1)
        else:
            w_incs2 = -1

            w_ = torch.repeat_interleave(
                best_w.unsqueeze(dim=0), updated_features, dim=0)
            # w_[cords, np.arange(updated_features)] *= w_incs2
            for i in range(updated_features):
                w_[i, cords[i][0], cords[i][1], cords[i][2]] *= w_incs2

        ic = updated_features * w_incs1.shape[0] if 'si' in layer else \
            updated_features
        for i in range(ic):
            net._modules[layer].weight[idx] = w_[i]
            net._modules[layer].bias[idx].data.zero_()
            p1 = net(data, layer=layer)
            # sorted_p1, sorted_index = torch.sort(p1[:, idx])
            # temp_p1 = sorted_p1.clone()
            # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
            # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
            unique_p1 = torch.unique(p1[:, idx], sorted=True).to(dtype=dtype)
            temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2

            # intervals = (torch.arange(temp_bias.shape[0]) - best_idx).abs().argsort()[:scd_args.interval]
            intervals = (temp_bias - global_bias).abs().argsort()[:scd_args.interval]
            for bias_idx, bias in enumerate(temp_bias[intervals]):
                net._modules[layer].bias[idx].data.fill_(bias)
                output = net(data)
                loss = criterion(output, target).item()
                # print(loss)
                if loss < train_loss:
                    best_bias = bias.data.item()
                    # print('current loss: %.5f' % loss)
                    train_loss = loss
                    # best_idx = temp_index[bias_idx]
                    best_w = w_[i]

        net._modules[layer].bias[idx].data.fill_(best_bias)
        net._modules[layer].weight[idx] = best_w


def update_first_layer_conv(net, layer, data, dtype, scd_args, criterion,
                            target, train_loss):
    for idx in np.random.permutation(
            net._modules[layer].weight.shape[0])[:]:
        # train_loss = 5
        best_bias = net._modules[layer].bias[idx].data.item()
        net._modules[layer].bias[idx].zero_()
        p1 = net(data, layer=layer)
        p1 = p1.transpose(0, 1).reshape((p1.size(1), -1))
        # sorted_p1, sorted_index = torch.sort(p1[:, idx])
        # temp_p1 = sorted_p1.clone()
        # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
        # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
        unique_p1 = torch.unique(p1[idx], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2

        for bias_idx, bias in enumerate(temp_bias[::scd_args.width]):
            net._modules[layer].bias[idx].fill_(bias)
            output = net(data)
            loss = criterion(output, target).item()
            # print(loss)
            if loss < train_loss:
                best_bias = bias.data.item()
                # print('current loss: %.5f' % loss)
                train_loss = loss
                # best_idx = bias_idx * scd_args.interval
        # print(best_bias)
        net._modules[layer].bias[idx].fill_(best_bias)
        global_bias = best_bias
        weight_size = net._modules[layer].weight.size()[1:]
        n_nodes = weight_size[0] * weight_size[1] * weight_size[2]

        updated_features = min(n_nodes, scd_args.updated_features)
        cords_index = np.random.choice(n_nodes, updated_features, False)
        cords = []
        for i in range(weight_size[0]):
            for j in range(weight_size[1]):
                for k in range(weight_size[2]):
                    cords.append([i, j, k])
        cords = torch.tensor(cords)[cords_index]

        # cords = np.random.permutation(updated_features)
        w = net._modules[layer].weight.clone()
        best_w = w[idx].clone()
        w_incs1 = torch.tensor([-1, 1]).type_as(w) * scd_args.w_inc1

        inc = []
        for i in range(w_incs1.shape[0]):
            w_inc = w_incs1[i]
            w_ = torch.repeat_interleave(
                best_w.unsqueeze(dim=0), updated_features, dim=0)
            # w_[cords, np.arange(updated_features)] += w_inc
            for i in range(updated_features):
                w_[i, cords[i][0], cords[i][1], cords[i][2]] += w_inc
            inc.append(w_)
        w_ = torch.cat(inc, dim=0)
        del inc
        w_ /= w_.view(
            (updated_features * w_incs1.shape[0], -1)
        ).norm(dim=1).view((-1, 1, 1, 1))
        # w_ = torch.cat([w_, -1.0 * w_], dim=1)

        ic = updated_features * w_incs1.shape[0]
        for i in range(ic):
            net._modules[layer].weight[idx] = w_[i]
            net._modules[layer].bias[idx].data.zero_()
            p1 = net(data, layer=layer)
            # sorted_p1, sorted_index = torch.sort(p1[:, idx])
            # temp_p1 = sorted_p1.clone()
            # temp_bias = (sorted_p1 + torch.cat([temp_p1[0:1] - 0.1, temp_p1[:-1]])) / 2
            # temp_bias = -1.0 * torch.unique(temp_bias.float(), sorted=True).to(dtype=dtype)
            unique_p1 = torch.unique(p1[:, idx], sorted=True).to(dtype=dtype)
            temp_bias = -1.0 * (unique_p1 + torch.cat([unique_p1[0:1] - 0.1, unique_p1[:-1]])) / 2

            # intervals = (torch.arange(temp_bias.shape[0]) - best_idx).abs().argsort()[:scd_args.interval]
            intervals = (temp_bias - global_bias).abs().argsort()[:scd_args.interval]
            for bias_idx, bias in enumerate(temp_bias[intervals]):
                net._modules[layer].bias[idx].data.fill_(bias)
                output = net(data)
                loss = criterion(output, target).item()
                # print(loss)
                if loss < train_loss:
                    best_bias = bias.data.item()
                    # print('current loss: %.5f' % loss)
                    train_loss = loss
                    # best_idx = temp_index[bias_idx]
                    best_w = w_[i]

        net._modules[layer].bias[idx].data.fill_(best_bias)
        net._modules[layer].weight[idx] = best_w

        return train_loss