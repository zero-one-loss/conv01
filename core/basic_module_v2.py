"""
Conv01 neural networks' optimization function defined here
"""
import torch
import numpy as np
import torch.nn as nn


def init_bias(net, data, criterion, target, dtype, scd_args):
    train_loss = 5

    layers = list(net._modules.keys())
    # initialize layer in order

    for layer in layers[:-1]:
        # set bias as projection's mean
        projection = net(data, input_=layers[0], layer=layer+'_projection')
        if 'fc' in layer:
            net._modules[layer].bias = torch.nn.Parameter(
                projection.mean(dim=0), requires_grad=False)
        elif 'conv' in layer:
            net._modules[layer].bias = torch.nn.Parameter(
                projection.transpose(0, 1).reshape((projection.size(1), -1)).mean(dim=1)
                , requires_grad=False)

    # net._modules[layers[-1]].bias.data.zero_()

    # initial last fc layer
    # new_loss = init_bias_last_layer(net, data, layers, criterion,
    #                      target, dtype)

    # return new_loss
    return None


def init_bias_last_layer(net, data, layer, criterion, target, dtype, input_=None):
    p2 = net(data, input_=input_, layer=layer+'_projection')
    unique_p2 = torch.unique(p2, sorted=True).to(dtype=dtype)
    temp_bias = -1.0 * (
            unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

    new_projection = p2 + temp_bias.reshape((1, -1))
    yp = net(new_projection, input_=layer+'_ap')
    loss_group = criterion(yp, target)
    best_index = loss_group.argmin()
    best_bias = temp_bias[best_index]
    net._modules[layer].bias.data.fill_(best_bias)

    return loss_group[best_index].item(), best_bias


def update_final_layer_fc(net, layers, dtype, data_, scd_args, criterion, target,
                          device):

    layer = layers[0]
    previous_layer = layers[1]
    # Get the features fed into this layer

    data = net(data_, input_=layers[-1], layer=previous_layer + '_output')
    # Get the global bias for the new batch
    train_loss, global_bias = init_bias_last_layer(net, data,
                            layer, criterion, target, dtype, input_=layer)

    weights = net._modules[layer].weight
    num_input_features = weights.size(1)  # get input's dimension
    updated_features = min(num_input_features, scd_args.updated_fc_features)
    # shuffle updating order
    cords = np.random.choice(num_input_features, updated_features, False)

    best_w = weights.clone()
    # w_incs2 = torch.tensor([-1, 1]).type_as(w) * scd_args.w_inc2
    w_incs2 = -1

    w_ = torch.repeat_interleave(
        best_w, updated_features, dim=0)
    w_[np.arange(updated_features), cords] *= w_incs2

    # w_ = torch.cat([w_, -1.0 * w_], dim=1)

    temp_module = torch.nn.Conv1d(in_channels=1, out_channels=updated_features,
                                  kernel_size=weights.size(1)).to(dtype=dtype, device=device)

    temp_module.weight = nn.Parameter(w_.unsqueeze(dim=1))
    temp_module.bias.fill_(0)
    temp_module.requires_grad_(False)

    projection = temp_module(data.unsqueeze(dim=1))
    del temp_module
    new_projection, bias = update_fc_weight(projection, global_bias, scd_args)
    del projection

    yp = net(new_projection, input_=layer+'_ap')

    loss_group = criterion(yp, target.unsqueeze(1))
    del new_projection
    loss_group = loss_group.cpu().numpy()
    new_loss = loss_group.min()
    # print(train_loss, new_loss)
    if new_loss <= train_loss:
        row, col = np.unravel_index(loss_group.argmin(), loss_group.shape)
        net._modules[layer].weight = nn.Parameter(w_[row:row+1], requires_grad=False)
        net._modules[layer].bias.fill_(bias[row, col])

    del w_, loss_group, bias
    return min(new_loss, train_loss)


def update_fc_weight(projection, global_bias, scd_args):
    # projection = temp_module(data.unsqueeze(dim=1))
    p_t = projection.transpose(0, 1).squeeze(dim=2)
    bias_candidates = []
    num_temp_bias = []
    interval_candidates = []
    for i in range(p_t.size(0)):
        unique_p2 = torch.unique(p_t[i], sorted=True)
        temp_bias = -1.0 * (
            unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        bias_candidates.append(temp_bias)
        num_temp_bias.append(temp_bias.size(0))
        intervals = (temp_bias - global_bias).abs().argsort()
        interval_candidates.append(intervals)

    del p_t
    neighbor = min(scd_args.interval, min(num_temp_bias))
    bias_candidates = torch.stack(
        [temp[interval_candidates[i][:neighbor]]
         for i, temp in enumerate(bias_candidates)], dim=0)

    return projection + bias_candidates.unsqueeze(dim=0), bias_candidates


def init_fc(net, data, layer, criterion, target, dtype, idx):

    p2 = net(data, input_=layer, layer=layer+'_projection')

    # p2 shape nrows(1500) * nodes(20)
    unique_p2 = torch.unique(p2[:, idx], sorted=True).to(dtype=dtype)
    temp_bias = -1.0 * (
            unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
    del unique_p2
    new_projection = p2[:, idx:idx+1] + temp_bias.reshape((1, -1))
    # new_projection shape nrows(1500) * n_bias(508)
    new_p2 = torch.repeat_interleave(
        p2.unsqueeze(dim=1), new_projection.size(1), dim=1)  # nrows * n_bias * nodes
    del p2
    # replace original idx's projection by variations with different bias
    new_p2[:, :, idx] = new_projection
    nrows = new_p2.size(0)
    n_variations = new_p2.size(1)
    # switch dimension of nrows and n_bias, then flatten these two dimension
    new_projection = new_p2.transpose(0, 1).reshape((-1, new_p2.size(2)))  # n_bias * nrows * nodes
    del new_p2
    # get the final output, and reverse to original dimension order  nrows * n_bias * 1
    yp = net(new_projection, input_=layer+'_ap').reshape((n_variations, nrows, 1)).transpose(0, 1)
    del new_projection
    loss_group = criterion(yp, target.unsqueeze(dim=1)).squeeze(dim=1)
    best_index = loss_group.argmin()
    best_bias = temp_bias[best_index]
    net._modules[layer].bias[idx].fill_(best_bias)

    del temp_bias
    return loss_group[best_index], best_bias

def update_mid_layer_fc(net, layers, layer_index, data_, dtype, scd_args, criterion, target,
                        device):
    layer = layers[layer_index]
    previous_layer = layers[layer_index+1]
    batch_size = scd_args.batch_size
    data = net(data_, input_=layers[-1], layer=previous_layer+'_output')
    # projection = net(data, input_=layer, layer=layer + '_projection')
    for idx in np.random.permutation(net._modules[layer].weight.shape[0])[:scd_args.updated_nodes]:
        net._modules[layer].bias[idx].zero_()
        train_loss, global_bias = init_fc(net, data, layer, criterion,
                                          target, dtype, idx)

        weights = net._modules[layer].weight
        num_input_features = weights.size(1)  # get input's dimension
        updated_features = min(num_input_features, scd_args.updated_fc_features)
        # shuffle updating order
        cords = np.random.choice(num_input_features, updated_features, False)

        best_w = weights[idx:idx+1].clone()

        w_incs2 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc2
        if 'si' in layer:
            inc = []
            for i in range(w_incs2.shape[0]):
                w_inc = w_incs2[i]
                w_ = torch.repeat_interleave(
                    best_w, updated_features, dim=0)
                w_[np.arange(updated_features), cords] += w_inc
                inc.append(w_)
            w_ = torch.cat(inc, dim=0)
            del inc
            if scd_args.normalize:
                w_ /= w_.norm(dim=1, keepdim=True)
            # w_ = torch.cat([w_, -1.0 * w_], dim=1)
        else:
            w_incs2 = -1

            w_ = torch.repeat_interleave(
                best_w, updated_features, dim=0)
            w_[np.arange(updated_features), cords] *= w_incs2

        ic = updated_features * w_incs2.shape[0] if 'si' in layer else \
            updated_features

        temp_module = torch.nn.Conv1d(in_channels=1, out_channels=ic,
                kernel_size=weights.size(1)).to(dtype=dtype, device=device)
        temp_module.weight = nn.Parameter(w_.unsqueeze(dim=1))
        temp_module.bias.zero_()
        temp_module.requires_grad_(False)

        # projection's shape  nrows(1500) * ic(12) * 1
        projection = temp_module(data.unsqueeze(dim=1))
        del temp_module
        new_projection, bias = update_fc_weight(projection, global_bias,
                                                    scd_args)
        del projection
        n_batch = data_.size(0) // batch_size
        yps = []
        for i in range(n_batch):
            new_projection_batch = new_projection[
                    batch_size * i: batch_size * (i + 1)]

            n_r = new_projection_batch.size(0)  # 1500
            n_w = new_projection_batch.size(1)  # 12
            n_b = new_projection_batch.size(2)  # 20
            new_projection_batch = new_projection_batch.reshape((new_projection_batch.size(0), n_w*n_b))
            # new_projection 1500*12*20  bias 12*20
            # original projection feed into next layer
            projection_batch = net(
                data[batch_size * i: batch_size * (i + 1)],
                input_=layer, layer=layer+'_projection')  # 1500 * 20

            # replace projection[:, idx] after flatten variations
            projection_batch = torch.repeat_interleave(projection_batch.unsqueeze_(dim=1), n_w*n_b, dim=1)
            projection_batch[:, :, idx] = new_projection_batch
            del new_projection_batch
            projection_batch = projection_batch.transpose_(0, 1).reshape((-1, projection_batch.size(2)))

            yp = net(projection_batch, input_=layer+'_ap').reshape((n_w*n_b, n_r, -1))
            del projection_batch
            yp = yp.transpose_(0, 1).reshape((n_r, n_w, n_b))
            yps.append(yp)
        yps = torch.cat(yps, dim=0)
        loss_group = criterion(yps, target[:n_batch * batch_size].unsqueeze(dim=1))
        loss_group = loss_group.cpu().numpy()
        new_loss = loss_group.min()
        if new_loss <= train_loss:
            row, col = np.unravel_index(loss_group.argmin(), loss_group.shape)
            net._modules[layer].weight[idx] = nn.Parameter(w_[row], requires_grad=False)
            net._modules[layer].bias[idx].fill_(bias[row, col])

        del w_, loss_group, bias
        return min(new_loss, train_loss)


def init_conv(net, data, layer, criterion, target, dtype, idx, scd_args):

    p2 = net(data, input_=layer, layer=layer+'_projection')
    if not scd_args.percentile:
        unique_p2 = torch.unique(p2[:, idx], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (
                    unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

        if temp_bias.size(0) < scd_args.width:
            subset_bias = temp_bias
        else:
            percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
            subset_bias = torch.from_numpy(
                np.quantile(temp_bias.cpu(), percentile)).type_as(temp_bias)
        del unique_p2
    else:
        sorted_pti = p2[:, idx].flatten().sort()[0]
        percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
        unique_p2 = torch.from_numpy(
            np.quantile(sorted_pti.cpu(), percentile)).type_as(p2)
        subset_bias = -1.0 * (
                unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        del unique_p2

    new_projection = p2[:, idx:idx+1] + subset_bias.reshape((1, -1, 1, 1))
    # new_projection shape nrows(1500) * n_bias(100) * h(32) * w(32)
    new_p2 = torch.repeat_interleave(
        p2.unsqueeze(dim=1), subset_bias.size(0), dim=1)
    del p2
    # replace original idx's projection by variations with different bias
    new_p2[:, :, idx] = new_projection
    nrows = new_p2.size(0)
    n_variations = new_p2.size(1)
    # switch dimension of nrows and n_bias, then flatten these two dimension
    # n_bias * nrows * nodes * H * W
    new_projection = new_p2.transpose(0, 1).reshape(
        (-1, new_p2.size(2), new_p2.size(3), new_p2.size(4)))
    del new_p2
    # get the final output, and reverse to original dimension order  nrows * n_bias * 1
    yp = net(new_projection, input_=layer+'_ap').reshape((n_variations, nrows, 1)).transpose(0, 1)
    del new_projection
    loss_group = criterion(yp, target.unsqueeze(dim=1)).squeeze(dim=1)
    best_index = loss_group.argmin()
    best_bias = subset_bias[best_index]
    net._modules[layer].bias[idx].fill_(best_bias)

    del subset_bias
    return loss_group[best_index], best_bias




def update_mid_layer_conv(net, layers, layer_index, data, dtype, scd_args, criterion, target,
                        train_loss):
    layer = layers[layer_index]
    previous_layer = layers[layer_index+1]

    data = net(data, layer=previous_layer+'_output')
    train_loss, global_loss = init_conv()
    for idx in np.random.permutation(
            net._modules[layer].weight.shape[0])[:scd_args.updated_nodes]:
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


def update_first_layer_conv(net, layers, layer_index, data_, dtype, scd_args, criterion,
                            target, device):
    layer = layers[layer_index]
    data = data_
    batch_size = scd_args.batch_size
    for idx in np.random.permutation(
            net._modules[layer].weight.shape[0])[:scd_args.updated_nodes]:
        net._modules[layer].bias[idx].zero_()
        # Get the global bias for this batch
        train_loss, global_bias = init_conv(net, data, layer, criterion,
                                          target, dtype, idx, scd_args)

        weights = net._modules[layer].weight
        weight_size = weights.size()[1:]
        n_nodes = weight_size[0] * weight_size[1] * weight_size[2]
        updated_features = min(n_nodes, scd_args.updated_conv_features)
        cords_index = np.random.choice(n_nodes, updated_features, False)
        cords = []
        for i in range(weight_size[0]):
            for j in range(weight_size[1]):
                for k in range(weight_size[2]):
                    cords.append([i, j, k])
        cords = torch.tensor(cords)[cords_index]

        best_w = weights[idx:idx+1].clone()
        w_incs1 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc1
        if 'si' in layer:
            inc = []
            for i in range(w_incs1.shape[0]):
                w_inc = w_incs1[i]
                w_ = torch.repeat_interleave(
                    best_w, updated_features, dim=0)
                for i in range(updated_features):
                    w_[i, cords[i][0], cords[i][1], cords[i][2]] += w_inc
                inc.append(w_)
            w_ = torch.cat(inc, dim=0)
            del inc
            if scd_args.normalize:
                w_ /= w_.view((updated_features* w_incs1.shape[0], -1)).norm(dim=1).view((-1, 1, 1, 1))
            # w_ = torch.cat([w_, -1.0 * w_], dim=1)
        else:
            w_incs2 = -1

            w_ = torch.repeat_interleave(
                best_w, updated_features, dim=0)
            for i in range(updated_features):
                w_[i, cords[i][0], cords[i][1], cords[i][2]] *= w_incs2

        ic = updated_features * w_incs1.shape[0] if 'si' in layer else \
            updated_features

        temp_module = torch.nn.Conv2d(in_channels=data.size(1), out_channels=ic,
                kernel_size=list(weights.size()[2:]),
            padding=net._modules[layer].padding).to(dtype=dtype, device=device)
        temp_module.weight = nn.Parameter(w_)
        temp_module.bias.zero_()
        temp_module.requires_grad_(False)

        # projection's shape  nrows(1500) * ic(96) * H * W
        projection = temp_module(data)
        del temp_module
        new_projection, bias = update_conv_weight(projection, global_bias,
                                                  scd_args)
        del projection
        n_batch = data_.size(0) // batch_size
        yps = []
        for i in range(n_batch):
            new_projection_batch = new_projection[
                    batch_size * i: batch_size * (i + 1)]
            n_r = new_projection_batch.size(0)  # 1500
            n_w = new_projection_batch.size(1)  # 16
            n_b = new_projection_batch.size(2)  # 20
            height = new_projection_batch.size(3)  # 32
            width = new_projection_batch.size(4)
            new_projection_batch = new_projection_batch.reshape((n_r, n_w * n_b, height, width))
            # new_projection 1500*16*20  bias 16*20
            # original projection feed into next layer
            projection_batch = net(data[batch_size * i: batch_size * (i + 1)], input_=layer, layer=layer+'_projection')

            # replace projection[:, idx] after flatten variations
            projection_batch = torch.repeat_interleave(projection_batch.unsqueeze_(dim=1), n_w*n_b, dim=1)
            projection_batch[:, :, idx] = new_projection_batch
            del new_projection_batch
            projection_batch = projection_batch.transpose_(0, 1).reshape((-1, projection_batch.size(2), height, width))
            yp = net(projection_batch, input_=layer + '_ap').reshape((n_w * n_b, n_r, -1))
            del projection_batch
            yp = yp.transpose_(0, 1).reshape((n_r, n_w, n_b))
            yps.append(yp)
        yps = torch.cat(yps, dim=0)
        loss_group = criterion(yps, target[:n_batch * batch_size].unsqueeze(dim=1))
        loss_group = loss_group.cpu().numpy()
        new_loss = loss_group.min()
        if new_loss <= train_loss:
            row, col = np.unravel_index(loss_group.argmin(), loss_group.shape)
            net._modules[layer].weight[idx] = nn.Parameter(w_[row], requires_grad=False)
            net._modules[layer].bias[idx].fill_(bias[row, col])

        del w_, loss_group, bias
        return min(new_loss, train_loss)


def update_conv_weight(projection, global_bias, scd_args):

    p_t = projection.transpose(0, 1)
    p_t = p_t.reshape((p_t.size(0), -1))
    bias_candidates = []
    num_subset_bias = []
    interval_candidates = []
    for i in range(p_t.size(0)):
        if scd_args.percentile:
            sorted_pti = p_t[i].sort()[0]
            percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
            unique_p2 = torch.from_numpy(
                    np.quantile(sorted_pti.cpu(), percentile)).type_as(p_t)
            subset_bias = -1.0 * (
                    unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        else:
            unique_p2 = torch.unique(p_t[i], sorted=True)
            temp_bias = -1.0 * (
                    unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

            if temp_bias.size(0) < scd_args.width:
                subset_bias = temp_bias
            else:
                percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
                subset_bias = torch.from_numpy(
                    np.quantile(temp_bias.cpu(), percentile)).type_as(temp_bias)
        del unique_p2
        bias_candidates.append(subset_bias)
        num_subset_bias.append(subset_bias.size(0))
        intervals = (subset_bias - global_bias).abs().argsort()
        interval_candidates.append(intervals)

    del p_t
    neighbor = min(scd_args.interval, min(num_subset_bias))
    bias_candidates = torch.stack(
        [subset[interval_candidates[i][:neighbor]]
         for i, subset in enumerate(bias_candidates)], dim=0)
    new_projection = projection.unsqueeze(dim=2) + \
           bias_candidates.reshape((1, projection.size(1), neighbor, 1, 1))

    return new_projection, bias_candidates