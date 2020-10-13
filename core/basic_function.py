import time



def evaluation(data_loader, use_cuda, device, dtype, net, key):
    a = time.time()
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        outputs = net(data).round()


        correct += outputs.eq(target).sum().item()

    acc = correct / len(data_loader.dataset)
    print("{} Accuracy: {:.5f}, "
          "cost {:.2f} seconds".format(key, acc, time.time() - a))

    return acc