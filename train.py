from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from myModel import UNet
from tqdm import tqdm


def get_dataset():
    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    trainloader = torch.utils.data.DataLoader(my_trainset,
    batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader


def myfunc(rank, world_size):
    torch.cuda.set_device(rank)

    # DDP：DDP backend初始化
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456',
                         world_size = world_size,rank=rank)

    # 准备数据，要在DDP初始化之后进行
    trainloader = get_dataset()

    # 构造模型
    model = UNet().to('cuda')

    # DDP: 构造DDP model
    model = DDP(model, device_ids=[rank])

    # DDP: 要在构造DDP model之后，才能用model初始化optimizer。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # 假设我们的loss是这个
    loss_func = nn.CrossEntropyLoss().to('cuda')

    ### 3. 网络训练  ###
    model.train()
    iterator = tqdm(range(100))
    for epoch in iterator:
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        trainloader.sampler.set_epoch(epoch)
        # 后面这部分，则与原来完全一致了。
        for data, label in trainloader:
            data, label = data.to('cuda'), label.to('cuda')
            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_func(prediction, label)
            loss.backward()
            iterator.desc = "loss = %0.3f" % loss
            optimizer.step()
