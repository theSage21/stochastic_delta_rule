import torch
import random
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Net:
    def __init__(self, idim, hdim, odim):
        self.idim = idim
        self.odim = odim
        self.means = [torch.randn(idim, hdim)*0.1,
                      torch.randn(hdim, hdim)*0.1,
                      torch.randn(hdim, odim)*0.1]
        self.devs = [torch.rand(idim, hdim),
                     torch.rand(hdim, hdim),
                     torch.rand(hdim, odim)]
        self.w = None

    def zero_grads(self):
        if self.w is not None:
            for w in self.w:
                if w.grad is not None:
                    w.grad.zero_()

    def forward(self, input):
        samples = []
        for i, (m, s) in enumerate(zip(self.means, self.devs)):
            d = torch.distributions.normal.Normal(m, s).sample()
            d.requires_grad = True
            input = torch.sigmoid(input @ d)
            samples.append(d)
        self.w = samples
        return input

    def update(self, alpha, beta, gamma):
        assert self.w is not None
        assert gamma < 1
        means, devs = [], []
        for w, mean, dev in zip(self.w, self.means, self.devs):
            mean += alpha * (- w.grad)
            dev += beta * torch.abs(w.grad)
            dev *= gamma
            means.append(mean)
            devs.append(dev)
        self.means = means
        self.devs = devs
        self.w = None


class NormalNet(nn.Module):
    def __init__(self, idim, hdim, odim):
        super().__init__()
        self.l1 = nn.Linear(idim, hdim, bias=False)
        self.l2 = nn.Linear(hdim, hdim, bias=False)
        self.l3 = nn.Linear(hdim, odim, bias=False)

    def forward(self, i):
        i = torch.sigmoid(self.l1(i))
        i = torch.sigmoid(self.l2(i))
        i = torch.sigmoid(self.l3(i))
        return i


logstep = 1
idim, hdim, odim, bs = 10, 50, 1, 2000000
alpha = 0.9
beta = 0.9
gamma = 0.999
# DATASET
x = [i for i in tqdm(itertools.islice(itertools.product([0, 1], repeat=idim), 2*bs),
                     total=min(2**idim, 2*bs))]
random.shuffle(x)
x = np.array(x)
print('Dataset size', x.shape)

mask = np.random.random(x.shape[0]) < 0.7

t_x = torch.from_numpy(x[mask]).float()
d_x = torch.from_numpy(x[~mask]).float()
t_y = torch.unsqueeze((torch.sum(t_x, dim=1) % 2 == 0).float(), 1)
d_y = torch.unsqueeze((torch.sum(d_x, dim=1) % 2 == 0).float(), 1)

# Network
net = Net(idim, hdim, odim)
net2 = NormalNet(idim, hdim, odim)
criterion = nn.MSELoss()
opt = optim.Adam(net2.parameters())

stoc_writer = SummaryWriter('logs/stochastic')
norm_writer = SummaryWriter('logs/normal')

# Train

def count():
    i = 0
    while True:
        yield i
        i += 1


for epoch in tqdm(count()):
    net.zero_grads()

    p = net.forward(t_x)
    loss = criterion(p, t_y)
    loss.backward()
    net.update(alpha, beta, gamma)
    if epoch % logstep == 0:
        stoc_writer.add_histogram('std-dev', np.concatenate([i.numpy().flatten() for i in net.devs]), epoch)
        stoc_writer.add_histogram('means', np.concatenate([i.numpy().flatten() for i in net.means]), epoch)
        stoc_writer.add_scalar('train-loss', loss, epoch)

    p = net.forward(d_x)
    loss = criterion(p, d_y)
    if epoch % logstep == 0:
        stoc_writer.add_scalar('dev-loss', loss, epoch)

    net2.zero_grad()

    p = net2(t_x)
    loss = criterion(p, t_y)
    loss.backward()
    if epoch % logstep == 0:
        norm_writer.add_scalar('train-loss', loss, epoch)

    opt.step()

    p = net2(d_x)
    loss = criterion(p, d_y)
    if epoch % logstep == 0:
        norm_writer.add_scalar('dev-loss', loss, epoch)
