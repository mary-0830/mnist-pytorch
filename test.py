from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import numpy 
import numpy as np
import math
import torchvision
import matplotlib.pyplot as plt

class pytorch_CNNet(nn.Module):
    def __init__(self):
        super(pytorch_CNNet, self).__init__()
        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 32, 3, 1)),
            ('rellu2', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(32, 64, 3, 1)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(64, 64, 3, 1)),
            ('rellu4', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2))
        ]))
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 200), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(200, 10))

    def forward(self, x):
        output = self.cnn(x)
        # output = output.permute((0, 2, 3, 1))
        # output = output.contiguous().view(-1, 4 * 4 * 64)
        output = output.view(-1, 4 * 4 * 64)
        output = self.fc(output)
        return F.log_softmax(output, dim=1)

class pytorch_Net(nn.Module):
    def __init__(self):
        super(pytorch_Net, self).__init__()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.conv2d_1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2d_2 = nn.Conv2d(32, 32, 3, 1)
        self.conv2d_3 = nn.Conv2d(32, 64, 3, 1)
        self.conv2d_4 = nn.Conv2d(64, 64, 3, 1)

        self.dense_1 = nn.Linear(4 * 4 * 64, 10)
        self.dense_2 = nn.Linear(200, 200)
        self.dense_3 = nn.Linear(200, 10)


    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2d_3(x))
        x = F.relu(self.conv2d_4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.permute((0, 2, 3, 1))

        x = x.contiguous().view(-1, 4 * 4 * 64)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return F.log_softmax(x, dim=1)


class pytorch_keras_Net(nn.Module):
    def __init__(self):
        super(pytorch_keras_Net, self).__init__()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.conv2d_1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2d_2 = nn.Conv2d(32, 32, 3, 1)
        self.conv2d_3 = nn.Conv2d(32, 64, 3, 1)
        self.conv2d_4 = nn.Conv2d(64, 64, 3, 1)

        self.dense_1 = nn.Linear(4 * 4 * 64, 200)
        self.dense_2 = nn.Linear(200, 200)
        self.dense_3 = nn.Linear(200, 10)


    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2d_3(x))
        x = F.relu(self.conv2d_4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.permute((0, 2, 3, 1))

        x = x.contiguous().view(-1, 4 * 4 * 64)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return F.log_softmax(x, dim=1)


# def cross_entropy(Y, P):
#     sum=0.0
#     P = torch.tensor(P)
#     P = F.softmax(P, dim=1)
#     for x in map(lambda y,p:-(1-y)*numpy.log(1-p)-y*numpy.log(p),Y.cpu().numpy(),P.cpu().numpy()):
#         sum+=x.sum().mean()
#     return sum/len(Y)
 
def cross_entropy(P, Y):
    count = Y.size(0)
    
    loss = 0.0
    for x, l in zip(P, Y):  # 并行遍历
        # print(-1 * x[l])
        # print(torch.log(torch.exp(x).sum()))
        loss += -1 * x[l] + torch.log(torch.exp(x).sum())
    return loss / count


def train(args, model, device, train_loader, optimizer, epoch, tb_writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_flip = torch.flip(data, [2, 3])  # 上下翻转
        data_flip = torch.rot90(data, 1, [2, 3]) # 旋转90度
        data_change = torch.cat((data, data_flip), 0) # 维度[2,1,28,28]
        target_change = torch.cat((target, target))
        data_, target_ = data_change.to(device), target_change.to(device)
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_)
             
        # print("data:", data.shape)
        # print("target:", target.shape)
        # loss = F.nll_loss(output, target_)
        # loss.backward()
        # loss = nn.CrossEntropyLoss()
        # loss = loss(output, target_)
        # loss.backward()
        loss = cross_entropy(output, target_)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            
            # tb_writer.add_scalar('loss', loss.item(), epoch*len(train_loader) + batch_idx)
            tb_writer.add_scalar('loss', loss.item(), epoch*len(train_loader) + batch_idx)

        # 旋转可视化
        images = next(iter(data_flip))
        images_example = torchvision.utils.make_grid(images)
        images_example = images_example.numpy().transpose(1,2,0)
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]
        images_example = images_example * std + mean
        # plt.imshow(images_example)
        # # plt.show()
        # plt.savefig("flip_{}.png".format(batch_idx))



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print("______data.shape:", data.shape)
            output = model(data)
            # test_loss = nn.CrossEntropyLoss()
            # test_loss = test_loss(output, target).item()
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # test_loss = cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--load_keras', type=bool, default=False,
                        help='if the model is load from keras')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([
                        # transforms.RandomRotation(90),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                       ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if args.load_keras:
        model = pytorch_keras_Net.to(device)
        model.load_state_dict(torch.load("pyt_model.pt"))
    else:
        model = pytorch_CNNet().to(device)
        # model.load_state_dict(torch.load("mnist_cnn.pt"))
        # 删除网络参数后，使用下面的代码
        # model = pytorch_CNNet().cuda()
        # model.load_state_dict(torch.load("mnist_deleted.pt"))
    for name, param in model.named_parameters():
        print("name:", name)
        print("param:", param.shape)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    tb_writer = SummaryWriter()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, tb_writer)
        test(args, model, device, test_loader)
    
    # tb_writer.add_scalar('loss', loss.item(), epoch*len(train_loader) + batch_idx)
    tb_writer.close()
    # print("Run 'tensorboard --logdir=./' to view tensorboard at http://localhost:6006/")

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")   # 只保存模型参数

    images, labels = next(iter(train_loader))
    images_example = torchvision.utils.make_grid(images)
    images_example = images_example.numpy().transpose(1,2,0)
    
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    images_example = images_example * std + mean
    # plt.imshow(images_example )
    # plt.show()
    
if __name__ == '__main__':
    main()