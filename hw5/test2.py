# CMSC 426: This is the starting point for HW5. It was prepared by Luyu with
# some tweaks from Carlos. If you find issues with the starting point code
# please post them on Piazza.
#
# Usually I run the code like this:
#
# python .\starting_point.py --data_path C:\users\carlos\Downloads\hw3-dataset\ --lr 0.01 --epochs 50
#
# But you need to experiment on what set of learning rate and epochs works best.

import argparse
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from PIL import Image
from torch.autograd import Variable

# This is an example of an MLP in PyTorch. It has 24576 inputs, 10 hidden layers
# and 2 outputs and it is connected to a softmax/cross entropy loss loss for
# training purposes.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*64*128, 10)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = x.view(-1, 24576)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
        self.conv2= nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)

        self.fc1 = nn.Linear(in_features=3360, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, t):
        # conv layers
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # linear + output layer
        t = t.reshape(-1, 3360)
        t = F.relu(self.fc1(t))
        t = self.fc2(t)

        return t

class dataset(object):
    def __init__(self, path):
        # neg is 0, pos is 1
        pos_im = glob.glob(path + '/pos/*.png')
        neg_im = glob.glob(path + '/neg/*.png')
        img = [(x, 1) for x in pos_im]
        img = img + [(x, 0) for x in neg_im]
        random.shuffle(img)
        self.data = img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index][0]).resize((64, 128))
        img = np.array(img).transpose((2, 0, 1))[:3]
        img = img / 255. - 0.5
        img = torch.from_numpy(img).float()
        label = self.data[index][1]
        return img, label

def train(model, loader, optimizer, criterion, epoch, device, losses_arr, accs_arr):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        accuracies.update(accuracy)

    losses_arr.append(losses.avg)
    accs_arr.append(accuracies.avg)
    print('Train: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))


def test(model, loader, criterion, epoch, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        losses.update(loss.item())
        accuracies.update(accuracy)
    print('Test: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='code')
    parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.01, help='momentum')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Net()
    model = CNN()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = torch.utils.data.DataLoader(dataset(args.data_path + '/train_64x128_H96/'), batch_size=64, shuffle=True, num_workers=4)
    # train_loader = torch.utils.data.DataLoader(dataset('/Users/rohithvenkatesh/Downloads/hw3-dataset/train_64x128_H96/'), batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset(args.data_path + '/test_64x128_H96/'), batch_size=64, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, device, train_losses, train_accs)
        test(model, test_loader, criterion, epoch, device)

    # code I used to get graphs
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Plots of Accuracy and Loss over Epochs (optimized: lr=0.1)', fontsize=12)

    axs[0].plot(range(args.epochs), train_accs)
    axs[0].set_title('Training Accuracy w/ max: ' + str(round(max(train_accs).item(), 4)))
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')

    axs[1].plot(range(args.epochs), train_losses)
    axs[1].set_title('Training Loss w/ min: ' + str(round(min(train_losses), 4)))
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')

    plt.savefig('optimized.png')

    torch.save(model.state_dict(), 'trained_model.pt')