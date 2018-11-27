import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torchvision.transforms as t
import pickle
import numpy as np
__all__ = torch
from . import torch


BATCH_SIZE = 6
EPOCH = 1


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, transform, train, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.data = []
        self.test = []

        # load train data
        for j in range(5):
            with open('/home/bryce/Documents/Datasets/cifar-10-batches-py/data_batch_%s' % str(j + 1), 'rb') as fo:
                dict_load = pickle.load(fo, encoding='bytes')
                for k in range(len(dict_load[b'labels'])):
                    data_now = dict_load[b'data'][k].reshape((3, 32, 32)).astype(np.uint8)
                    data_now = data_now.transpose((1, 2, 0))             # convert to HWC
                    self.data.append((data_now, dict_load[b'labels'][k]))

        # load test data
        with open('/home/bryce/Documents/Datasets/cifar-10-batches-py/test_batch', 'rb') as fo:
            dict_load = pickle.load(fo, encoding='bytes')
            for l in range(len(dict_load[b'labels'])):
                data_now = dict_load[b'data'][l].reshape((3, 32, 32)).astype(np.uint8)
                data_now = data_now.transpose((1, 2, 0))                 # convert to HWC
                self.test.append((data_now, dict_load[b'labels'][l]))

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index]     # train
        else:
            img, target = self.test[index]     # test

        if self.transform is not None:         # transform img
            img = self.transform(img)

        if self.target_transform is not None:  # transform target
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.data)
        else:
            return len(self.test)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        nn.init.xavier_normal_(self.fc1.weight)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        nn.init.xavier_normal_(self.fc2.weight)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.bn1(self.pool(f.relu(self.conv1(x))))
        x = self.bn2(self.pool(f.relu(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = self.bn3(f.relu(self.fc1(x)))
        x = self.bn4(f.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    # load data
    transform = t.Compose([
        t.ToTensor(),
        t.ToPILImage(),
        t.RandomCrop(28),
        t.ToTensor(),
        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_loader = torch.utils.data.DataLoader(
        CIFAR10(transform=transform, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        CIFAR10(transform=transform, train=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # load
            inputs, labels = data

            # zero grad
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 500 mini-batches
                print('[%d, %5d]  Loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # predict
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
