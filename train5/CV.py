import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torchvision.transforms as t
import torch.utils.data
import torch
import pickle
import numpy as np
__all__ = torch


BATCH_SIZE = 64
EPOCH = 50
LEARNING_RATE = 0.003


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, transforms, train, target_transform=None):
        self.transform = transforms
        self.target_transform = target_transform
        self.train = train
        self.data = []
        self.test = []

        # load train data
        for j in range(5):
            with open('cifar-10-batches-py/data_batch_%s' % str(j + 1), 'rb') as fo:
                dict_load = pickle.load(fo, encoding='bytes')
                for k in range(len(dict_load[b'labels'])):
                    data_now = dict_load[b'data'][k].reshape((3, 32, 32)).astype(np.uint8)
                    data_now = data_now.transpose((1, 2, 0))             # convert to HWC
                    self.data.append((data_now, dict_load[b'labels'][k]))

        # load test data
        with open('cifar-10-batches-py/test_batch', 'rb') as fo:
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


class CNN(nn.Module):
    def __init__(self):                # 32*32*3
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv1.weight)    # 15
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)             # 14
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)  # 7
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)  # 7
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=2048)  # 13 13
        nn.init.kaiming_normal_(self.fc1.weight)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.bn7 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=10)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.bn1(self.pool(f.relu(self.conv1(x))))
        x = self.bn2(self.pool(f.relu(self.conv2(x))))
        x = self.bn3(f.relu(self.conv3(x)))
        x = self.bn4(f.relu(self.conv4(x)))
        x = self.bn5(f.relu(self.conv5(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = self.bn6(f.relu(self.fc1(x)))
        x = self.bn7(f.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def main():
    # load data
    transform_train = t.Compose([
        t.ToPILImage(),
        t.ToTensor(),
        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = t.Compose([
        t.ToPILImage(),
        t.ToTensor(),
        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_loader = torch.utils.data.DataLoader(
        CIFAR10(transforms=transform_train, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        CIFAR10(transforms=transform_test, train=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    device = torch.device("cuda")

    net = CNN()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # load
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero grad
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d]  Loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    # predict
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    torch.save(net.state_dict(), 'cnn')  # 储存整个网络


if __name__ == '__main__':
    main()
