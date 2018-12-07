import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torchtext import data
import glob
__all__ = torch


BATCH_SIZE = 128
EPOCH = 1
LEARNING_RATE = 0.01


def aclimbd(text_field, label_field):

    def txt_split(txt_path):
        split = 0
        with open("imdb%s" % txt_path) as txt:
            txt = txt.readlines()
            for content in txt:
                content = str.lower(content)
                content.lower()
                for old in [",", ":", '"', "-", "<br />", ".", "!", "?", ".", "(", ")", "*"]:
                    content = str.replace(content, old, " ")
                split = content.split(sep=" ")
                k = 0
                while k < len(split):
                    if split[k] == '':
                        del split[k]
                        continue
                    else:
                        k += 1
        return split

    train_examples = []
    test_examples = []
    fields = [("text", text_field), ("label", label_field)]

    # 读取所有txt
    text_name = glob.glob('imdb/aclImdb/train/pos/*.txt')
    text_name.extend(glob.glob('imdb/aclImdb/train/neg/*.txt'))

    # 初始化数组
    text_array = np.zeros((len(text_name), 2)).astype(dtype=str)

    # 把txt名和label放到数组
    for j in range(len(text_name)):
        text_array[j, 0] = text_name[j][4:]
        if text_name[j][19] == 'n':
            text_array[j, 1] = 'neg'
        else:
            text_array[j, 1] = 'pos'

    # shuffle
    np.random.shuffle(text_array)

    # 读取文本+split
    for j in range(len(text_name)):
        # train
        if j <= 0.7 * len(text_name):
            train_examples.append(data.Example.
                                  fromlist([txt_split(text_array[j, 0]), text_array[j, 1]], fields))
        # test
        else:
            test_examples.append(data.Example.
                                 fromlist([txt_split(text_array[j, 0]), text_array[j, 1]], fields))

    train_in = data.Dataset(train_examples, fields)
    test_in = data.Dataset(test_examples, fields)
    return train_in, test_in


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 100)
        self.fc = nn.Linear(20000, 8)

    def forward(self, x):
        x = self.word_embeddings(x).view(x.size(0), -1)
        x = f.softmax(self.fc(x), dim=1)
        return x


if __name__ == '__main__':
    # load
    TEXT = data.Field(lower=True, batch_first=True, fix_length=200)
    LABEL = data.Field(sequential=False)
    # train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL, root='./')
    train, test = aclimbd(TEXT, LABEL)

    # build vocab
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # iter
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=128, shuffle=True,
                                                       repeat=False, sort=False)

    net = CNN()
    device = torch.device("cuda")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data in enumerate(train_iter):
            # load
            inputs = data.text
            labels = data.label
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
        for i, data in enumerate(test_iter):
            inputs = data.text
            labels = data.label

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
