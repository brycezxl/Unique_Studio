import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchtext import data
import glob
__all__ = torch


BATCH_SIZE = 128
EPOCH = 10
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
    text_name = glob.glob(r'imdb/aclImdb/train/pos/*.txt')
    text_name.extend(glob.glob(r'imdb/aclImdb/train/neg/*.txt'))

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


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 200)
        self.lstm = nn.LSTM(input_size=200, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 2)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.word_embeddings(x)
        x, _ = self.lstm(x, None)
        x = f.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # load
    TEXT = data.Field(lower=True, batch_first=True, fix_length=200)
    LABEL = data.Field(sequential=False)
    train, test = aclimbd(TEXT, LABEL)

    # build vocab
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # iter
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=BATCH_SIZE, shuffle=True,
                                                       repeat=False, sort=False)

    net = RNN()
    device = torch.device("cuda")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    a = net.parameters()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        running_loss = 0.0

        for i, data in enumerate(train_iter):
            # load
            inputs = data.text
            labels = data.label - 1
            # labels = labels.view(BATCH_SIZE, -1)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero grad
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print loss
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d]  Loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    # predict
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_iter:
            inputs = data.text
            labels = data.label - 1
            # labels = labels.view(BATCH_SIZE, -1)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
