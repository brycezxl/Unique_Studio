import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
import torch.utils.data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import GloVe
from torchtext import data
import torch
import torchtext
import pickle
import numpy as np
__all__ = torch


# get_dataset构造并返回Dataset所需的examples和fields
# def get_dataset(train=True):
#
#     text_field = data.Field(sequential=True, lower=True)
#     label_field = data.Field(sequential=False, use_vocab=False)
#
#     # id数据对训练在训练过程中没用，使用None指定其对应的field
#     fields = [("id", None), ("comment_text", text_field), ("toxic", label_field)]
#     examples = []
#
#     for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
#             examples.append(data.Example.fromlist([None, text, label], fields))
#
#     return examples, fields
#
#
# # 得到构建Dataset所需的examples和fields
# train_examples, train_fields = get_dataset(train=True)
# test_examples, test_fields = get_dataset(train=False)
#
# # 构建Dataset数据集
# train_data = data.Dataset(train_examples, train_fields)
# test_data = data.Dataset(test_examples, test_fields)

# field
# text_in = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=200)
# label = data.Field(sequential=False, use_vocab=False)
#
# # create iterator
# dataset = torchtext.datasets.IMDB(text_field=text_in, label_field=label, path=r"/home/bryce/Documents/Datasets")
# train, test = dataset.iters(device=-1, root=r"/home/bryce/Documents/Datasets")

# 词表
# text.build_vocab(train)
# label.build_vocab(train)


class LSTM(nn.Module):

    def __init__(self, weight_matrix, text):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(text.vocab), 300)  # embedding之后的shape: torch.Size([200, 8, 300])
        # 若使用预训练的词向量，需在此处指定预训练的权重
        self.word_embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]  # lstm_out:200x8x128
        # 取最后一个时间步
        final = lstm_out[-1]  # 8*128
        y = self.decoder(final)  # 8*2
        return y


def data_iter(text_in, label_in):
    dataset = torchtext.datasets.IMDB(text_field=text_in, label_field=label_in, path=r"/home/bryce/Documents/Datasets")
    train_in, test_in = dataset.iters(device=-1, root=r"/home/bryce/Documents/Datasets")

    # build the vocabulary`
    text_in.build_vocab(train_in, )
    label_in.build_vocab(train_in)
    weight_matrix = text_in.vocab.vectors
    # 若只针对训练集构造迭代器
    # train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
    # train_iter = data.BucketIterator.splits(
    #         train_in,  # 构建数据集所需的数据集
    #         batch_sizes=(8, 8),
    #         # 如果使用gpu，此处将-1更换为GPU的编号
    #         device=-1,
    #         # the BucketIterator needs to be told what function it should use to group the data.
    #         sort_key=lambda x: len(x.comment_text),
    #         sort_within_batch=False,
    # )
    # test_iter = Iterator(test_in, batch_size=8, sort=False, sort_within_batch=False, repeat=False)
    return train_in, test_in, weight_matrix


def main():
    # field
    text = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=200)
    label = data.Field(sequential=False, use_vocab=False)

    train, test = torchtext.datasets.IMDB.splits(text, label)

    # build the vocabulary
    text.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    label.build_vocab(train)

    # iter
    train_iter, test_iter, weight_matrix = data_iter(text, label)

    model = LSTM(weight_matrix, text)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_function = f.cross_entropy

    for epoch, batch in enumerate(train_iter):
        optimizer.zero_grad()
        predicted = model(batch.comment_text)

        loss = loss_function(predicted, batch.toxic)
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == '__main__':
    main()
