import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchtext import data
import glob
import re
__all__ = torch

BATCH_SIZE = 64
EPOCH = 10
LEARNING_RATE = 0.0003
DIM = 300


def get_race(train_data=1):
    """
    读取数据集装载成dataset
    """
    examples = []
    fields = [("answers", ANSWERS), ("options1", OPTIONS), ("options2", OPTIONS), ("options3", OPTIONS),
              ("options4", OPTIONS), ("questions", QUESTIONS), ("articles", ARTICLES)]

    # train or dev
    if train_data:
        txt_path = r"RACE/train/middle/"
    else:
        txt_path = r"RACE/dev/middle/"

    text_name = glob.glob(r"%s*.txt" % txt_path)
    max_question = 0
    max_option = 0
    max_artcile = 0
    for p in range(len(text_name)):
        # 分别打开每一个txt
        with open("%s" % text_name[p]) as file:
            txt = str.lower(file.readlines()[0])                       # 小写

            # 取出答案
            answers = re.sub(r'"options": ', 'の', txt)                # 找到结束标记换成`
            answers = re.findall(r'"answers":[^の]*', answers)[0]      # 根据`正则查找
            answers = re.sub(r"answers|\W", ' ', answers)              # 去掉奇怪字符
            answers = answers.split(sep=' ')                           # 按空格分词
            while '' in answers:                                       # 去掉空字符
                answers.remove('')

            # 取出选项
            options = re.sub(r', "questions":', 'の', txt)             # 同上
            options = re.findall(r', "options": [^の]*', options)[0]
            options = re.sub(r', "options":', ' ', options)
            options = options.split(sep='"], ["')
            for l in range(len(options)):
                options[l] = options[l].split(sep='", "')
                for m in range(len(options[l])):
                    options[l][m] = re.sub(r"\W", ' ', options[l][m])
                    options[l][m] = options[l][m].split(sep=' ')
                    while '' in options[l][m]:
                        options[l][m].remove('')

            # 取出问题
            questions = re.sub(r', "article":', 'の', txt)                     # 同上
            questions = re.findall(r', "questions": [^の]*', questions)[0]
            questions = re.sub(r', "questions":', '', questions)
            questions = questions.split(sep='", "')
            for l in range(len(questions)):
                questions[l] = re.sub(r"\W", ' ', questions[l])
                questions[l] = questions[l].split(sep=" ")
                while '' in questions[l]:
                    questions[l].remove('')

            # 取出文章
            article_in = re.sub(r'", "id":', "の", txt)                         # 找到article结尾，换成#便于搜索
            article_in = re.findall(r'"article":[^の]*', article_in)[0]            # 找出article部分
            article_in = re.sub(r"\\n|article|[^0-9A-Za-zの ]", " ", article_in)   # 去掉article，换行符和一些奇奇怪怪的符号，换成空格
            article_in = re.sub(r" {2,100}", " ", article_in)                      # 把多于2个的空格合并成一个
            article_in = re.sub(r"^ | $", "", article_in)                          # 去掉头尾空格

            # 每个问题加一次
            for l in range(len(answers)):
                examples.append(data.Example.fromlist([answers[l], options[l][0], options[l][1], options[l][2],
                                                       options[l][3], questions[l], article_in], fields))

    examples = data.Dataset(examples, fields)
    return examples


class MRU(nn.Module):
    def __init__(self):
        super(MRU, self).__init__()

        # MRU Encoder
        self._r_mru_encoder = torch.tensor([1, 2, 4, 10, 25], requires_grad=False)
        self._w_mru_encoder = [torch.rand((DIM, DIM), requires_grad=True) /
                               6 for _ in range(len(self._r_mru_encoder))]                            # 每个r一个w
        self._b_mru_encoder = [torch.rand((1, DIM), requires_grad=True) /
                               6 for _ in range(len(self._r_mru_encoder))]                            # 每个r一个b
        self._w1_mru_encoder = (torch.rand((3, len(self._r_mru_encoder))
                                          , requires_grad=True) / 6).to(device)                       # 串联后的神经网络
        self._b1_mru_encoder = (torch.rand((3, DIM), requires_grad=True) / 6).to(device)
        self._w2_mru_encoder = (torch.rand((1, 3), requires_grad=True) / 6).to(device)
        self._b2_mru_encoder = (torch.rand((1, DIM), requires_grad=True) / 6).to(device)
        self._wz_mru_encoder = (torch.rand((DIM, DIM), requires_grad=True) / 6).to(device)              # 循环网络
        self._bz_mru_encoder = (torch.rand((1, DIM), requires_grad=True) / 6).to(device)              # 循环网络
        self._wo_mru_encoder = (torch.rand((DIM, DIM), requires_grad=True) / 6).to(device)  # 循环网络
        self._bo_mru_encoder = (torch.rand((1, DIM), requires_grad=True) / 6).to(device)  # 循环网络

        self.embedding = nn.Embedding(num_embeddings=len(ARTICLES.vocab), embedding_dim=DIM)
        pass

    def forward(self, option1_in, option2_in, option3_in, option4_in, question_in, article_in):

        # embedding
        article_in = self.embedding(article_in)
        option1_in = self.embedding(option1_in)
        option2_in = self.embedding(option2_in)
        option3_in = self.embedding(option3_in)
        option4_in = self.embedding(option4_in)
        question_in = self.embedding(question_in)

        # MRU Encoding
        article_in = self._mru(article_in)

        # BiAttention

    def _mru(self, inputs_mru):
        gate_mru = self._mru_multi_ranged_reasoning(inputs_mru)
        outputs_mru = self._mru_encoding(gate_mru, inputs_mru)
        return outputs_mru

    def _mru_contract_expand(self, inputs_c_e):
        new_inputs_c_e = 1

        # contract-expand
        for r, w, b in zip(self._r_mru_encoder, self._w_mru_encoder, self._b_mru_encoder):

            # gpu
            r = r.to(device)
            w = w.to(device)
            b = b.to(device)

            # contract
            groups = int(inputs_c_e.size(0) / r)  # 整除组
            rest = inputs_c_e.size(0) - r * groups  # 余数组
            word_temp = torch.zeros(((groups + int(rest != 0)), DIM))  # 临时储存压缩数组
            word_new = torch.zeros_like(inputs_c_e).to(device)  # 展开后的新数组

            # 整除组contract + 处理
            for group in range(groups):
                sum_group = 0  # 求出r组词的和
                for p in range((r * group), (r * (group + 1))):
                    sum_group += inputs_c_e[p, :]
                sum_group = sum_group.unsqueeze(0).to(device)
                word_temp[group, :] = f.relu(sum_group.mm(w)) + b  # 对r组词的和操作一番
            # 余数组contract + 处理
            if rest != 0:
                sum_group = 0
                for p in range((r * groups), (r * groups + rest)):
                    sum_group += inputs_c_e[p, :]
                sum_group = sum_group.unsqueeze(0).to(device)
                word_temp[groups, :] = f.relu((sum_group * r / rest).mm(w)) + b

            # 整除组expand
            for group in range(groups):
                for num in range(r):
                    word_new[(r * group + num), :] = word_temp[group, :] / r
            # 余数组expand
            if rest != 0:
                for num in range(rest):
                    word_new[(r * groups + num), :] = word_temp[groups, :] / r

            # 处理完成并入大数组
            if isinstance(new_inputs_c_e, int):
                new_inputs_c_e = word_new
            else:
                new_inputs_c_e = torch.cat((new_inputs_c_e, word_new), 0)

        return new_inputs_c_e

    def _mru_multi_ranged_reasoning(self, inputs_reasoning):
        outputs_reasoning = torch.zeros_like(inputs_reasoning)

        # every batch
        for batch in range(inputs_reasoning.size(0)):

            inputs_now = inputs_reasoning[batch, :, :].squeeze()  # 去掉第一维
            new_inputs = self._mru_contract_expand(inputs_now)

            # 合并后过两层神经网络
            for word in range(int(new_inputs.size(0) / len(self._r_mru_encoder))):  # 每个词
                word_pre = 0
                for r1 in range(len(self._r_mru_encoder)):  # 把一个词的r个表示串联
                    if isinstance(word_pre, int):
                        word_pre = new_inputs[(int(r1 * inputs_now.size(0)) + word), :].unsqueeze(0)
                    else:
                        word_pre = torch.cat((word_pre,
                                              new_inputs[(int(r1 * inputs_now.size(0)) + word), :].unsqueeze(0)), 0)

                word_pre = f.relu(self._w1_mru_encoder.mm(word_pre) + self._b1_mru_encoder)
                outputs_reasoning[batch, word, :] = f.relu(self._w2_mru_encoder.mm(word_pre) + self._b2_mru_encoder)

        return outputs_reasoning

    def _mru_encoding(self, gate_encoding, inputs_encoding):

        outputs_encoding = torch.zeros_like(inputs_encoding)

        for batch in range(inputs_encoding.size(0)):     # 每个batch
            temp_inputs_encoding = inputs_encoding[batch, :, :].squeeze()
            temp_gate_encoding = gate_encoding[batch, :, :].squeeze()
            c = 0          # cell
            for time in range(inputs_encoding.size(1)):          # 每个时间点
                z = f.tanh(self._wz_mru_encoder.mm(temp_inputs_encoding[time, :])) + self._bz_mru_encoder  # pre
                c = temp_gate_encoding[time, :] * c + (1 - temp_gate_encoding[time, :]) * z                # cell
                o = f.tanh(self._wo_mru_encoder.mm(temp_inputs_encoding[time, :])) + self._bo_mru_encoder  # output gate
                h = o * c           # hidden state + output
                outputs_encoding[batch, time, :] = h

        return outputs_encoding

    def _bi_attn(self):
        return 0


if __name__ == "__main__":

    # 列出Field
    QUESTIONS = data.Field(batch_first=True)
    OPTIONS = data.Field(batch_first=True)
    ARTICLES = data.Field(batch_first=True)
    ANSWERS = data.Field(sequential=False)

    # 数据集
    train = get_race(train_data=1)
    dev = get_race(train_data=0)
    device = torch.device("cuda")  # GPU

    # 建立词表
    QUESTIONS.build_vocab(train)
    OPTIONS.build_vocab(train)
    ARTICLES.build_vocab(train)
    ANSWERS.build_vocab(train)

    # 迭代器
    train_iter, dev_iter = data.BucketIterator.splits((train, dev), batch_size=BATCH_SIZE, shuffle=True,
                                                      repeat=False, device=device)

    net = MRU()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        running_loss = 0

        # 遍历iter
        for i, data_iter in enumerate(train_iter):

            # load to gpu
            answer = data_iter.answers
            option1 = data_iter.options1
            option2 = data_iter.options2
            option3 = data_iter.options3
            option4 = data_iter.options4
            question = data_iter.questions
            article = data_iter.articles

            optimizer.zero_grad()                                                  # zero grad
            output = net(option1, option2, option3, option4, question, article)    # forward
            loss = criterion(output, answer)                                       # loss
            loss.backward()                                                        # back
            optimizer.step()                                                       # optimize

            # print loss
            running_loss += loss.item()
            if i % 100 == 99:                                                      # print every 100 iter
                print("Epoch: %6d | ITER: %7d | LOSS: %.6f" % (epoch, i, running_loss))
                running_loss = 0.0

    # predict
    correct = 0
    total = 0
    with torch.no_grad():
        for data_iter in dev_iter:
            answer = data_iter.answers
            option1 = data_iter.options1
            option2 = data_iter.options2
            option3 = data_iter.options3
            option4 = data_iter.options4
            question = data_iter.questions
            article = data_iter.articles

            outputs = net(option1, option2, option3, option4, question, article)
            _, predicted = torch.max(outputs.data, 1)
            total += answer.size(0)
            correct += (predicted == answer).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))
