import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import torch.optim as optim
from torchtext import data
import glob
import re
__all__ = torch

BATCH_SIZE = 8
EPOCH = 3
LEARNING_RATE = 0.003          # 0.0003
DIM = 300
device = torch.device("cuda")  # GPU


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


# MRU #
class ContractExpand(nn.Module):
    def __init__(self):
        super(ContractExpand, self).__init__()
        # MRU Encoder
        self._r_mru_encoder = torch.tensor([1, 2, 4, 10, 25])
        self._f1_mru = nn.Linear(DIM, DIM)
        nn.init.xavier_normal_(self._f1_mru.weight)
        self._f2_mru = nn.Linear(DIM, DIM)
        nn.init.xavier_normal_(self._f2_mru.weight)
        self._f3_mru = nn.Linear(DIM, DIM)
        nn.init.xavier_normal_(self._f3_mru.weight)
        self._f4_mru = nn.Linear(DIM, DIM)
        nn.init.xavier_normal_(self._f4_mru.weight)
        self._f5_mru = nn.Linear(DIM, DIM)
        nn.init.xavier_normal_(self._f5_mru.weight)

    def forward(self, inputs_c_e):
        word_new = torch.zeros((5, inputs_c_e.size(0), inputs_c_e.size(1), inputs_c_e.size(2))).to(device)  # 新数组
        for r, func, n in zip(self._r_mru_encoder,
                              [self._f1_mru, self._f2_mru, self._f3_mru, self._f4_mru, self._f5_mru],
                              range(5)):
            r = r.to(device)                        # gpu
            groups = int(inputs_c_e.size(1) / r)    # 整除组
            rest = inputs_c_e.size(1) - r * groups  # 余数组

            # 整除组contract + expand
            for group in range(groups):
                sum_group1 = 0  # 求出r组词的和
                for p in range((r * group), (r * (group + 1))):
                    sum_group1 += inputs_c_e[:, p, :].unsqueeze(1)
                for p in range((r * group), (r * (group + 1))):
                    word_new[n, :, p, :] = f.relu(func(sum_group1.squeeze().to(device))) / r   # 对r组词的和操作一番

            # 余数组contract + expand
            if rest != 0:
                sum_group2 = 0
                for p in range((r * groups), (r * groups + rest)):
                    sum_group2 += inputs_c_e[:, p, :].unsqueeze(1)
                for p in range((r * groups), (r * groups + rest)):
                    word_new[n, :, p, :] = f.relu(func(sum_group2.to(device) * r / rest)).squeeze() / r
        return word_new


class MultiRangedReasoning(nn.Module):
    def __init__(self):
        super(MultiRangedReasoning, self).__init__()
        self._1_mru_contract_expand = nn.Linear(5, 3)
        nn.init.xavier_normal_(self._1_mru_contract_expand.weight)
        self._2_mru_contract_expand = nn.Linear(3, 1)
        nn.init.xavier_normal_(self._2_mru_contract_expand.weight)
        self._mru_contract_expand = ContractExpand().to(device)

    def forward(self, inputs_reasoning):

        outputs_reasoning = torch.zeros_like(inputs_reasoning).to(device)      # 初始化输出
        new_inputs = self._mru_contract_expand(inputs_reasoning).to(device)    # contract-expand后

        # 合并后过两层神经网络
        for word in range(new_inputs.size(2)):                        # 每个词
            word_pre = 0
            for r1 in range(5):                                                # 把一个词的r个表示串联
                if isinstance(word_pre, int):
                    word_pre = new_inputs[r1, :, word, :].unsqueeze(1).to(device)
                else:
                    word_pre = torch.cat((word_pre, new_inputs[r1, :, word, :].unsqueeze(1)), 1).to(device)
            for batch in range(inputs_reasoning.size(0)):      # 分batch把串联表示过2层fc，合并
                outputs_reasoning[batch, word, :] = f.relu(self._2_mru_contract_expand(
                    f.relu(self._1_mru_contract_expand(word_pre[batch, :, :].t())).t().t())).t()

        return outputs_reasoning.to(device)


class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()
        self._fz_mru = nn.Linear(DIM, DIM)
        nn.init.kaiming_normal_(self._fz_mru.weight)
        self._fo_mru = nn.Linear(DIM, DIM)
        nn.init.kaiming_normal_(self._fo_mru.weight)

    def forward(self, gate_encoding, inputs_encoding):
        outputs_encoding = torch.zeros_like(inputs_encoding).to(device)     # 初始化输出
        c = 0                                                               # 初始化cell
        for time in range(inputs_encoding.size(1)):                         # 每个时间点
            z = torch.tanh(self._fz_mru(inputs_encoding[:, time, :])).to(device)                    # input gate
            c = gate_encoding[:, time, :] * c + (1 - gate_encoding[:, time, :]).to(device) * z      # cell
            o = torch.tanh(self._fo_mru(inputs_encoding[:, time, :])).to(device)                    # output gate
            h = o * c           # hidden state + output
            outputs_encoding[:, time, :] = h.to(device)
        return outputs_encoding


class MRUEncoding(nn.Module):
    def __init__(self):
        super(MRUEncoding, self).__init__()
        self._mru_multi_ranged_reasoning = MultiRangedReasoning().to(device)
        self._mru_encoding = Encoding().to(device)

    def forward(self, inputs_mru):
        gate_mru = self._mru_multi_ranged_reasoning(inputs_mru).to(device)
        outputs_mru = self._mru_encoding(gate_mru, inputs_mru).to(device)
        return outputs_mru.to(device)
# End


# Attention
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, main_input, attn_input, f_input):

        main_on_attn = torch.zeros_like(main_input).to(device)

        for time_attn in range(main_input.size(1)):     # 每一个时间点（main input 的每个单词）
            attn_score = torch.zeros((attn_input.size(0), attn_input.size(1))).to(device)       # score

            for batch in range(main_input.size(0)):                # 必须循环每一个batch才能获得1×1的score
                for word_in_attn in range(attn_input.size(1)):     # 给每一个attn input的单词打分
                    attn_score[batch, word_in_attn] = (f_input(attn_input[batch, word_in_attn, :].unsqueeze(0))
                                                       .mm(main_input[batch, time_attn, :].unsqueeze(0).t()))
            attn_score = f.softmax(attn_score, dim=1).to(device)   # softmax

            for batch in range(main_input.size(0)):
                for word_in_attn in range(attn_input.size(1)):         # score * context
                    main_on_attn[batch, time_attn, :] += (attn_score[batch, word_in_attn] * attn_input[batch, word_in_attn, :])
        return main_on_attn


class BiAttention(nn.Module):
    def __init__(self):
        super(BiAttention, self).__init__()
        self._f1_bi_attn = nn.Linear(DIM, DIM)
        nn.init.kaiming_normal_(self._f1_bi_attn.weight)
        self._f2_bi_attn = nn.Linear(DIM, DIM)
        nn.init.kaiming_normal_(self._f2_bi_attn.weight)
        self._f3_bi_attn = nn.Linear(DIM, DIM)
        nn.init.kaiming_normal_(self._f3_bi_attn.weight)

        self._attn1 = Attention().to(device)
        self._attn2 = Attention().to(device)
        self._attn3 = Attention().to(device)

    def forward(self, input1, input2, mode):
        if mode == 0:
            # input1是article, input2是question，只算article on question
            article_on_question_attn = self._attn1(input1, input2, self._f1_bi_attn)
            return article_on_question_attn

        elif mode == 1:
            # input1是article, input2是option, bi-attn
            article_on_option_attn = self._attn2(input1, input2, self._f2_bi_attn)
            option_on_article_attn = self._attn3(input1, input2, self._f3_bi_attn)
            return article_on_option_attn, option_on_article_attn

        else:
            raise AttributeError("Wrong mode!")
# End


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, article_concat, options_concat):
        new_article_concat = torch.zeros((article_concat.size(0), DIM)).to(device)
        new_options_concat = torch.zeros((article_concat.size(0), DIM)).to(device)

        for word in range(article_concat.size(1)):      # 求每个文章word的和
            new_article_concat += article_concat[:, word, :]

        for word in range(options_concat.size(1)):      # 求每个选项word的和
            new_options_concat += options_concat[:, word, :]

        final_concat = torch.cat(((new_article_concat / article_concat.size(1)),
                                  (new_options_concat / options_concat.size(1))), 1).to(device)
        return final_concat.to(device)


class AnswerSelection(nn.Module):
    def __init__(self):
        super(AnswerSelection, self).__init__()
        self._f1_answer_selection = nn.Linear((2 * DIM), int(DIM / 4))
        self._f2_answer_selection = nn.Linear(int(DIM / 4), 1)

    def forward(self, answer_vector):
        answer_selected = torch.zeros((answer_vector.size(1), answer_vector.size(0))).to(device)
        for option in range(answer_vector.size(0)):
            answer_selected[:, option] = self._f2_answer_selection(
                f.relu(self._f1_answer_selection(answer_vector[option, :, :]))).squeeze()

        return answer_selected


class BiAttentionMRU(nn.Module):
    def __init__(self):
        super(BiAttentionMRU, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(ARTICLES.vocab), embedding_dim=DIM).to(device)
        self._mru = MRUEncoding().to(device)
        self._bi_attention = BiAttention().to(device)
        self._concat = Concat().to(device)
        self._answer_selection = AnswerSelection().to(device)

    def forward(self, option1_in, option2_in, option3_in, option4_in, question_in, article_in):

        # Input Encoding
        article_in = self.embedding(article_in).to(device)
        option1_in = self.embedding(option1_in).to(device)
        option2_in = self.embedding(option2_in).to(device)
        option3_in = self.embedding(option3_in).to(device)
        option4_in = self.embedding(option4_in).to(device)
        question_in = self.embedding(question_in).to(device)

        # MRU Encoding
        article_in = self._mru(article_in).to(device)

        # Bi-Attention Layer
        answer_pred = torch.zeros((4, BATCH_SIZE, (2 * DIM))).to(device)  # 最终的向量
        article_on_question = self._bi_attention(article_in, question_in, mode=0)
        for (option_in, num) in [(option1_in, 0), (option2_in, 1), (option3_in, 2), (option4_in, 3)]:
            article_on_question_option, option_on_article = self._bi_attention(article_on_question, option_in, mode=1)
            # Concat
            answer_pred[num, :, :] = self._concat(article_on_question_option, option_on_article)

        # Answer Selection
        answer_pred = self._answer_selection(answer_pred)

        return answer_pred


if __name__ == "__main__":

    # 列出Field
    QUESTIONS = data.Field(batch_first=True, fix_length=16)
    OPTIONS = data.Field(batch_first=True, fix_length=16)
    ARTICLES = data.Field(batch_first=True, pad_first=True)
    ANSWERS = data.Field(sequential=False)

    # 数据集
    train = get_race(train_data=1)
    dev = get_race(train_data=0)

    # 建立词表
    QUESTIONS.build_vocab(train)
    OPTIONS.build_vocab(train)
    ARTICLES.build_vocab(train)
    ANSWERS.build_vocab(train)

    # 迭代器
    train_iter, dev_iter = data.BucketIterator.splits((train, dev), batch_size=BATCH_SIZE, shuffle=True,
                                                      repeat=False, device=device, sort=True,
                                                      sort_key=lambda x: len(x.articles))
    net = BiAttentionMRU().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    j_list = []

    for epoch in range(EPOCH):
        running_loss = 0

        # 遍历iter
        for i, data_iter in enumerate(train_iter):

            # load
            answer = data_iter.answers - 1
            option1 = data_iter.options1
            option2 = data_iter.options2
            option3 = data_iter.options3
            option4 = data_iter.options4
            question = data_iter.questions
            article = data_iter.articles

            optimizer.zero_grad()                                                  # zero grad
            output = net(option1, option2, option3, option4, question, article)    # forward
            loss = criterion(output, answer)                                       # loss
            if not loss < 10:               # 控制梯度爆炸
                print("Abandoned epoch!")
                continue
            loss.backward()                                                        # back
            optimizer.step()                                                       # optimize

            # print loss
            running_loss += loss.item()
            # if i % 2 == 1:                                                      # print every 100 iter
            print("Epoch: %6d %4d%% | ITER: %7d | LOSS: %.6f" % (epoch, (epoch / EPOCH * 100), (i + 1), running_loss))
            j_list.append(running_loss)
            running_loss = 0.0

    plt.plot(range(len(j_list)), j_list, c="r")
    plt.show()

    # predict
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data_iter in enumerate(dev_iter):
            answer = data_iter.answers - 1
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
            if total > 100:
                break

    print('Accuracy: %d %%' % (100 * correct / total))
