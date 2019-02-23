import re

"""
在腾讯提供的词向量里把食谱中的菜都选出来存到新的txt中
"""


def load_embedding(path):
    embedding_index = {}
    f1 = open(path, encoding='utf8')
    with open("food_sel.txt", "w+") as f2:
        for index1, line1 in enumerate(f1):
            if index1 == 0:
                continue
            values = line1.split(' ')
            word = values[0]
            if word in food_list:
                f2.write(line1)            # 找到了就写入新的txt
                print(line1)
                food_list.remove(word)
            if index1 % 10000 == 0:
                print(index1)
    f1.close()

    return embedding_index


food_list = []
with open(r"data/SplitAndIngreLabel/food.txt") as f:      # 加载中文菜名
    for index, line in enumerate(f):
        line = re.sub(r"\n", "", line)
        food_list.append(line)

load_embedding(r'Tencent_AILab_ChineseEmbedding.txt')
print(food_list)           # 没被选到的菜
