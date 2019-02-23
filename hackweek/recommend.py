def recommend(food_list):
    """
    Recommend dished according to a user's past diet.
    :param food_list: a list that contains the dished that a user have had recently
    :return: a list contains 4 recommended food
    """
    def _get_dic(list_in):
        """
        Turn the list in to a dict, eg: {dish_name:weight, ...}
        :param list_in: a list that contains the dished that a user have had recently
        :return: dic_get: dict with food name and its weight
                 w_sum: the sum of all weights
        """
        dic_get = {}
        w_sum = 0  # record the total weight
        for food in list_in:  # to dict
            if food in model.index2word:  # in pre-trained food list
                if food in dic_get:
                    dic_get[food] += 1
                    w_sum += 1
                else:
                    dic_get[food] = 3
                    w_sum += 3
        return dic_get, w_sum

    def _get_favor_dish(dict_in, sum_in):
        """
        Select 4 favored dishes according to weight.
        :param dict_in: dict with food name and its weight
        :param sum_in: sum of weights
        :return: random_list_in: a list of selected dishes
        """
        # get 4 random number
        rand1 = random.randint(1, sum_in)
        rand2 = random.randint(1, sum_in)
        rand3 = random.randint(1, sum_in)
        rand4 = random.randint(1, sum_in)

        # get random dish based on weight
        random_sum = 0
        flag = 0
        random_list_in = []
        for key, value in dict_in.items():
            random_sum += value
            if random_sum >= rand1:
                random_list_in.append(key)
                flag += 1
                rand1 = sum_in + 1  # dead
            if random_sum >= rand2:
                random_list_in.append(key)
                flag += 1
                rand2 = sum_in + 1
            if random_sum >= rand3:
                random_list_in.append(key)
                flag += 1
                rand3 = sum_in + 1
            if random_sum >= rand4:
                random_list_in.append(key)
                flag += 1
                rand4 = sum_in + 1
            if flag == 4:
                break
        return random_list_in

    def _get_recommend(list_in):
        """
        Recommend dished according to favored dish'es most similar dish.
        :param list_in: selected 4 favored dishes
        :return: recommend_list_in: recommend dished
        """
        recommend_list_in = []
        for dish in list_in:
            up = 2
            random_num = random.randint(0, up)
            item_in, _ = model.most_similar(dish)[random_num]
            while item_in in recommend_list_in:
                random_num = random.randint(0, up + 1)
                item_in, _ = model.most_similar(dish)[random_num]
            else:
                recommend_list_in.append(item_in)
        return recommend_list_in

    if not isinstance(food_list, list):
        raise AttributeError("Input a list!")
    from gensim import models
    import random
    model = models.KeyedVectors.load_word2vec_format(r"food_vec.txt", binary=False)   # load pre-trained word vectors
    food_dic, weight_sum = _get_dic(food_list)            # turn to dict
    random_list = _get_favor_dish(food_dic, weight_sum)   # random from favored
    recommend_list = _get_recommend(random_list)          # recommend
    return recommend_list
