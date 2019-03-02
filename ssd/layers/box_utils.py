import torch


def log_sum_exp(x):
    """
    Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def point_form(boxes):
    """
    将 (cx, cy, w, h) 形式的prior box坐标转换成 (x_min, y_min, x_max, y_max)
    即左下角与右上角的形式, 来与ground truth比较
    (cx, cy) - (w, h)/2 得到x_min, y_min
    (cx, cy) + (w, h)/2 得到x_max, y_max
    :param boxes: prior boxes of size(num, 4), 4 represents cx,cy,w,h
    :return boxes: boxes of size(num, 4), 4 represents x_min,y_min,x_max,y_max
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,      # x_min, y_min
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # x_max, y_max


def center_size(boxes):
    """
    和point form相反
    Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    :param  boxes: (tensor) point_form boxes
    :return:
        boxes: (tensor) Converted x_min, y_min, x_max, y_max form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)     # w, h


def intersect(box_a, box_b):
    """
    返回 box_a 与 box_b 集合中元素的交集
    We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.

    1.先将两个box的维度扩展至相同维度: [num_obj, num_priors, 4], 然后计算面积的交集
    2.两个box的交集可以看成是一个新的box, 该box的左下角坐标是box_a和box_b左下角坐标的较大值,
    右上角坐标是box_a和box_b的右上角坐标的较小值

    Args:
        box_a: (truth) bounding boxes, Shape: [num_obj, 4]
        box_b: (prior) bounding boxes, Shape: [num_priors, 4]
    Return:
    """
    a = box_a.size(0)
    b = box_b.size(0)

    # unsqueeze 为增加维度的位置, expand 为扩展维度的大小
    # 求左上角(max_xy)的较大者(max)
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(a, b, 2),
                       box_b[:, :2].unsqueeze(0).expand(a, b, 2))
    # 求右上角(max_xy)的较小者(min)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(a, b, 2),
                       box_b[:, 2:].unsqueeze(0).expand(a, b, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)  # 右上角减去左下角, 如果为负值, 说明没有交集, 置为0
    return inter[:, :, 0] * inter[:, :, 1]  # 高×宽, 返回交集的面积, shape 刚好为 [A, B]


def jaccard(box_a, box_b):
    """
    返回 box_a 与 box_b 集合中元素的交并比, 即jaccard overlap
    Here we operate on ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)     # 交集面积
    # shape [A,B], 这里会将A/B中的元素复制B/A次
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def encode(matched, priors, variances):
    """
    将 box 边框坐标编码成小数形式, 方便网络训练, 需要宽度方差和高度方差两个参数
    ground truth 对 default box 的转换
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
                shape: [num_priors, 4]
        priors: (tensor) Prior boxes in center-offset form
                shape: [num_priors,4]
        variances: (list[float]) Variances of prior boxes
    Returns:
        Encoded boxes (tensor), Shape: [num_priors, 4]

    """
    return 0


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    pass
