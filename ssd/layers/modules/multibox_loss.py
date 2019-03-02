import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
# from data import cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """
    计算目标:
    1.输出那些与真实框的iou大于一定阈值的框的下标
    2.根据与真实框的偏移量输出localization目标,计算loc损失
    3.用难样例挖掘算法去除大量负样本(默认正负样本比例为1:3),计算包括正样本和负样本的置信度损失
    :param:
        L(x,c,l,g) = [L_conf(x,c), L_loc(x,l,g)] / N
        x: {1,0}
        c: class confidences
        l: predicted boxes
        g: ground truth boxes
        N: number of matched boxes
    """
    def __init__(self, class_num):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = class_num  # 列表数
        self.threshold = 0.5          # 交并比阈值, 0.5
        self.background_label = 0     # 背景标签, 0
        self.neg_pos_ratio = 3         # 负样本和正样本的比例, 3:1
        self.neg_overlap = 0.5        # 0.5 判定负样本的阈值.
        self.variance = [0.1, 0.2]

    def forward(self, predictions, targets):
        """
        Multibox Loss
            Args:
                predictions (tuple): A tuple containing loc preds, conf preds, and prior boxes from SSD net.
                    conf shape:   (batch_size, num_priors, num_classes)
                    loc shape:    (batch_size, num_priors, 4)
                    priors shape: (num_priors, 4)
                targets (tensor): Ground truth boxes and labels for a batch [batch_size,num_objs,5]
                (last idx is the label).
        """
        loc_data, conf_input, priors = predictions
        num = loc_data.size(0)                 # num = batch_size
        num_priors = (priors.size(0))          # num_priors = 8732

        # 1.
        # 将priors(default boxes)与ground truth匹配
        loc_t = torch.Tensor(num, num_priors, 4)  # [batch_size, 8732, 4]
        conf_t = torch.Tensor(num, num_priors)    # [batch_size, 8732]
        for idx in range(num):
            truths = targets[idx, :, :-1].data  # [num_objs, 4]
            labels = targets[idx, :, -1].data   # [num_objs]
            defaults = priors.data              # [8732, 4]
            # 关键函数, 实现候选框与真实框之间的匹配, 注意是候选框而不是预测结果框! 这个函数实现较为复杂, 会在后面着重讲解
            # 注意! 要清楚 Python 中的参数传递机制, 此处在函数内部会改变 loc_t, conf_t 的值
            # 关于 match 的详细讲解可以看后面的代码解析
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)  # 求和, 取得满足条件的box的数量, [batch_size, num_gt_threshold]

        # 2.
        # 位置(localization)损失函数, 使用 Smooth L1 函数求损失
        # loc_data: [batch, num_priors, 4]
        # pos:      [batch, num_priors]
        # pos_idx:  [batch, num_priors, 4], 复制下标成坐标格式, 以便获取坐标值
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # pos值下标
        loc_p = loc_data[pos_idx].view(-1, 4)  # 预测框偏移量
        loc_t = loc_t[pos_idx].view(-1, 4)     # default偏移量
        loss_l = f.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # 3.1
        # 计算最大的置信度, 以进行难负样本挖掘
        # conf_input: [batch, num_priors, num_classes]
        conf_batch = conf_input.view(-1, self.num_classes)  # reshape, combine all batches
        # batch_conf: [batch × num_priors, num_classes]
        # conf_t: [batch, num_priors]
        # loss_c: [batch*num_priors, 1], 计算每个prior box预测后的损失
        loss_c = log_sum_exp(conf_batch) - conf_batch.gather(1, conf_t.view(-1, 1))
        # 进行降序排序, 并获取到排序的下标        代表从大到小各是原来的第几位
        _, loss_idx = loss_c.sort(1, descending=True)
        # 将下标进行升序排序, 并获取到下标的下标   相当于反向降序排序 序号递增 每个序号对应刚刚的排名 代表第几名回哪里去
        _, idx_rank = loss_idx.sort(1)
        # 根据pos的个数, 确定负样本的个数(正样本的3倍)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        # 获取到负样本的下标 形式为【0，1,0,1,1,0】
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 3.2
        # 计算confidence loss
        # 计算包括正样本和负样本的置信度损失
        # conf_input: [batch, num_priors, num_classes]
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_input)
        # neg: [batch, num_priors]
        # neg_idx: [batch, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_input)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
        conf_p = conf_input[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的加权标签
        # [batch_size, num_priors]
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # 计算二者的交叉熵
        loss_c = f.cross_entropy(conf_p, targets_weighted, size_average=False)

        n = num_pos.data.sum()
        loss_l /= n
        loss_c /= n
        return loss_l, loss_c
