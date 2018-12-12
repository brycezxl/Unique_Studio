import torch
import torchvision.models as models
import torch.nn.functional as f
import torch.nn as nn


ALPHA_BETA = 0.01
EPOCH = 20
WEIGHT = [1, 1, 1, 1, 1]


class Picture(nn.Module):

    def __init__(self):
        self.picture = 1
        super(Picture, self).__init__()

        self.picture_conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.picture_conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.picture_conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.picture_conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.picture_conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.picture_conv5_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self):
        picture_list = []

        picture_in = f.relu(self.picture_bn1_1(self.picture_conv1_1(self.picture)))
        picture_in = f.relu(self.picture_bn1_2(self.picture_conv1_2(picture_in)))
        picture_list.append(picture_in)
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_bn2_1(self.picture_conv2_1(picture_in)))
        picture_in = f.relu(self.picture_bn2_2(self.picture_conv2_2(picture_in)))
        picture_list.append(picture_in)
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_bn3_1(self.picture_conv3_1(picture_in)))
        picture_in = f.relu(self.picture_bn3_2(self.picture_conv3_2(picture_in)))
        picture_in = f.relu(self.picture_bn3_3(self.picture_conv3_3(picture_in)))
        picture_in = f.relu(self.picture_bn3_4(self.picture_conv3_4(picture_in)))
        picture_list.append(picture_in)
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_bn4_1(self.picture_conv4_1(picture_in)))
        picture_in = f.relu(self.picture_bn4_2(self.picture_conv4_2(picture_in)))
        picture_in = f.relu(self.picture_bn4_3(self.picture_conv4_3(picture_in)))
        picture_in = f.relu(self.picture_bn4_4(self.picture_conv4_4(picture_in)))
        picture_list.append(picture_in)
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_bn5_1(self.picture_conv5_1(picture_in)))
        picture_in = f.relu(self.picture_bn5_2(self.picture_conv5_2(picture_in)))
        picture_in = f.relu(self.picture_bn5_3(self.picture_conv5_3(picture_in)))
        picture_in = f.relu(self.picture_bn5_4(self.picture_conv5_4(picture_in)))
        picture_list.append(picture_in)

        return picture_list


class Content(nn.Module):

    def __init__(self):
        super(Content, self).__init__()
        self.content_conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.content_conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.content_conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.content_conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.content_conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, content_in):
        content_in = f.relu(self.content_bn1_1(self.content_conv1_1(content_in)))
        content_in = f.relu(self.content_bn1_2(self.content_conv1_2(content_in)))
        content_in = self.pool(content_in)
        content_in = f.relu(self.content_bn2_1(self.content_conv2_1(content_in)))
        content_in = f.relu(self.content_bn2_2(self.content_conv2_2(content_in)))
        content_in = self.pool(content_in)
        content_in = f.relu(self.content_bn3_1(self.content_conv3_1(content_in)))
        content_in = f.relu(self.content_bn3_2(self.content_conv3_2(content_in)))
        content_in = f.relu(self.content_bn3_3(self.content_conv3_3(content_in)))
        content_in = f.relu(self.content_bn3_4(self.content_conv3_4(content_in)))
        content_in = self.pool(content_in)
        content_in = f.relu(self.content_bn4_1(self.content_conv4_1(content_in)))
        content_in = f.relu(self.content_bn4_2(self.content_conv4_2(content_in)))
        content_in = f.relu(self.content_bn4_3(self.content_conv4_3(content_in)))
        content_in = f.relu(self.content_bn4_4(self.content_conv4_4(content_in)))

        return content_in


class Style(nn.Module):

    def __init__(self):
        super(Style, self).__init__()
        self.style_conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)  # same global pool

        self.style_conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.style_conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.style_conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.style_conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.style_conv5_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, style_in):
        style_list = []

        style_in = f.relu(self.style_bn1_1(self.style_conv1_1(style_in)))
        style_in = f.relu(self.style_bn1_2(self.style_conv1_2(style_in)))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_bn2_1(self.style_conv2_1(style_in)))
        style_in = f.relu(self.style_bn2_2(self.style_conv2_2(style_in)))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_bn3_1(self.style_conv3_1(style_in)))
        style_in = f.relu(self.style_bn3_2(self.style_conv3_2(style_in)))
        style_in = f.relu(self.style_bn3_3(self.style_conv3_3(style_in)))
        style_in = f.relu(self.style_bn3_4(self.style_conv3_4(style_in)))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_bn4_1(self.style_conv4_1(style_in)))
        style_in = f.relu(self.style_bn4_2(self.style_conv4_2(style_in)))
        style_in = f.relu(self.style_bn4_3(self.style_conv4_3(style_in)))
        style_in = f.relu(self.style_bn4_4(self.style_conv4_4(style_in)))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_bn5_1(self.style_conv5_1(style_in)))
        style_in = f.relu(self.style_bn5_2(self.style_conv5_2(style_in)))
        style_in = f.relu(self.style_bn5_3(self.style_conv5_3(style_in)))
        style_in = f.relu(self.style_bn5_4(self.style_conv5_4(style_in)))
        style_list.append(style_in)

        return style_list


def get_module(module_in):

    vgg19 = models.vgg19_bn(pretrained=True)  # 加载vgg
    key_list = []
    dic = {}

    # 加载所有层的名字
    for key in module_in.state_dict().keys():
        key_list.append(key)

    count = 0
    for _, v in vgg19.state_dict().items():
        dic[key_list[count]] = v
        count += 1
        if count == len(key_list):
            break

    # 载入参数
    module_in.load_state_dict(dic)

    return module_in


class Loss(nn.Module):
    def __init__(self, alpha_in, beta_in, weight_in):
        super(Loss, self).__init__()
        self.alpha = alpha_in
        self.beta = beta_in
        self.weight = weight_in

    def forward(self, style_list, picture_list, content_list):

        e_list = []

        for i in range(5):
            style_now = style_list[i]
            picture_now = picture_list[i]

            g = 0
            a = 0

            for filter1 in range(style_now.size(0)):
                for filter2 in range(style_now.size(0)):
                    g += torch.sum(picture_now[filter1, :, :] * picture_now[filter2, :, :])
                    a += torch.sum(style_now[filter1, :, :] * style_now[filter2, :, :])

            e_list.append((g - a) ** 2)

        style_loss = 0

        for layer in range(5):
            style_loss += e_list[layer] * self.weight(layer)

        prod = content_list[0] - picture_list[3]
        content_loss = 0.5 * torch.sum(prod * prod)

        loss_total = self.alpha * content_loss + self.beta * style_loss
        return loss_total


device = torch.device("cuda")

content_pic = 1.to(device)
style_pic = 2.to(device)

content_module = get_module(Content()).to(device)
style_module = get_module(Style()).to(device)
picture_module = get_module(Picture()).to(device)

content = content_module(content_pic)
style = style_module(style_pic)

BETA = torch.Tensor(1).to(device)
ALPHA = torch.Tensor(ALPHA_BETA).to(device)
weight = torch.Tensor(WEIGHT).to(device)

criterion = Loss(ALPHA, BETA, weight).to(device)
optimizer = torch.optim.LBFGS(params=picture_module.parameters())

running_loss = 0

for epoch in range(EPOCH):
    optimizer.zero_grad()

    outputs = picture_module()
    loss = criterion(style, outputs, content)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if epoch % 10 == 9:  # print every 10 mini-batches
        print('EPOCH: %5d  Loss: %.6f' %
              (epoch + 1, running_loss / 10))
        running_loss = 0.0

