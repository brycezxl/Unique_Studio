import torch
import numpy as np
import torchvision.models as models
import torch.nn.functional as f
import torch.nn as nn
from torchvision import transforms
import matplotlib.image as mpimg
from torch.autograd import Variable
import matplotlib.pyplot as plt


ALPHA_BETA = 0.01
LR = 0.01
EPOCH = 1
WEIGHT = [1/5, 1/5, 1/5, 1/5, 1/5]
style_path = './picture/picasso.jpg'
content_path = './picture/blue-moon-lake.jpg'


class Picture(nn.Module):

    def __init__(self, height_in, width_in):
        self.picture = Variable(self.ini_pic(height_in, width_in), requires_grad=True)
        super(Picture, self).__init__()

        self.picture_conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)  # same global pool

        self.picture_conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.picture_conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.picture_conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.picture_conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.picture_conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self):
        picture_list = []

        picture_in = f.relu(self.picture_conv1_1(self.picture))
        picture_list.append(picture_in)
        picture_in = f.relu(self.picture_conv1_2(picture_in))
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_conv2_1(picture_in))
        picture_list.append(picture_in)
        picture_in = f.relu(self.picture_conv2_2(picture_in))
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_conv3_1(picture_in))
        picture_list.append(picture_in)
        picture_in = f.relu(self.picture_conv3_2(picture_in))
        picture_in = f.relu(self.picture_conv3_3(picture_in))
        picture_in = f.relu(self.picture_conv3_4(picture_in))
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_conv4_1(picture_in))
        picture_list.append(picture_in)
        picture_in = f.relu(self.picture_conv4_2(picture_in))
        pic_to_content = picture_in
        picture_in = f.relu(self.picture_conv4_3(picture_in))
        picture_in = f.relu(self.picture_conv4_4(picture_in))
        picture_in = self.pool(picture_in)
        picture_in = f.relu(self.picture_conv5_1(picture_in))
        picture_list.append(picture_in)
        picture_list.append(pic_to_content)

        return picture_list

    def ini_pic(self, height_inner, width_inner):
        pic_in = np.random.random(size=(height_inner, width_inner, 3))
        pic_in = pic_in.astype(np.float32)
        pic_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        pic_in = pic_transform(pic_in).view(1, 3, height_inner, width_inner).to(device)
        pic_in.requires_grad = True

        return pic_in


class Content(nn.Module):

    def __init__(self):
        super(Content, self).__init__()
        self.content_conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)  # same global pool

        self.content_conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.content_conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.content_conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.content_conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, content_in):
        content_in = f.relu(self.content_conv1_1(content_in))
        content_in = f.relu(self.content_conv1_2(content_in))
        content_in = self.pool(content_in)
        content_in = f.relu(self.content_conv2_1(content_in))
        content_in = f.relu(self.content_conv2_2(content_in))
        content_in = self.pool(content_in)
        content_in = f.relu(self.content_conv3_1(content_in))
        content_in = f.relu(self.content_conv3_2(content_in))
        content_in = f.relu(self.content_conv3_3(content_in))
        content_in = f.relu(self.content_conv3_4(content_in))
        content_in = self.pool(content_in)
        content_in = f.relu(self.content_conv4_1(content_in))
        content_in = f.relu(self.content_conv4_2(content_in))

        return content_in


class Style(nn.Module):

    def __init__(self):
        super(Style, self).__init__()
        self.style_conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)  # same global pool

        self.style_conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.style_conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.style_conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.style_conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.style_conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, style_in):
        style_list = []

        style_in = f.relu(self.style_conv1_1(style_in))
        style_in = f.relu(self.style_conv1_2(style_in))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_conv2_1(style_in))
        style_in = f.relu(self.style_conv2_2(style_in))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_conv3_1(style_in))
        style_in = f.relu(self.style_conv3_2(style_in))
        style_in = f.relu(self.style_conv3_3(style_in))
        style_in = f.relu(self.style_conv3_4(style_in))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_conv4_1(style_in))
        style_in = f.relu(self.style_conv4_2(style_in))
        style_in = f.relu(self.style_conv4_3(style_in))
        style_in = f.relu(self.style_conv4_4(style_in))
        style_list.append(style_in)
        style_in = self.pool(style_in)
        style_in = f.relu(self.style_conv5_1(style_in))
        style_list.append(style_in)

        return style_list


def get_module(module_in):

    vgg19 = models.vgg19(pretrained=True)  # 加载vgg
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


def get_content_style():
    content_module = get_module(Content()).to(device)
    torch.save(content_module.state_dict(), 'content_module.pth')
    style_module = get_module(Style()).to(device)
    torch.save(style_module.state_dict(), 'style_module.pth')

    content_in = content_module(content_pic)
    style_in = style_module(style_pic)

    return content_in, style_in


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

            g_a = 0

            for filter1 in range(style_now.size(1)):
                for filter2 in range(style_now.size(1)):
                    g = torch.sum(picture_now[:, filter1, :, :] * picture_now[:, filter2, :, :])
                    a = torch.sum(style_now[:, filter1, :, :] * style_now[:, filter2, :, :])
                    g_a += (g - a) ** 2

            e_list.append(g_a / style_now.size(1) ** 2 / (style_now.size(2) * style_now.size(3)) ** 2 / 4)

        style_loss = 0

        for layer in range(5):
            style_loss += e_list[layer] * self.weight[layer]

        prod = content_list[0] - picture_list[5]
        content_loss = 0.5 * torch.sum(prod * prod)

        loss_total = self.alpha * content_loss + self.beta * style_loss

        return loss_total


def get_pic(style_path_in, content_path_in):
    style_in = mpimg.imread(style_path_in)
    content_in = mpimg.imread(content_path_in)
    height_in, width_in, _ = content_in.shape

    height_in = int(height_in / 50)
    width_in = int(width_in / 50)

    style_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((height_in, width_in)),
            transforms.ToTensor()
        ]
    )

    content_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((height_in, width_in)),
            transforms.ToTensor()
        ]
    )

    style_in = style_transform(style_in).view(1, 3, height_in, width_in).to(device)
    content_in = content_transform(content_in).view(1, 3, height_in, width_in).to(device)

    return style_in, content_in, height_in, width_in


if __name__ == '__main__':

    print("Preparing for epochs...")

    device = torch.device("cuda")

    # get pictures
    content_pic, style_pic, height, width = get_pic(style_path, content_path)

    with torch.no_grad():

        # create modules
        picture_module = get_module(Picture(height, width)).to(device)
        torch.save(picture_module.state_dict(), 'picture_module.pth')
        content, style = get_content_style()

    # params process
    # BETA = torch.Tensor(1).to(device)
    # ALPHA = torch.Tensor(ALPHA_BETA)
    # weight = torch.Tensor(WEIGHT)

    criterion = Loss(ALPHA_BETA, 1, WEIGHT).to(device)
    # optimizer = torch.optim.LBFGS(params=picture_module.parameters())
    optimizer = torch.optim.Adam([picture_module.picture], lr=LR)
    running_loss = 0

    print("Starting epochs...")
    for epoch in range(EPOCH):

        # def closure():
        #     optimizer.zero_grad()
        #     outputs = picture_module()
        #     loss = criterion(style, outputs, content)
        #     loss.backward()
        #     global running_loss
        #     running_loss += loss.item()
        #     return loss
        #
        # optimizer.step(closure)

        optimizer.zero_grad()
        outputs = picture_module()
        loss = criterion(style, outputs, content)
        loss.backward()
        running_loss += loss.item()
        # print(picture_module.picture.grad)
        # picture_module.picture.data -= 100 * picture_module.picture.grad
        optimizer.step()

        if epoch % 1 == 0:  # print every 10 mini-batches
            print('EPOCH: %5d  Loss: %.6f' % (epoch + 1, running_loss / 1))
            running_loss = 0.0

    device = torch.device("cpu")
    pic = picture_module.picture.data.to(device).view(3, height, width)
    pic = pic.numpy()
    plt.imshow(np.transpose(pic, (1, 2, 0)))
    plt.savefig('fig_plt.png')
    plt.show()

