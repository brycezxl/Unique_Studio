import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np


ALPHA_BETA_LIST = [8 * 10 ** -3, 1 * 10 ** -2, 3 * 10 ** -2]
EPOCH = 200000
MAX_SIZE = 1200
style_path = './picture/starry.jpg'
content_path = './picture/house.jpg'


def load_image(image_path, max_size=None, shape=None):
    image = Image.open(image_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ]
    )

    if max_size is not None:
        # 获取图像size
        image_width, image_height = image.size

        # 转化为float的array
        image_width = np.array(image_width).astype(float)
        image_height = np.array(image_height).astype(float)

        # resize
        image_height = (max_size / image_width * image_height).astype(int)
        image_width = (max_size * image_width / image_width).astype(int)
        image = image.resize((image_width, image_height), Image.ANTIALIAS)

    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)

    image = transform(image).unsqueeze(0).to(device)

    return image


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg19 = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []

        # name类型为str，x为Variable
        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


if __name__ == '__main__':
    count = 0
    for ALPHA_BETA in ALPHA_BETA_LIST:
        count += 1

        device = torch.device("cuda")

        # 加载图片
        content = load_image(content_path, max_size=MAX_SIZE)
        style = load_image(style_path, shape=[content.size(2), content.size(3)])

        vgg = VGGNet().to(device)

        # 将concent复制一份作为target，并需要计算梯度，作为最终的输出
        target = Variable(content.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([target])

        with torch.no_grad():
            content_features = vgg(Variable(content))[3]
            style_features = vgg(Variable(style))

        for epoch in range(EPOCH):

            content_loss = 0.0
            style_loss = 0.0

            target_features = vgg(target)

            for f1, f2 in zip(target_features, style_features):

                _, d, h, w = f1.size()

                # 将特征reshape成二维矩阵相乘，求gram矩阵
                f1 = f1.view(d, h * w)
                f2 = f2.view(d, h * w)

                f1 = torch.mm(f1, f1.t())
                f2 = torch.mm(f2, f2.t())

                # 计算style_loss
                style_loss += torch.mean((f1 - f2) ** 2) / (d * h * w) / 5

            # 计算content_loss
            content_loss += torch.mean((content_features - target_features[3]) ** 2)

            # 计算总的loss
            loss = content_loss * ALPHA_BETA + style_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if (epoch + 1) % 500 == 0:
                print('Step: %4d / %4d  |  Content Loss:  %.4f  |  Style Loss:  %.4f'
                      % (epoch + 1, EPOCH, content_loss, style_loss))

        # Save the generated image
        img = target.clone().cpu().squeeze()
        torchvision.utils.save_image(img, '%d-output.png' % count)
