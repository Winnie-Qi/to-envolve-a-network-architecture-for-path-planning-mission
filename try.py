import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomConvNet(nn.Module):
    def __init__(self):
        super(CustomConvNet, self).__init__()
        # 第一个卷积层：5个节点
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        # 第二个卷积层：8个节点
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3)

    def forward(self, x):
        # 第一层节点1只对输入的第一层卷积
        conv1_output1 = F.conv2d(x[0:1, :, :, :], self.conv1.weight[0:1, :, :, :], self.conv1.bias[0:1],
                                 self.conv1.stride, self.conv1.padding)
        # 第一层节点3只对输入的第二层卷积
        conv1_output3 = F.conv2d(x[1:2, :, :, :], self.conv1.weight[2:3, :, :, :], self.conv1.bias[2:3],
                                 self.conv1.stride, self.conv1.padding)
        # 第一层节点5只对输入的第三层卷积
        conv1_output5 = F.conv2d(x[2:3, :, :, :], self.conv1.weight[4:5, :, :, :], self.conv1.bias[4:5],
                                 self.conv1.stride, self.conv1.padding)

        # 其余节点常规卷积
        conv1_output_rest = F.conv2d(x, self.conv1.weight[1:4, :, :, :], self.conv1.bias[1:4], self.conv1.stride,
                                     self.conv1.padding)

        # 检查各个张量的大小
        print(conv1_output1.size())
        print(conv1_output_rest.size())
        print(conv1_output3.size())
        print(conv1_output_rest.size())
        print(conv1_output5.size())

        # 合并第一层节点的输出
        conv1_output = torch.cat([conv1_output1, conv1_output_rest, conv1_output3, conv1_output_rest, conv1_output5],
                                 dim=1)

        # 第二层常规卷积
        conv2_output = F.conv2d(conv1_output, self.conv2.weight, self.conv2.bias, self.conv2.stride, self.conv2.padding)

        return conv2_output


# 创建模型
model = CustomConvNet()

# 示例用法
input_tensor = torch.randn(1, 3, 11, 11)
output_tensor = model(input_tensor)
print(output_tensor.shape)