import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


# 1. 定义一个继承自 nn.Module 的类
class GalaxyResNet(nn.Module):
    def __init__(self, num_classes=37):
        super(GalaxyResNet, self).__init__()

        # 载入预训练好的经典 ResNet18 骨架
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # 获取原本全连接层（fc）的输入维度
        num_ftrs = self.backbone.fc.in_features

        # 将骨架中的全连接层替换为你自己设计的定制层
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 定义前向传播的数据流向：数据从 x 进来，流经被修改后的 backbone，最后输出
        return self.backbone(x)


if __name__ == "__main__":
    # 2. 实例化这个类（默认输出 37 维）
    model = GalaxyResNet(num_classes=37)

    # 模拟一张图片输入 [batch_size=1, channels=3, height=224, width=224]
    fake_input = torch.randn(1, 3, 224, 224)
    output = model(fake_input)
    print(f"模型前向传播测试成功！输出维度: {output.shape} (预期应该是 [1, 37])")