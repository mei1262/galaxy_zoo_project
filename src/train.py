import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# 导入你之前写好的模块
from dataset import GalaxyDataset, train_transform
from model import GalaxyResNet  # 假设你在 model.py 里定义了这个名字


def train_model():
    # 1. 硬件加速检测（优先使用显卡）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前正在使用设备: {device}")

    # 2. 动态获取绝对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'training_solutions_rev1.csv')
    img_dir_path = os.path.join(base_dir, 'data', 'images_training_rev1')

    # 3. 加载完整数据集并切分出 20% 作为本地验证集
    full_dataset = GalaxyDataset(csv_file=csv_path, img_dir=img_dir_path, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. 创建 DataLoader（批处理器）
    # 如果你在本地电脑跑，建议 batch_size 先设为 32 或 64；num_workers 可以设为 2 或 4 加速读取
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 5. 初始化模型并送入 GPU/CPU
    model = GalaxyResNet(num_classes=37).to(device)

    # 6. 定义损失函数和优化器
    criterion = nn.MSELoss()  # 预测概率偏离度，天文学任务常用 MSE 或 BCE
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 使用 Adam 优化器，学习率设小一点保护预训练权重

    # 7. 开始训练循环（Epoch）
    num_epochs = 10  # 暑期项目建议先跑 10 个 Epoch 看看效果
    print("开始训练...")

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 每个 Epoch 结束时跑一下验证集，检查有没有过拟合
        model.eval()  # 切换到评估模式
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"=== Epoch {epoch + 1} 完成! 训练集平均 Loss: {running_loss / len(train_loader):.4f}, 验证集平均 Loss: {val_loss / len(val_loader):.4f} ===")

    # 8. 训练结束，保存你的成果（模型权重）
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'galaxy_resnet_v1.pth')
    torch.save(model.state_dict(), save_path)
    print(f"成果已成功保存至: {save_path}")


if __name__ == "__main__":
    train_model()