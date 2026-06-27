import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class GalaxyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): 标签 CSV 文件的绝对路径
            img_dir (string): 存放所有图片的目录绝对路径
            transform (callable, optional): 可选的图像变换操作（数据增强）
        """
        # 确保传入的路径是绝对路径
        self.csv_file = os.path.abspath(csv_file)
        self.img_dir = os.path.abspath(img_dir)

        self.labels_frame = pd.read_csv(self.csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 获取图片名称并加载
        img_id = self.labels_frame.iloc[idx, 0]
        img_name = os.path.join(self.img_dir, f"{int(img_id)}.jpg")

        # 稳健性检查：防止某些图片损坏或丢失导致训练中断
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"找不到图片: {img_name}，请检查路径或数据完整性")

        image = Image.open(img_name).convert('RGB')

        # 2. 获取该图片对应的 37 个形态学的分类概率
        labels = self.labels_frame.iloc[idx, 1:].values
        labels = torch.tensor(labels.astype('float32'))

        # 3. 图像预处理/数据增强
        if self.transform:
            image = self.transform(image)

        return image, labels


# 提前定义好数据增强流程
train_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试代码是否正确
if __name__ == "__main__":
    # 动态获取项目根目录的绝对路径，从而精准定位 data 文件夹
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'training_solutions_rev1.csv')
    img_dir_path = os.path.join(base_dir, 'data', 'images_training_rev1')

    try:
        dataset = GalaxyDataset(csv_file=csv_path,
                                img_dir=img_dir_path,
                                transform=train_transform)
        print(f"成功加载数据集！数据条数: {len(dataset)}")
        img, label = dataset[0]
        print(f"单张图片 Tensor 形状: {img.shape}, 标签 Tensor 形状: {label.shape}")
    except Exception as e:
        print(f"代码还有小Bug，错误原因: {e}")