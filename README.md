# LUNA16 肺部CT图像分割项目

基于UNet的肺部CT图像分割项目，使用LUNA16数据集进行训练。该项目实现了从数据预处理到模型训练的完整流程。

## 项目结构

```
.
├── data/                   # 原始数据目录
│   ├── subset0-9/         # LUNA16数据集子集
│   └── seg-lungs-LUNA16/  # 肺部分割标注数据
├── LUNA16/                # 预处理后的数据
│   ├── train/            # 训练集
│   └── test/             # 测试集
├── src/                   # 源代码
│   └── UNet.py           # UNet模型定义
├── getLUNA16.py          # 数据预处理脚本
├── train.py              # 训练脚本
├── plot_training.py      # 训练过程可视化脚本
└── requirements.txt      # 项目依赖
```

## 环境要求

- Python 3.10+
- CUDA支持（推荐用于GPU训练）
- 其他依赖见`requirements.txt`

## 安装步骤

1. 克隆项目：
```bash
git clone https://github.com/yourusername/luna16-segmentation.git
cd luna16-segmentation
```

2. 创建虚拟环境并安装依赖：
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## 数据预处理

1. 将LUNA16数据集放在`data`目录下
2. 运行预处理脚本：
```bash
python getLUNA16.py
```

这将：
- 读取.mhd格式的CT图像
- 提取中间切片
- 进行归一化和二值化处理
- 按8:2比例分割训练集和测试集
- 保存为PNG格式

## 模型训练

1. 开始训练：
```bash
python train.py --data-path ./LUNA16 --batch-size 4 --epochs 10
```

主要参数说明：
- `--data-path`: 数据路径
- `--batch-size`: 批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率（默认0.01）
- `--device`: 训练设备（默认"cuda"）
- `--save-best`: 是否只保存最佳模型（默认True）

2. 查看训练过程：
```bash
python plot_training.py
```
这将生成`training_curves.png`，包含：
- 训练损失曲线
- 学习率变化曲线
- Dice系数变化曲线

## 模型保存

训练过程中会保存：
- 最佳模型：`save_weights/best_model.pth`
- 训练日志：`results{时间戳}.txt`

## 项目特点

- 完整的数据预处理流程
- 基于UNet的语义分割模型
- 支持混合精度训练
- 自动学习率调度
- 训练过程可视化
- 模型检查点保存

## 引用

如果您使用了本项目，请引用：
```
@misc{luna16-segmentation,
  author = {Your Name},
  title = {LUNA16 Lung CT Segmentation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/luna16-segmentation}
}
```

## 许可证

MIT License 