# PyTorch/ML 最佳实践

## 概述

本指南介绍 PyTorch 和机器学习开发的最佳实践，包括设备处理、模型模式、梯度控制、随机种子等。

---

## 设备处理

### 自动选择设备

```python
import torch

def get_device() -> torch.device:
    """自动选择最佳设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用设备
device = get_device()
print(f"Using device: {device}")

# 将模型和数据移动到设备
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

### 多 GPU 训练（可选）

```python
# 检查可用 GPU 数量
num_gpus = torch.cuda.device_count()
print(f"可用 GPU 数量: {num_gpus}")

if num_gpus > 1:
    model = nn.DataParallel(model)
    model = model.to(device)
```

---

## 模型模式

### train() 模式

```python
# 训练模式：启用 dropout、batchnorm 等
model.train()

for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(batch['inputs'])
    loss = criterion(outputs, batch['labels'])
    loss.backward()
    optimizer.step()
```

### eval() 模式

```python
# 评估模式：禁用 dropout、batchnorm 等
model.eval()

with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch['inputs'])
        predictions = outputs.argmax(dim=1)
        # 计算 accuracy 等
```

---

## 梯度控制

### 训练时：计算梯度

```python
with torch.set_grad_enabled(True):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### 推理时：禁用梯度计算（节省内存）

```python
with torch.no_grad():
    outputs = model(inputs)
    predictions = outputs.argmax(dim=1)
```

### 梯度裁剪（防止梯度爆炸）

```python
# 在 loss.backward() 之后，optimizer.step() 之前
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 随机种子（可复现性）

### 设置所有随机种子

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """设置所有随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保每次运行结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在实验开始时调用
set_seed(42)
```

### 在 DataLoader 中使用 worker_init_fn

```python
def worker_init_fn(worker_id):
    """为每个 DataLoader worker 设置随机种子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn,
)
```

---

## 模型保存与加载

### 保存模型

```python
# 保存完整模型（包括架构）
torch.save(model, 'model.pth')

# 仅保存模型参数（推荐）
torch.save(model.state_dict(), 'model_weights.pth')

# 保存多个组件（模型 + 优化器 + epoch 等）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

### 加载模型

```python
# 加载完整模型
model = torch.load('model.pth')

# 加载模型参数（推荐）
model = MyModelClass()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 加载多个组件
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## PyTorch 2.10.0 新特性

### 模型编译（推荐）

```python
# 使用 PyTorch 2.0+ 的编译器，提升性能 30-50%
model = torch.compile(model)

# 注意：首次运行会有编译开销，后续执行更快
```

### 默认设备设置

```python
# 设置默认设备，简化代码
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

# 之后创建的张量会自动在默认设备上
x = torch.randn(10, 10)  # 自动在 GPU 上（如果可用）
```

---

## 数据预处理

### 常用预处理操作

```python
from torchvision import transforms

# 图像数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
])

# 文本数据预处理
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
```

### 自定义 Dataset

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """自定义数据集类。"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

# 使用自定义数据集
dataset = CustomDataset(data, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 常用数据集加载示例

### HuggingFace Datasets

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('glue', 'mrpc')

# 训练集
train_data = dataset['train']

# 验证集
val_data = dataset['validation']

# 测试集
test_data = dataset['test']

# 转换为 PyTorch 格式
train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```

---

## 实验结果可视化建议

### 使用 TensorBoard

```bash
# 安装 TensorBoard
pip install tensorboard

# 启动 TensorBoard
tensorboard --logdir=outputs/logs
```

```python
from torch.utils.tensorboard import SummaryWriter

# 创建日志记录器
writer = SummaryWriter('outputs/logs/experiment_1')

# 记录标量（如损失、准确率）
for epoch in range(num_epochs):
    loss = train_one_epoch(model, dataloader, optimizer)
    accuracy = evaluate(model, val_dataloader)

    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

# 记录模型参数
writer.add_graph(model, sample_input)

writer.close()
```

### 使用 Matplotlib

```python
import matplotlib.pyplot as plt

# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('outputs/results/loss_curve.png')
```

---

## 常见反模式

### 错误 1: 忘记 model.eval()

```python
# 错误
predictions = model(inputs)

# 正确
model.eval()
with torch.no_grad():
    predictions = model(inputs)
```

### 错误 2: 忘记 zero_grad()

```python
# 错误
for batch in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 正确
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### 错误 3: 在 eval() 模式下计算梯度

```python
# 错误
model.eval()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()  # 错误！

# 正确
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

---

## 下一步

- [代码风格](code-style.md) - 命名约定、类型提示
- [调试技巧](debugging.md) - 常见问题与解决方案

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
