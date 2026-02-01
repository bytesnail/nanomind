# 调试技巧与常见问题

## 概述

本指南介绍深度学习开发中的常见问题和调试技巧，帮助快速定位和解决问题。

---

## GPU 相关问题

### 检查 CUDA 是否可用

```python
# 检查 CUDA 是否可用
print(torch.cuda.is_available())

# 查看 GPU 信息
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())
```

### GPU 内存管理

```python
# 查看已分配的 GPU 内存
print(torch.cuda.memory_allocated())

# 查看已保留的 GPU 内存
print(torch.cuda.memory_reserved())

# 清空 GPU 缓存
torch.cuda.empty_cache()
```

### GPU 不可用的解决方案

```bash
# 1. 检查 CUDA 版本匹配
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. 确保 conda 中的 CUDA 版本匹配
conda install -c nvidia cuda=12.8 -y

# 3. 重新安装 PyTorch
uv pip install -r requirements.txt
```

---

## 内存优化

### 减少内存占用

```python
# 1. 使用混合精度训练（需要安装 apex）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. 适当减小 batch_size
batch_size = 16  # 从 32 减小到 16

# 3. 在训练循环中定期清理缓存
for epoch in range(num_epochs):
    for batch in dataloader:
        # 训练代码
        pass

    # 每个 epoch 后清理缓存
    torch.cuda.empty_cache()
```

### 优化数据加载

```python
# 使用 num_workers 加速数据加载
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 使用 4 个 worker
    pin_memory=True,  # 加速 CPU 到 GPU 的数据传输
)
```

---

## 性能优化

### 使用 DataLoader 的 num_workers

```python
# 使用多个 worker 并行加载数据
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 根据CPU核心数调整
    pin_memory=True,  # 加速 CPU 到 GPU 的数据传输
)
```

### 使用 pin_memory

```python
# 启用 pin_memory 加速数据传输
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # 将数据锁定在内存中，加速 GPU 传输
)

# 将数据移动到 GPU（非阻塞）
inputs = inputs.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

### 避免在训练循环中进行不必要的 CPU 操作

```python
# 错误：在循环中进行不必要的计算
for batch in dataloader:
    inputs = batch['inputs']
    labels = batch['labels']

    # 不必要的操作
    dataset_size = len(inputs)  # 每次都重新计算

    outputs = model(inputs)
    loss = criterion(outputs, labels)

# 正确：移到循环外部
dataset_size = len(dataloader.dataset)  # 计算一次

for batch in dataloader:
    inputs = batch['inputs']
    labels = batch['labels']

    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### 使用 torch.jit.script 或 torch.compile

```python
# 方法 1: 使用 torch.jit.script（旧版本）
model = torch.jit.script(model)

# 方法 2: 使用 torch.compile（PyTorch 2.0+，推荐）
model = torch.compile(model)

# 注意：首次运行会有编译开销，后续执行更快
```

---

## 梯度问题

### 检测梯度问题

```python
# 启用梯度异常检测
torch.autograd.set_detect_anomaly(True)

# 如果出现 NaN 或 Inf，会抛出异常并显示详细信息
```

### 梯度爆炸或消失

```python
# 梯度裁剪（防止梯度爆炸）
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# 梯度监控（检查梯度消失）
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean()}, grad std={param.grad.std()}")
```

---

## 调试技巧

### 使用 print 或 logging 记录训练进度

```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 训练循环中记录
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 每 100 个 batch 打印一次
        if batch_idx % 100 == 0:
            logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
```

### 使用 pdb 进行调试

```python
import pdb

# 在代码中设置断点
def train_model(model, dataloader, optimizer):
    for batch in dataloader:
        pdb.set_trace()  # 设置断点

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
```

### 检查张量形状

```python
# 打印张量形状（调试非常有用）
print(f"inputs shape: {inputs.shape}")
print(f"outputs shape: {outputs.shape}")
print(f"labels shape: {labels.shape}")
```

---

## 常见错误及解决方案

### 错误 1: CUDA out of memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 1. 清空 GPU 缓存
torch.cuda.empty_cache()

# 2. 减小 batch_size
batch_size = 16  # 从 32 减小到 16

# 3. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 错误 2: Expected input batch_size (X) to match target batch_size (Y)

**症状**: 输入和标签的 batch_size 不匹配

**解决方案**:
```python
# 检查 batch_size
print(f"inputs batch_size: {inputs.shape[0]}")
print(f"labels batch_size: {labels.shape[0]}")

# 确保 inputs 和 labels 的 batch_size 相同
inputs = inputs[:16]  # 截断到相同的 batch_size
labels = labels[:16]
```

### 错误 3: RuntimeError: Expected object of scalar type Float but got scalar type Long

**症状**: 数据类型不匹配

**解决方案**:
```python
# 检查数据类型
print(f"inputs dtype: {inputs.dtype}")
print(f"labels dtype: {labels.dtype}")

# 转换数据类型
inputs = inputs.float()  # 转换为 float32
labels = labels.long()   # 转换为 int64
```

### 错误 4: RuntimeError: element 0 of tensors does not require grad

**症状**: 张量不需要梯度

**解决方案**:
```python
# 确保模型参数需要梯度
for param in model.parameters():
    param.requires_grad = True

# 确保输入需要梯度（如果需要）
inputs.requires_grad = True
```

---

## TensorBoard 使用指南

### 安装和启动

```bash
# 安装 TensorBoard
uv add tensorboard --no-sync
uv pip install -r requirements.txt

# 启动 TensorBoard
tensorboard --logdir=outputs/logs

# 在浏览器中访问 http://localhost:6006
```

### 记录训练过程

```python
from torch.utils.tensorboard import SummaryWriter

# 创建日志记录器
writer = SummaryWriter('outputs/logs/experiment_1')

# 记录标量（如损失、准确率）
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_dataloader, optimizer)
    val_accuracy = evaluate(model, val_dataloader)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

# 记录学习率
for epoch in range(num_epochs):
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

# 记录模型参数
for name, param in model.named_parameters():
    writer.add_histogram(f'Parameters/{name}', param, epoch)
    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

# 记录模型图
writer.add_graph(model, sample_input)

writer.close()
```

---

## 性能分析

### 使用 PyTorch Profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=profiler.tensorboard_trace_handler('./outputs/logs/profiler'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prof.step()

# 查看 TensorBoard 中的性能分析结果
# tensorboard --logdir=outputs/logs/profiler
```

---

## 下一步

- [代码风格](code-style.md) - 命名约定、类型提示
- [最佳实践](best-practices.md) - PyTorch 开发规范

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 开发指南导航
- [README.md](../../README.md) - 项目简介和快速开始
