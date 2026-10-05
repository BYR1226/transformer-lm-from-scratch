# Transformer LM from Scratch

这是我完成 Stanford CS336 Assignment 1（Basics）的实现记录。项目从基础组件开始，逐步完成 tokenizer、Transformer、优化器、数据处理、checkpoint 和训练流程，最后在 TinyStories 数据集上完成了实际训练和文本生成。

耗时断断续续大概1个月，完成了从单元测试、内存问题、性能优化到云 GPU 训练的完整过程。

## 项目内容

主要实现包括：

- GPT 风格 BPE tokenizer
- Unicode 预分词和特殊 token 处理
- Embedding、Linear、RMSNorm、SwiGLU
- RoPE 旋转位置编码
- 缩放点积注意力和因果多头注意力
- Pre-norm Transformer block
- Transformer language model
- Cross-entropy loss
- AdamW 优化器
- Warmup + Cosine 学习率调度
- 梯度裁剪
- checkpoint 保存和恢复
- 基于 `np.memmap` 的大规模数据加载
- TinyStories 训练和文本生成

## 实现过程

### 1. 先完成基础模块和测试

按照官方作业文件接口逐个实现基础运算和 Transformer 组件，并使用课程提供的 pytest 进行验证。

测试过程中重点检查了：

- 张量形状是否正确；
- batch 维度是否可以扩展；
- attention mask 是否保持因果关系；
- RoPE 是否支持不同 token position；
- checkpoint 保存后能否恢复模型和优化器状态。

课程提供的 pytest 测试全部通过。

### 2. tokenizer 的内存问题

最初训练 TinyStories tokenizer 时，直接读取完整的 2GB 文本文件，WSL 进程因为内存不足被系统强制终止。

为了解决WSL运行内存不足的问题，我将 tokenizer 改成流式处理：

- 不再使用 `read()` 一次性读取整个文件；
- 按块读取训练文本；
- 使用特殊 token 分割文档；
- 只保存预分词结果和统计信息；
- 使用反向索引跟踪受某次 merge 影响的 token。

优化后的 BPE 训练可以在有限内存下（峰值大概是800M/7.5GB）完成 TinyStories 数据集处理。

正式 tokenizer 配置：

```text
vocab size: 10,000
training corpus: TinyStoriesV2-GPT4-train.txt
special token: <|endoftext|>
```

正式 tokenizer 训练完成后保存为：

```text
vocab.json
merges.txt
```

### 3. 数据准备

训练集和验证集分别经过 tokenizer 编码，并保存为bin原始二进制文件（数据量较小时也可以直接用npy格式）：

```text
data/tinystories_train.bin
data/tinystories_valid.bin
```

为了避免再次把整个数据集加载到内存，训练时使用：

```python
np.memmap(
    path,
    dtype=np.uint16,
    mode="r",
)
```

最终数据规模：

```text
training tokens:   541,229,347
validation tokens:   5,465,883
```

### 4. 小规模训练验证

在正式训练之前，我先使用较小的数据和模型进行 smoke test，确认：

- 前向传播正常；
- 反向传播正常；
- loss 可以下降；
- 验证流程正常；
- checkpoint 可以保存；
- 从 checkpoint 恢复后 iteration 可以继续。

这一步可以提前发现数据路径、模块导入和 checkpoint 恢复方面的问题，避免在正式训练突然崩溃。

### 5. 云 GPU 训练

由于本地 WSL 的内存和计算资源有限，正式训练迁移到了 RTX 4090 云 GPU（显存24GB）。

最终训练配置：

```text
device: cuda
GPU: NVIDIA GeForce RTX 4090
vocab_size: 10,000
context_length: 256
d_model: 256
num_layers: 4
num_heads: 8
d_ff: 1,024
batch_size: 32
max_iters: 20,000
```

训练过程中使用：

```text
max learning rate: 3e-4
min learning rate: 3e-5
weight decay: 0.1
gradient clipping: 1.0
```

最终结果：

```text
train_loss: 1.7142
val_loss:   1.7355
```

对应的验证集困惑度约为：

```text
exp(1.7355) ≈ 5.67
```

训练得到的 checkpoint：

```text
data/checkpoints/tinystories_4090.pt
```

checkpoint 同时保存了：

- 模型参数；
- AdamW 优化器状态；
- 当前 iteration。

可以用于继续训练，也可以单独加载进行推理。

## 文本生成

加载训练好的 checkpoint 后，可以使用生成脚本（提示词可以自行修改）：

```bash
uv run python -u cs336_basics/generate.py
```

模型能够根据提示生成具有 TinyStories 风格的短故事，例如：

```text
Once upon a time, there was a little girl named Lily.
She loved to play with her toys in her room...
```

生成脚本支持调整：

- `temperature`
- `top_k`
- `max_new_tokens`

示例：

```python
temperature = 0.8
top_k = 50
max_new_tokens = 200
```

## 项目结构

```text
.
├── cs336_basics/
│   ├── checkpointing.py
│   ├── dataloader.py
│   ├── generate.py
│   ├── loss.py
│   ├── modules.py
│   ├── optimizers.py
│   ├── tokenizer.py
│   └── train.py
├── data/（参考，实际上传只有valid.txt）
│   ├── tinystories_tokenizer/
│   │   ├── vocab.json
│   │   └── merges.txt
│   ├── tinystories_train.bin
│   ├── tinystories_valid.bin
│   └── checkpoints/
│       └── tinystories_4090.pt
├── tests/
├── encode_full_data.py
├── prepare_data.py
├── train_full_tokenizer.py
├── pyproject.toml
└── uv.lock
```

## 环境配置

项目使用 `uv` 管理 Python 环境和依赖：

```bash
uv sync
```

运行测试：

```bash
uv run pytest
```

运行训练：

```bash
uv run python -u cs336_basics/train.py
```

运行文本生成：

```bash
uv run python -u cs336_basics/generate.py
```

## 主要收获

这个项目让我真正理解了一个语言模型从零开始运行所需要的完整链路：

```text
原始文本
→ tokenizer 训练
→ 文本编码
→ memmap 数据加载
→ Transformer 前向传播
→ loss 和反向传播
→ AdamW 更新
→ checkpoint 保存
→ checkpoint 加载
→ 文本生成
```

困难不是单独实现某一个函数，而是让所有模块在同一个环境下能够正确、平稳地运行。尤其是大文件 tokenizer 的内存问题、Windows 与 Linux 测试环境的差异、WSL 的 OOM，以及云 GPU 上的训练管理，让我对模型预训练有了更具体的认识。

## 说明

本项目是我的课程学习和实验实现，代码结构和参数配置服务于 CS336 Assignment 1 的学习目标。考虑到个人项目呈现以体现核心过程为主，删除了一些非必要的官方作业文件（比如ideas配置、agent.md等），增加了我个人实验中的训练日志（见logs，比较短），同时由于一些文件过大（比如tinystories原始txt，编码后的bin二进制文件以及训练好的权重ckpt等）没有上传，需要复现的小伙伴请参考官方文件或者其它教程

本人代码能力不算出众，部分非核心的代码的编写、优化借助了vibecoding，由于初学，还有大把时间花在了环境的配置上，但无论如何算是跑通了toy模型的训练全流程，如果觉得还行不妨给个star，也当是对我学习的鼓励。

感谢浏览！希望与大家一起学习进步，也欢迎大家指点
