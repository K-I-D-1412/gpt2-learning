# GPT-2 自学项目：从这里开始

## 学习目标

目标：理解语言模型的基本结构、训练和生成，为后续 Agent 开发建立基础。
按模块理解计算原理、完成实现，再用小张量验证输出和梯度。

第一课配套笔记：[从文本、batch 到多头注意力](GPT2_ATTENTION_NOTES.md)，包含完整流程图、各步维度和 batch 并行计算的解释。

第二课配套笔记：[词元与位置嵌入](GPT2_EMBEDDING_NOTES.md)。包含查表与位置相加的数值例子、维度流程图、dropout、已完成的 `GPT2Model.embed` 实现回顾与验证记录。

## 学习路线

1. 理解张量形状、多头拆分和因果注意力，实现 `CausalSelfAttention.attention`。
2. 实现词元与位置嵌入、Transformer 层，串起完整前向计算。
3. 实现 Adam 参数更新，理解损失、梯度和优化器的分工。
4. 加载预训练权重，对齐参考模型输出，比较分类层训练和全量微调。
5. 完成复述检测与文本生成，分析正确与失败的例子。
6. 选择一个扩展做对照实验，然后衔接工具调用与 Agent 项目。

## 本机环境

项目采用独立 Python 3.12 环境，位置在项目根目录的 `.venv`。
课程原有 `env.yml` 保留；本机学习环境以 `requirements-local.lock.txt` 记录实际版本。
第一阶段使用 CPU 和小张量，不需要下载模型权重。后续训练前再验证设备支持和资源需求。

在终端中执行：

```bash
cd /Users/pengyu/Projects/gpt2-learning
source .venv/bin/activate
export HF_HOME="$PWD/work/huggingface"
python gpt2-first-steps.py
```

编辑器的 Python 解释器应选择：

```text
/Users/pengyu/Projects/gpt2-learning/.venv/bin/python
```

## 第一课：先认识输入，再写注意力

### GPT-2 在做什么

文本先被分词器转换为 token ID。token 可能是一个词、词的一部分或其他文本片段。
模型将 ID 转为向量，并加入位置信息；若干 Transformer 层更新这些向量。
生成时，最后一个位置的表示被映射为词表上的分数，再转换为概率，以选择下一个 token。
把新 token 加入上下文后重复这个过程。

注意：课程的 `GPT2Model` 首先输出隐藏表示；词表映射和具体任务的输出在相应模块里完成。

### 今天只读一个文件

打开 `modules/attention.py`，依次阅读：

1. `__init__`：同一份输入分别经过三个线性变换，得到 Q、K、V。
2. `transform`：把特征维度拆成多个注意力头，并调整维度顺序。
3. `forward`：调用上述变换，再交给已完成的 `attention` 计算注意力。

直觉上，Q 表示当前位置想寻找的信息，K 表示各位置用来匹配的特征，V 表示被汇总的信息。
它们都是通过训练得到的向量，不是人工填写的关键词或规则。

### 小例子

设 `B=2, T=4, D=8, H=2`：

- B：一次处理 2 个序列。
- T：每个序列有 4 个位置。
- D：每个位置用 8 个数表示。
- H：拆成 2 个头，每个头的维度为 `d=D/H=4`。

输入形状是 `[2,4,8]`，拆头后的 Q、K、V 形状都是 `[2,2,4,4]`。
随附观察脚本只调用起始代码已提供的 `transform`，不实现你的注意力函数。
脚本使用随机向量，因此用于理解结构，不能展示语言理解能力。

### 第一课复习：这三个问题

1. `Q @ K.transpose(-2, -1)` 的形状是什么？最后两个维度各表示什么？
2. 以 `I / like / deep / learning` 作为示意 token 序列，第 3 个位置允许读取哪些位置？为什么？
3. softmax 应沿哪一维计算，才能让每个 query 对所有 key 的权重加起来为 1？

注意力函数现已由你完成；复习时可以先口头推导，再对照自己的实现。
实现时还要区分两类掩码：因果掩码防止读取未来位置；padding 掩码排除补齐位置。
此项目传给 `attention` 的 padding 掩码已经是可加到分数上的形式：有效位置为 0，补齐位置为很大的负数。

## 验证原则

起步阶段完成导入、张量运算和反向传播检查。
实现注意力后，再检查输出形状、未来信息是否泄漏、padding 是否有效、梯度是否有限。

注意力的局部验证脚本为 [attention-check.py](attention-check.py)。本项目的 `attention()` 已由用户完成并通过该脚本验证；后续修改后可再次运行：

```bash
.venv/bin/python attention-check.py
```

它仅使用 CPU 小张量，对照独立参考计算，并检查掩码、梯度、batch 独立性和 dropout，不下载模型权重。

第二课的 `embed()` 也已由用户完成。运行 [embedding-check.py](embedding-check.py) 可单独检查查表、位置广播、梯度和 dropout，不执行 Transformer 层：

```bash
HF_HOME="$PWD/work/huggingface" .venv/bin/python embedding-check.py
```

补齐完整模型后才运行需要预训练权重的 `sanity_check.py`；未补齐时失败是预期现象。

## 来源

起始代码：https://github.com/cfifty/public_cs224n_gpt

本地起始版本：`7570cfa4385f3417298573c770df5ddfe2d97f89`。

原始项目说明保存在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。
