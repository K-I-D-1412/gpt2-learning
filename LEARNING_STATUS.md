# 当前学习进度

更新：2026-09-24

- 起始仓库： https://github.com/cfifty/public_cs224n_gpt
- 起始提交：7570cfa4385f3417298573c770df5ddfe2d97f89
- 正式目录：/Users/pengyu/Projects/gpt2-learning
- 个人 GitHub 仓库：https://github.com/K-I-D-1412/gpt2-learning （公开，主分支 main）；原课程仓库作为 upstream 保留。
- README 已更新为个人自学项目说明，记录完成范围、运行方式、验证结果、贡献边界与后续计划；原课程 README 保存在 docs/UPSTREAM_README.md。
- 已准备起始代码、独立 Python 环境、学习指南和 Q/K/V 投影观察脚本。
- 用户已完成并保存 CausalSelfAttention.attention，CPU 小张量局部验证全部通过；嵌入、Transformer 层和优化器等其余待实现部分仍保留。尚未训练或下载 GPT-2 权重。
- 用户已完成 VS Code 打开项目、选择 .venv 解释器和运行观察脚本的操作，理解不必另外重复终端启动命令。
- 已讲解 token、batch、特征维度、多头和因果注意力的基本作用；当前继续第一课的注意力内部计算。
- 用户已正确推导 B=3、T=5、D=8、H=2 时，输入为 [3,5,8]，拆头后的 Q/K/V 为 [3,2,5,4]；已确认 D、H、d 的含义及 d=D/H。
- 用户理解线性层通过权重与偏置产生新向量，形状相同也有意义；已确认每个输出特征使用一组针对全部输入特征的权重，不同 token 共享同一线性层的参数。
- 用户理解 query/key/value 是三组独立线性层，Q/K/V 是各自对输入的计算结果；transform 中 proj 在线性变换后就已是对应的 Q/K/V，后续 rearrange 仅拆头并调整维度顺序。
- 已逐步核对点积、按 sqrt(d) 缩放、因果掩码、softmax 沿 key 维度归一化、权重乘 V、同一 token 的各头输出拼接；已解释每头 [T,T] 分数矩阵与完整 [B,H,T,T] 的区别。
- 已核对 padding 掩码 [B,1,1,T] 的广播及各序列分别屏蔽补齐 key；已讲解 softmax 后的 dropout、训练/评估差异和丢弃后行和不一定为 1，相关行为现已通过代码验证。
- 用户在开始实现之前明确要求暂停编码，表示整体流程仍混乱，尤其误以为一段话必须拆成若干 batch。需要区分 token 化、长文本切成序列、多个序列组成 batch，以及模型参数初始化与每次输入的向量计算。
- 已用 B=2、T=5、D=8、H=2 的统一流程图完成上述梳理；用户表示其他流程已明白，进一步询问 batch 为什么便于设备同时处理多个序列。
- 已按用户要求整理 GPT2_ATTENTION_NOTES.md，保留三张流程图、初始化说明、维度速查表，并补充独立计算、规则张量、底层批量算子与硬件并行的关系。START_HERE.md 已加入笔记入口。
- 用户已恢复编码，并在对话中逐段写出 attention 所需全部步骤：QK^T 与缩放、加性 padding 掩码、同设备布尔上三角因果掩码、masked_fill 负无穷、softmax(dim=-1)、已有 dropout、权重乘 value、rearrange(context, 'b h t d -> b t (h d)') 合头及 return；逐段检查均正确。
- 用户随后保存了完整 attention 实现，助手已阅读磁盘文件并确认各步正确，没有代写或修改用户的核心函数。
- 已新增并运行 attention-check.py：与 PyTorch scaled_dot_product_attention 独立参考对照，检查多组 B/T/D/H 形状和单 token 情况、输入与 Q/K/V 参数梯度、未来信息不泄漏、padding（含过去的被屏蔽 key）有效、不同 batch 行互不影响、dropout 训练/评估行为。CPU float64 最大参考输出误差 2.22e-16，所有检查通过。
- gpt2-first-steps.py 的投影与反向传播检查也通过；没有运行完整模型、没有加载预训练权重。

下一步：第一课 attention 实现与局部验证已完成。先解答用户对实现或验证的疑问；继续时进入词元与位置嵌入，以 models/gpt2.py 的 embed 为下一个小目标，先解释查表与位置相加再由用户实现。不要直接完成后续核心代码，不运行尚未实现的完整模型验证。
每次围绕一个小目标推进，不要直接给出完整 attention() 实现。
