"""Observe supplied Q/K/V projections; leave core attention for the learner."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.attention import CausalSelfAttention


def main():
    torch.manual_seed(42)
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        attention_probs_dropout_prob=0.0,
    )
    layer = CausalSelfAttention(config)
    x = torch.randn(2, 4, 8, requires_grad=True)
    print("PyTorch:", torch.__version__)
    print("输入 [B,T,D]:", list(x.shape))
    projections = []
    for name, linear in [("Q", layer.query), ("K", layer.key), ("V", layer.value)]:
        projected = layer.transform(x, linear)
        projections.append(projected)
        print(f"{name} [B,H,T,d]:", list(projected.shape))
    # A toy scalar verifies autograd only; it is not a language-model loss.
    loss = sum(p.square().mean() for p in projections)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("CPU 张量计算与反向传播：通过")
    print("本脚本仅运行投影与拆头；完整注意力验证请运行 attention-check.py。")
    print("复习：推导 Q @ K.transpose(-2, -1) 的形状及其含义。")


if __name__ == "__main__":
    main()
