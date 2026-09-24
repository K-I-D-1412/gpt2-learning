"""Check the learner's completed attention using CPU tensors, without model weights.

Uses PyTorch's independent scaled-dot-product attention as a numerical reference.
An interior masked key makes the padding check independent of the causal check;
it is a diagnostic input, not a change to the project's right-padding convention.
"""

from types import SimpleNamespace

import torch
from torch.nn import functional as F

from modules.attention import CausalSelfAttention


def make_layer(hidden_size=12, heads=3, dropout=0.0):
    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        attention_probs_dropout_prob=dropout,
    )
    return CausalSelfAttention(config).double().cpu()


def additive_mask(valid):
    return (1.0 - valid.to(torch.float64))[:, None, None, :] * -10000.0


def reference(layer, x, valid):
    """Reference excludes masked keys with booleans, not additive scores."""
    batch, length, hidden = x.shape
    q = layer.transform(x, layer.query)
    k = layer.transform(x, layer.key)
    v = layer.transform(x, layer.value)
    allowed = torch.ones(length, length, dtype=torch.bool).tril()
    allowed = allowed[None, None, :, :] & valid[:, None, None, :]
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
    return result.transpose(1, 2).reshape(batch, length, hidden)


def check_reference_and_gradients():
    largest_error = 0.0
    for batch, length, hidden, heads in [(2, 5, 12, 3), (3, 7, 8, 2), (1, 1, 6, 3)]:
        layer = make_layer(hidden, heads).eval()
        x = torch.randn(batch, length, hidden, dtype=torch.float64, requires_grad=True)
        valid = torch.ones(batch, length, dtype=torch.bool)
        if length > 1:
            valid[-1, -2:] = False
            valid[0, 1] = False
        actual = layer(x, additive_mask(valid))
        expected = reference(layer, x, valid)
        assert actual.shape == (batch, length, hidden)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
        largest_error = max(largest_error, (actual - expected).abs().max().item())

        probe = torch.randn_like(actual)
        parameters = (x, *layer.parameters())
        actual_gradients = torch.autograd.grad((actual * probe).sum(), parameters)
        expected_gradients = torch.autograd.grad((expected * probe).sum(), parameters)
        for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-8, atol=1e-9)
        assert actual_gradients[0].abs().sum() > 0
    return largest_error


@torch.no_grad()
def check_causality():
    layer = make_layer().eval()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    mask = additive_mask(torch.ones(2, 5, dtype=torch.bool))
    original = layer(x, mask)
    for boundary in (1, 3, 4):
        changed = x.clone()
        changed[:, boundary:] += 4.0 * torch.randn_like(changed[:, boundary:])
        output = layer(changed, mask)
        torch.testing.assert_close(output[:, :boundary], original[:, :boundary], rtol=0, atol=1e-12)
        assert not torch.allclose(output[:, boundary:], original[:, boundary:])


@torch.no_grad()
def check_padding():
    layer = make_layer().eval()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    valid = torch.tensor([[True, False, True, True, True], [True, True, True, False, False]])
    mask = additive_mask(valid)
    original = layer(x, mask)
    changed = x.clone()
    changed[~valid] += 3.0 * torch.randn_like(changed[~valid])
    output = layer(changed, mask)
    torch.testing.assert_close(output[valid], original[valid], rtol=0, atol=1e-12)

    # Opening those keys must make this perturbation observable at later valid queries.
    open_mask = additive_mask(torch.ones_like(valid))
    open_original = layer(x, open_mask)
    open_changed = layer(changed, open_mask)
    assert not torch.allclose(open_original[0, 2:], open_changed[0, 2:])


@torch.no_grad()
def check_batch_independence():
    layer = make_layer().eval()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    valid = torch.ones(2, 5, dtype=torch.bool)
    original = layer(x, additive_mask(valid))
    changed = x.clone()
    changed[1] += 4.0 * torch.randn_like(changed[1])
    changed_valid = valid.clone()
    changed_valid[1, 2:] = False
    output = layer(changed, additive_mask(changed_valid))
    torch.testing.assert_close(output[0], original[0], rtol=0, atol=1e-12)


@torch.no_grad()
def check_dropout_modes():
    layer = make_layer(dropout=0.4)
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    valid = torch.ones(2, 5, dtype=torch.bool)
    mask = additive_mask(valid)
    layer.eval()
    expected = reference(layer, x, valid)
    torch.testing.assert_close(layer(x, mask), expected, rtol=1e-9, atol=1e-10)

    layer.train()
    torch.manual_seed(101)
    first = layer(x, mask)
    torch.manual_seed(101)
    repeated = layer(x, mask)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.manual_seed(202)
    different = layer(x, mask)
    assert torch.isfinite(first).all() and torch.isfinite(different).all()
    assert not torch.allclose(first, different)


def main():
    torch.manual_seed(42)
    error = check_reference_and_gradients()
    print(f"通过：多种输出形状、单 token、参考输出与输入/参数梯度；最大输出误差 {error:.2e}")
    for check, message in [
        (check_causality, "改变未来 token 不影响此前位置"),
        (check_padding, "被屏蔽的 key 不影响有效 query，包含过去位置的屏蔽"),
        (check_batch_independence, "不同序列的数据和掩码互不干扰"),
        (check_dropout_modes, "dropout 的训练随机性与评估模式"),
    ]:
        check()
        print(f"通过：{message}")
    print("attention 局部验证全部通过；未加载预训练权重，未运行完整模型。")


if __name__ == "__main__":
    main()
