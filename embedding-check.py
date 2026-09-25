"""CPU-only checks for the completed embed method; no weights are downloaded."""

import torch

from config import GPT2Config
from models.gpt2 import GPT2Model


def main():
    torch.manual_seed(7)
    model = GPT2Model(GPT2Config(
        vocab_size=9, hidden_size=4, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=6, hidden_dropout_prob=0.5,
    )).double()
    # Distinct, positive table entries make indexing and dropout observable.
    with torch.no_grad():
        model.word_embedding.weight.copy_(torch.arange(36).reshape(9, 4) + 1)
        model.pos_embedding.weight.copy_(100 * (torch.arange(24).reshape(6, 4) + 1))

    def expected(ids):
        result = torch.empty(*ids.shape, 4, dtype=torch.float64)
        for b in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                for d in range(4):
                    result[b, t, d] = (int(ids[b, t]) * 4 + d + 1
                                        + 100 * (t * 4 + d + 1))
        return result

    model.eval()
    cases = [torch.tensor([[2, 5, 2], [7, 1, 8]]),
             torch.tensor([[8], [2], [5]]),
             torch.tensor([[1, 2, 3, 4, 5, 6]])]
    for ids in cases:
        output = model.embed(ids)
        assert output.shape == (*ids.shape, 4)
        torch.testing.assert_close(output, expected(ids), rtol=0, atol=0)
    print('PASS: lookup values, shapes, positional broadcasting, T=1 and maximum T')

    ids = cases[0]
    changed = ids.clone()
    changed[1] = 3
    torch.testing.assert_close(model.embed(ids)[0], model.embed(changed)[0])
    print('PASS: batch rows are independent')

    # A weighted loss checks accumulation for repeated token IDs and positions.
    model.zero_grad(set_to_none=True)
    output = model.embed(ids)
    probe = torch.arange(1, output.numel() + 1, dtype=torch.float64).reshape_as(output)
    (output * probe).sum().backward()
    word_grad = torch.zeros_like(model.word_embedding.weight)
    pos_grad = torch.zeros_like(model.pos_embedding.weight)
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            word_grad[ids[b, t]] += probe[b, t]
            pos_grad[t] += probe[b, t]
    for actual, reference in [(model.word_embedding.weight.grad, word_grad),
                              (model.pos_embedding.weight.grad, pos_grad)]:
        assert actual is not None and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    print('PASS: exact word/position gradients, including repeated IDs and unused rows')

    model.train()
    torch.manual_seed(11)
    first = model.embed(ids)
    torch.manual_seed(11)
    torch.testing.assert_close(first, model.embed(ids), rtol=0, atol=0)
    torch.manual_seed(12)
    second = model.embed(ids)
    baseline = expected(ids)
    for output in (first, second):
        assert ((output == 0) | (output == baseline * 2)).all()
        assert (output == 0).any() and (output != 0).any()
    assert not torch.equal(first, second)
    model.eval()
    torch.testing.assert_close(model.embed(ids), baseline, rtol=0, atol=0)
    print('PASS: dropout randomness, scaling, and evaluation behavior')
    print('All embedding checks passed. Only embed() was called; no Transformer forward.')


if __name__ == '__main__':
    main()
