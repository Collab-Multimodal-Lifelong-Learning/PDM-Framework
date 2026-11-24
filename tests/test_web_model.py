from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from pdm import web_model


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.embedding.requires_grad_(False)

    def forward(self, input_ids, attention_mask, use_cache):
        assert input_ids.shape == attention_mask.shape
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_base_model(self):
        return self

    def generate(self, input_ids, **kwargs):
        self.generated_prefix = input_ids
        self.generation_limit = kwargs["max_new_tokens"]
        return torch.cat([input_ids, input_ids.new_tensor([[3, 7]])], dim=1)


@pytest.fixture
def policy(monkeypatch):
    def build(device, seed):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = Backbone()
        tokenizer = SimpleNamespace(eos_token_id=7, encode=lambda text, **kw: [int(i) for i in text],
                                    decode=lambda ids, **kw: ''.join(str(int(i)) for i in ids if i != 7))
        return SimpleNamespace(tokenizer=tokenizer), model

    monkeypatch.setattr(web_model, 'build_policy', build)
    monkeypatch.setattr(web_model.WebPolicy, '_inputs', lambda self, system, prompt, image: {
        'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long)})
    return web_model.WebPolicy('cpu')


def example(action):
    return dict(system='system', prompt='prompt', image='image', action=action)


def test_action_loss_matches_masked_full_sequence_and_replay_mean(policy):
    expected = []
    for action in ('3', '456'):
        tokens = torch.tensor([1, 2, *map(int, action), 7])
        logits = policy.backbone.lm_head(policy.backbone.model.embedding(tokens[:-1]))
        labels = tokens[1:].clone()
        labels[0] = -100
        expected.append(nn.functional.cross_entropy(logits, labels))
    expected = torch.stack(expected).mean()
    parameter = policy.backbone.lm_head.weight
    gradient = torch.autograd.grad(expected, parameter)[0]
    loss = policy.update([example('3'), example('456')])
    assert loss == pytest.approx(expected.item())
    torch.testing.assert_close(policy.optimizer.state[parameter]['exp_avg'], 0.1 * gradient)
    policy.update([example('3')])
    assert policy.optimizer.state[parameter]['step'].item() == 2
    assert policy.generate('system', 'prompt', 'image') == '3'
    assert policy.backbone.generated_prefix.tolist() == [[1, 2]]


def test_snapshot_restores_exact_adapter_and_rejects_wrong_revision(policy, tmp_path):
    path = tmp_path / 'adapter.safetensors'
    policy.update([example('3')])
    policy.save(path)
    fresh = web_model.WebPolicy('cpu', seed=42)
    fresh.load(path)
    assert torch.equal(fresh.backbone.lm_head.weight, policy.backbone.lm_head.weight)
    weights = load_file(path)
    assert weights.keys() == {'lm_head.weight'}
    with safe_open(path, framework='pt') as stream:
        metadata = stream.metadata()
    assert metadata['model_revision'] == web_model.QWEN_REVISION
    metadata['model_revision'] = 'wrong'
    save_file(weights, path, metadata=metadata)
    with pytest.raises(ValueError, match='revision|metadata'):
        fresh.load(path)


def test_action_nll_scores_each_demonstration_without_changing_training_state(policy):
    rows = [example('3'), example('456')]
    policy.update(rows)
    params = list(policy.backbone.parameters())
    weights = [p.detach().clone() for p in params]
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    means = []
    with torch.no_grad():
        for row in rows:
            ids = [*map(int, row["action"]), 7]
            context = torch.tensor([2, *ids[:-1]])
            logits = policy.backbone.lm_head(policy.backbone.model.embedding(context))
            means.append(nn.functional.cross_entropy(logits, torch.tensor(ids)).item())
    modes = []
    with policy.backbone.model.register_forward_pre_hook(
        lambda m, _: modes.append((m.training, torch.is_inference_mode_enabled()))
    ):
        assert policy.action_nll(rows) == pytest.approx(sum(means) / len(means))
    assert modes == [(False, True), (False, True)]
    assert policy.backbone.training
    for p, weight, grad in zip(params, weights, grads, strict=True):
        assert torch.equal(p, weight)
        assert (p.grad is None) if grad is None else torch.equal(p.grad, grad)


def test_generation_honors_a_smaller_token_limit(policy):
    policy.generate('system', 'prompt', 'image', max_new_tokens=12)
    assert policy.backbone.generation_limit == 12
    for invalid in (0, -1, 129, True, 1.5):
        with pytest.raises(ValueError, match='max_new_tokens'):
            policy.generate('system', 'prompt', 'image', max_new_tokens=invalid)
