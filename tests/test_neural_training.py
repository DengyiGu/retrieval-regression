"""Optional actual gradient-update check; no remote model or dataset required."""

from types import SimpleNamespace

import pytest

from experiments.reranking.training import train_model


@pytest.mark.neural
def test_training_loop_learns_separable_pairs():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(1, 1)
            torch.nn.init.zeros_(self.classifier.weight)
            torch.nn.init.zeros_(self.classifier.bias)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.classifier(input_ids))

    class Tokenizer:
        def __call__(self, queries, documents, **kwargs):
            return {
                "input_ids": torch.tensor(
                    [[1.0 if text == "positive" else -1.0] for text in documents]
                )
            }

    model = TinyModel()
    pairs = [{"qid": "q", "docid": "p", "label": 1}, {"qid": "q", "docid": "n", "label": 0}]
    config = {
        "seed": 42,
        "learning_rate": 0.2,
        "weight_decay": 0,
        "epochs": 5,
        "train_batch_size": 2,
        "max_length": 8,
    }
    inputs = torch.tensor([[1.0], [-1.0]])
    labels = torch.tensor([1.0, 0.0])
    before = torch.nn.functional.binary_cross_entropy_with_logits(
        model(inputs).logits[:, 0], labels
    )
    result = train_model(
        model, Tokenizer(), pairs, {"q": "query"}, {"p": "positive", "n": "negative"}, config, "cpu"
    )
    after = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs).logits[:, 0], labels)
    assert after < before
    assert result["steps"] == 5 and result["examples_seen"] == 10
    assert model.classifier.weight.item() > 0
