"""Small Cross Encoder training/inference helpers; torch is imported only on use."""

from __future__ import annotations

import random
import time


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def load_model(path, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="eager",
        )
        .float()
        .to(device)
    )
    return model, tokenizer


def synchronize(device):
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def encode(tokenizer, pairs, device, max_length):
    batch = tokenizer(
        [q for q, _ in pairs],
        [d for _, d in pairs],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


class ModelScorer:
    def __init__(self, model, tokenizer, device, max_length):
        self.model, self.tokenizer = model, tokenizer
        self.device, self.max_length = device, max_length
        self.model.eval()

    def __call__(self, pairs):
        import torch

        with torch.inference_mode():
            logits = self.model(
                **encode(self.tokenizer, pairs, self.device, self.max_length)
            ).logits
        return logits.reshape(-1).float().cpu().tolist()


def train_model(model, tokenizer, pairs, queries, corpus, config, device):
    import torch

    seed_everything(config["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    model.train()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    start = time.perf_counter()
    steps, losses, total = 0, [], 0
    for epoch in range(config["epochs"]):
        indices = list(range(len(pairs)))
        random.Random(config["seed"] + epoch).shuffle(indices)
        for offset in range(0, len(indices), config["train_batch_size"]):
            batch = [pairs[i] for i in indices[offset : offset + config["train_batch_size"]]]
            texts = [(queries[p["qid"]], corpus[p["docid"]]) for p in batch]
            encoded = encode(tokenizer, texts, device, config["max_length"])
            labels = torch.tensor([p["label"] for p in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits.reshape(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            total += len(batch)
            steps += 1
            if steps % 50 == 0:
                print(f"training step={steps}, loss={losses[-1]:.6f}", flush=True)
    synchronize(device)
    return {
        "steps": steps,
        "examples_seen": total,
        "first_step_loss": losses[0],
        "last_step_loss": losses[-1],
        "mean_step_loss": sum(losses) / len(losses),
        "wall_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
    }
