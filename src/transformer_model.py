"""
Transformer extension (DistilBERT) for binary sentiment classification.

Owner: Juan Diego Balanta (T7).

The model is fine-tuned end to end on the *raw* comment text (column ``text``):
DistilBERT ships with its own WordPiece tokenizer and was pre-trained on cased
punctuation and stopwords, so removing them (as done for the Dense / RNN / LSTM
models) would only throw information away, e.g. negations such as "not".

The training loop is written in plain PyTorch instead of the Hugging Face
``Trainer`` on purpose: every step (tokenization, forward pass, loss, optimizer
and scheduler step, validation) is visible and easy to explain in the defense.

Public API
----------
TransformerConfig            hyperparameters of one run
load_tokenizer_and_model     downloads (or builds) tokenizer + classifier
train_transformer            fine-tunes and keeps the best epoch on validation
predict_transformer          labels and positive-class probabilities
cls_attention                [CLS] attention of the last layer, for analysis
count_parameters             total / trainable parameters
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

DEFAULT_MODEL = "distilbert-base-uncased"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TransformerConfig:
    model_name: str = DEFAULT_MODEL
    max_len: int = 64            # GoEmotions comments have at most 30 words
    learning_rate: float = 3e-5
    batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: Optional[float] = None  # head dropout; None keeps DistilBERT default (0.2)
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)


def set_seed(seed: int = 42) -> None:
    """Fix every random source so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CommentDataset(Dataset):
    """Tokenizes once and serves tensors to the DataLoader."""

    def __init__(self, texts: Sequence[str], labels: Optional[Sequence[int]],
                 tokenizer, max_len: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = None if labels is None else torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_tokenizer_and_model(config: TransformerConfig, model=None, tokenizer=None):
    """
    Returns (tokenizer, model). DistilBERT + a classification head
    (pre_classifier Linear 768->768, ReLU, dropout, classifier Linear 768->2).

    ``model`` / ``tokenizer`` can be injected (used by the offline smoke test).
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if model is None:
        kwargs = {"num_labels": 2,
                  # eager attention so that output_attentions works (cls_attention)
                  "attn_implementation": "eager",
                  "id2label": {0: "negative", 1: "positive"},
                  "label2id": {"negative": 0, "positive": 1}}
        if config.dropout is not None:
            kwargs["seq_classif_dropout"] = config.dropout
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, **kwargs)
    return tokenizer, model


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _run_eval(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds, probs = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        if "labels" in batch:
            losses.append(out.loss.item())
        p = torch.softmax(out.logits, dim=-1)[:, 1]
        probs.append(p.cpu().numpy())
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, np.concatenate(preds), np.concatenate(probs)


def train_transformer(train_texts: Sequence[str], train_labels: Sequence[int],
                      val_texts: Sequence[str], val_labels: Sequence[int],
                      config: TransformerConfig = TransformerConfig(),
                      model=None, tokenizer=None, verbose: bool = True):
    """
    Fine-tunes the model and restores the weights of the epoch with the best
    validation macro-F1 (a simple form of early stopping). Macro-F1 is used
    instead of binary F1 because predicting "positive" for everything already
    gives a binary F1 of 0.67, while its macro-F1 is only 0.33.

    Returns (model, tokenizer, history) where history contains per-epoch
    losses / validation metrics, best_epoch, training time and parameter count.
    """
    from sklearn.metrics import accuracy_score, f1_score

    set_seed(config.seed)
    device = get_device()
    tokenizer, model = load_tokenizer_and_model(config, model, tokenizer)
    model.to(device)

    train_ds = CommentDataset(train_texts, train_labels, tokenizer, config.max_len)
    val_ds = CommentDataset(val_texts, val_labels, tokenizer, config.max_len)
    gen = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=64)

    # No weight decay on biases and LayerNorm weights (standard BERT recipe).
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_ratio * total_steps), total_steps)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": [],
                                       "val_macro_f1": []}
    best_f1, best_state, best_epoch = -1.0, None, 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        val_loss, val_pred, _ = _run_eval(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_pred)
        val_f1 = f1_score(val_labels, val_pred, average="macro", zero_division=0)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        if verbose:
            print(f"epoch {epoch}/{config.epochs}  train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    history.update({
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_f1),
        "train_time_sec": round(time.time() - start, 2),
        "parameters": count_parameters(model),
        "device": str(device),
        "config": config.to_dict(),
    })
    return model, tokenizer, history


def predict_transformer(model, tokenizer, texts: Sequence[str], max_len: int = 64,
                        batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (predicted labels, probability of the positive class)."""
    device = get_device()
    model.to(device)
    loader = DataLoader(CommentDataset(texts, None, tokenizer, max_len), batch_size=batch_size)
    _, preds, probs = _run_eval(model, loader, device)
    return preds, probs


@torch.no_grad()
def cls_attention(model, tokenizer, text: str, max_len: int = 64) -> Tuple[List[str], np.ndarray]:
    """
    Attention paid by the [CLS] token to every input token in the last layer,
    averaged over heads. [CLS] is the representation the classifier reads, so
    this gives a rough (not causal) view of which words the decision looks at.
    """
    device = get_device()
    if hasattr(model, "set_attn_implementation") and model.config._attn_implementation != "eager":
        model.set_attn_implementation("eager")
    model.to(device).eval()
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    att = out.attentions[-1][0].mean(dim=0)[0]          # (seq_len,)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, att.cpu().numpy()
