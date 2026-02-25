"""Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.


We adapt shape suffixes style when working with tensors.
See https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd.

Dimension key:

b: batch size
l: max sequence length
c: number of classes
v: vocabulary size

For example,

feature_b_l means a tensor of shape (b, l) == (batch_size, max_sequence_length).
length_1 means a tensor of shape (1) == (1,).
loss means a tensor of shape (). You can retrieve the loss value with loss.item().
"""


from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple
import time
import statistics
import re
import string

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import DataPoint, DataType, load_data, save_results


class Tokenizer:
    TOK_PADDING_INDEX = 0
    TOK_UNK_INDEX = 1
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    NEGATION_KEEP = {
        "no", "nor", "not",
        "don", "don't",
        "didn", "didn't",
        "doesn", "doesn't",
        "isn", "isn't",
        "wasn", "wasn't",
        "weren", "weren't",
        "won", "won't",
        "wouldn", "wouldn't",
        "shouldn", "shouldn't",
        "couldn", "couldn't",
        "haven", "haven't",
        "hasn", "hasn't",
        "hadn", "hadn't",
        "mustn", "mustn't",
        "needn", "needn't",
        "mightn", "mightn't",
        "shan", "shan't",
        "aren", "aren't",
    }

    def __init__(
        self,
        data: List[DataPoint],
        max_vocab_size: int = None,
        remove_stopwords: bool = True,
    ):
        self.remove_stopwords = remove_stopwords

        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]

        # Reserve:
        # 0 = <PAD>, 1 = <UNK>, real tokens start at 2
        self.token2id = {
            "<PAD>": Tokenizer.TOK_PADDING_INDEX,
            "<UNK>": Tokenizer.TOK_UNK_INDEX,
        }
        for i, t in enumerate(tokens):
            self.token2id[t] = i + 2
        self.id2token = {i: t for t, i in self.token2id.items()}

    def _pre_process_text(self, text: str) -> List[str]:
        tokens: List[str] = []
        for tok in text.split():
            tok = tok.strip().lower()
            if not tok:
                continue
            # Remove stopwords (optionally), keep negations.
            if (
                self.remove_stopwords
                and tok in self.STOP_WORDS
                and tok not in self.NEGATION_KEEP
            ):
                continue
            tokens.append(tok)
        return tokens

    def tokenize(self, text: str) -> List[int]:
        toks = self._pre_process_text(text)
        return [self.token2id.get(t, Tokenizer.TOK_UNK_INDEX) for t in toks]


def get_label_mappings(
    data: List[DataPoint],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Deterministic label mapping."""
    labels = sorted(set(d.label for d in data if d.label is not None))
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return label2id, id2label


class BOWDataset(Dataset):
    def __init__(
        self,
        data: List[DataPoint],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_length: int = 100,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dp: DataPoint = self.data[idx]

        token_ids = self.tokenizer.tokenize(dp.text)
        token_ids = token_ids[: self.max_length]
        length = len(token_ids)

        pad_id = self.tokenizer.TOK_PADDING_INDEX
        if length < self.max_length:
            token_ids = token_ids + [pad_id] * (self.max_length - length)

        if dp.label is None:
            y = 0
        else:
            y = self.label2id[dp.label]

        features_l = torch.tensor(token_ids, dtype=torch.int64)
        length_t = torch.tensor(length, dtype=torch.int64)
        label_t = torch.tensor(y, dtype=torch.int64)

        return features_l, length_t, label_t


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        padding_index: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.padding_index = padding_index

        # Increased embedding dimension for better feature representation
        embed_dim = 256 
        hidden_dims = [256, 128]
        dropout_p = 0.3 # Increased dropout to reduce the 15% gap between dev/test

        activation = activation.lower()
        if activation == "relu":
            act_cls = nn.ReLU
        elif activation == "tanh":
            act_cls = nn.Tanh
        elif activation == "sigmoid":
            act_cls = nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_index,
        )
        
        # Initialize embeddings with a small variance
        nn.init.xavier_uniform_(self.embedding.weight.data)

        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            linear = nn.Linear(in_dim, h)
            # Kaiming initialization is better for ReLU
            nn.init.kaiming_normal_(linear.weight, nonlinearity=activation)
            layers.append(linear)
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h
        
        final_layer = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(final_layer.weight)
        layers.append(final_layer)
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor) -> torch.Tensor:
        # Shape: (batch, length, embed_dim)
        emb_b_l_d = self.embedding(input_features_b_l)

        # Create mask for padding tokens
        mask_b_l = (input_features_b_l != self.padding_index).to(emb_b_l_d.dtype)
        mask_b_l_1 = mask_b_l.unsqueeze(-1)

        # Masked Sum
        summed_b_d = (emb_b_l_d * mask_b_l_1).sum(dim=1)

        # Masked Average (Mean Pooling)
        # clamp(min=1) prevents division by zero for empty strings
        denom_b_1 = input_length_b.clamp(min=1).to(emb_b_l_d.dtype).unsqueeze(1)
        pooled_b_d = summed_b_d / denom_b_1

        output_b_c = self.mlp(pooled_b_d)
        return output_b_c


def _move_batch_to_device(batch, device: torch.device):
    inputs_b_l, lengths_b, labels_b = batch
    return inputs_b_l.to(device), lengths_b.to(device), labels_b.to(device)


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def predict(self, data: BOWDataset) -> List[int]:
        all_predictions: List[int] = []
        dataloader = DataLoader(data, batch_size=64, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs_b_l, lengths_b, labels_b = _move_batch_to_device(batch, self.device)
                logits_b_c = self.model(inputs_b_l, lengths_b)
                preds_b = torch.argmax(logits_b_c, dim=1)
                all_predictions.extend(preds_b.cpu().tolist())

        return all_predictions

    def evaluate(self, data: BOWDataset) -> float:
        self.model.eval()
        dataloader = DataLoader(data, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs_b_l, lengths_b, labels_b = _move_batch_to_device(batch, self.device)
                logits_b_c = self.model(inputs_b_l, lengths_b)
                preds_b = torch.argmax(logits_b_c, dim=1)
                correct += (preds_b == labels_b).sum().item()
                total += labels_b.size(0)

        return correct / total if total > 0 else 0.0

    def train(
        self,
        training_data: BOWDataset,
        val_data: BOWDataset,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    ) -> Tuple[List[float], List[float]]:
        criterion = nn.CrossEntropyLoss()
        best_val = -1.0
        best_state = None
        patience = 3 
        bad_epochs = 0

        g = torch.Generator()
        g.manual_seed(0)

        train_losses: List[float] = []
        val_accs: List[float] = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_examples = 0

            dataloader = DataLoader(training_data, batch_size=32, shuffle=True, generator=g)

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                inputs_b_l, lengths_b, labels_b = _move_batch_to_device(batch, self.device)
                optimizer.zero_grad()

                logits_b_c = self.model(inputs_b_l, lengths_b)
                loss = criterion(logits_b_c, labels_b)

                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                bs = labels_b.size(0)
                total_loss += loss.item() * bs
                total_examples += bs

            avg_loss = total_loss / total_examples
            val_acc = self.evaluate(val_data)

            train_losses.append(avg_loss)
            val_accs.append(val_acc)

            print(f"Epoch: {epoch + 1} | Loss: {avg_loss:.4f} | Val Acc: {100 * val_acc:.2f}%")

            if val_acc > best_val:
                best_val = val_acc
                bad_epochs = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best: {100 * best_val:.2f}%")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return train_losses, val_accs

def benchmark_inference(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_sizes: List[int],
    num_examples: int = 1000,
    repeats: int = 10,
) -> Dict[int, Tuple[float, float]]:
    """
    Returns:
        dict[batch_size] = (mean_seconds_per_1000, std_seconds_per_1000)
    """
    model.eval()
    results: Dict[int, Tuple[float, float]] = {}

    n = min(len(dataset), num_examples)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            # MPS is async; synchronize via torch.mps
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    for bs in batch_sizes:
        times: List[float] = []
        for _ in range(repeats):
            loader = DataLoader(dataset, batch_size=bs, shuffle=False)

            # warm-up
            with torch.no_grad():
                seen = 0
                for batch in loader:
                    x, l, _y = _move_batch_to_device(batch, device)
                    _ = model(x, l)
                    seen += x.size(0)
                    if seen >= min(200, n):
                        break

            _sync()
            start = time.perf_counter()

            with torch.no_grad():
                seen = 0
                for batch in loader:
                    x, l, _y = _move_batch_to_device(batch, device)
                    _ = model(x, l)
                    seen += x.size(0)
                    if seen >= n:
                        break

            _sync()
            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed * (num_examples / n))

        results[bs] = (statistics.mean(times), statistics.pstdev(times))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="newsgroups",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
        help="Activation function to use in the MLP",
    )

    # Device + benchmarking (for M4 Pro use: --device mps)
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Device to run on (mps recommended for Apple Silicon).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run batching inference benchmark (ms per 1000 examples).",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Benchmark repeats per batch size")
    parser.add_argument("--num_examples", type=int, default=1000, help="Benchmark examples (<= dataset size)")

    args = parser.parse_args()

    # Reproducibility
    random.seed(0)
    torch.manual_seed(0)

    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("[WARN] MPS requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    # Turn OFF stopword removal for SST-2 only, keep ON for newsgroups.
    remove_sw = (data_type != DataType("sst2"))
    tokenizer = Tokenizer(train_data, max_vocab_size=20000, remove_stopwords=remove_sw)

    label2id, id2label = get_label_mappings(train_data)
    pprint(id2label)

    max_length = 100
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
        activation=args.activation,
    ).to(device)

    trainer = Trainer(model, device=device)

    print("Training the model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_accs = trainer.train(train_ds, val_ds, optimizer, num_epochs)

    # save the plots for the writeup
    try:
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)

        # Plot: training loss
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Average Training Loss")
        plt.title(f"MLP Training Loss per Epoch ({args.data.upper()})")
        loss_fig_path = os.path.join("results", f"mlp_{args.data}_{args.activation}_train_loss.png")
        plt.savefig(loss_fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved loss plot to {loss_fig_path}")

        # Plot: validation accuracy
        plt.figure()
        plt.plot(range(1, len(val_accs) + 1), [100 * a for a in val_accs])
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.title(f"MLP Validation Accuracy per Epoch ({args.data.upper()})")
        acc_fig_path = os.path.join("results", f"mlp_{args.data}_{args.activation}_val_accuracy.png")
        plt.savefig(acc_fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved val accuracy plot to {acc_fig_path}")

    except ModuleNotFoundError:
        print("[WARN] matplotlib not installed; skipping plots.")

    # Dev accuracy
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Benchmark inference
    if args.benchmark:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        results = benchmark_inference(
            model=model,
            dataset=dev_ds,
            device=device,
            batch_sizes=batch_sizes,
            num_examples=args.num_examples,
            repeats=args.repeats,
        )

        print("\nInference benchmark (ms per 1000 examples)")
        print("BatchSize\tMean(ms)\tStd(ms)")
        for bs in batch_sizes:
            mean_s, std_s = results[bs]
            print(f"{bs}\t\t{mean_s*1000:.2f}\t\t{std_s*1000:.2f}")

    # Test predictions
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
    )
