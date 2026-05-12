# Central hyperparameters, dataset paths, loss weights, and device (MPS on Apple Silicon).
from dataclasses import dataclass, field, asdict
from typing import List

import torch


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    # ---- Dataset ----
    dataset_name: str = "amazon_movies"
    data_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    min_interactions: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_explanation_length: int = 128

    # ---- Model ----
    embedding_dim: int = 64
    num_gcn_layers: int = 3
    num_aspects: int = 50
    num_selected_aspects: int = 5
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.1
    adapter_dim: int = 768
    flan_model_name: str = "google/flan-t5-base"
    dropout: float = 0.1

    # ---- Training ----
    device: str = field(default_factory=_auto_device)
    batch_size: int = 16
    num_epochs: int = 5
    warmup_epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_rec: float = 1.0
    lambda_text: float = 1.0
    lambda_faith: float = 0.5
    lambda_sparse: float = 0.01
    fidelity_margin: float = 0.1
    patience: int = 10
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # ---- Evaluation ----
    top_k_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    num_neg_samples: int = 100

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, f"splits must sum to 1.0, got {total}"
        assert self.num_selected_aspects <= self.num_aspects
        assert self.warmup_epochs <= self.num_epochs

    def as_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()


def _print_summary(cfg: Config) -> None:
    groups = {
        "Dataset": [
            "dataset_name", "data_dir", "processed_dir", "min_interactions",
            "train_ratio", "val_ratio", "test_ratio", "max_explanation_length",
        ],
        "Model": [
            "embedding_dim", "num_gcn_layers", "num_aspects",
            "num_selected_aspects", "gumbel_tau_start", "gumbel_tau_end",
            "adapter_dim", "flan_model_name", "dropout",
        ],
        "Training": [
            "device", "batch_size", "num_epochs", "warmup_epochs",
            "learning_rate", "weight_decay", "lambda_rec", "lambda_text",
            "lambda_faith", "lambda_sparse", "fidelity_margin", "patience",
            "save_dir", "log_dir",
        ],
        "Evaluation": ["top_k_list", "num_neg_samples"],
    }
    d = cfg.as_dict()
    width = max(len(k) for keys in groups.values() for k in keys)
    print("=" * 60)
    print("CaFE-Rec Config")
    print("=" * 60)
    for group, keys in groups.items():
        print(f"\n[{group}]")
        for k in keys:
            print(f"  {k.ljust(width)} = {d[k]}")
    print("=" * 60)


if __name__ == "__main__":
    _print_summary(CONFIG)
