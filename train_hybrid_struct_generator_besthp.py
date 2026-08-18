#!/usr/bin/env python3
"""
train_hybrid_struct_generator_besthp.py

Training-only script for the hybrid conditional generator with four conditioning
features:

    RL + GC + MFE + dot-bracket secondary structure

This is the mode-free/comment-free replacement for the old "comment/uncomment
main section" workflow. It retrains one final model using the best hyper-parameters:

    batch_size = 32
    dropout    = 0.15
    ff_dim     = 256
    d_model    = 64
    epochs     = 8
    layers     = 3
    lr         = 1e-4

It also supports optional train/validation split so you can confirm the loss,
but it always saves a final full-data model by default.

Expected CSV columns:
    Required:
        sequence  OR  utr

    Optional:
        rl / RL / mrl / MRL / target_rl
        gc / GC / target_gc
        mfe / MFE / target_mfe
        structure / target_structure

If MFE or structure are missing, they are computed using ViennaRNA. If you pass
--modification_json, the script loads that JSON before folding.

Example:
    python train_hybrid_struct_generator_besthp.py \
      --train_csv benchmark_sequences.csv \
      --output_dir hybrid_struct_besthp_training \
      --out_model hybrid_struct_besthp_final.pt \
      --modification_json rna_mod_n1methylpseudouridine_parameters.json

Optional validation split:
    python train_hybrid_struct_generator_besthp.py \
      --train_csv benchmark_sequences.csv \
      --output_dir hybrid_struct_besthp_training \
      --val_fraction 0.1 \
      --modification_json rna_mod_n1methylpseudouridine_parameters.json
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import RNA  # ViennaRNA Python package
except Exception:
    RNA = None


# ============================================================
# Constants
# ============================================================

BASES = ["A", "C", "G", "U"]
BASE_TO_ID = {b: i for i, b in enumerate(BASES)}
ID_TO_BASE = {i: b for b, i in BASE_TO_ID.items()}
PAD_ID = 4
VOCAB_SIZE = 5

STRUCT_VOCAB = [".", "(", ")", "?"]
STRUCT_TO_ID = {ch: i for i, ch in enumerate(STRUCT_VOCAB)}
STRUCT_PAD_ID = len(STRUCT_VOCAB)
STRUCT_VOCAB_SIZE = STRUCT_PAD_ID + 1


# ============================================================
# Configs
# ============================================================

@dataclass
class FoldingConfig:
    modification_json: Optional[str] = None
    modified_base_symbol: str = "1"
    modified_base_replacement: str = "U"


@dataclass
class GeneratorConfig:
    seq_len: int = 50

    # Best hyper-parameters supplied by you
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.15
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 8

    device: str = "cpu"
    use_rl: bool = True
    use_gc: bool = True
    use_mfe: bool = True
    use_structure: bool = True
    method_name: str = "hybrid_rl_gc_mfe_structure_besthp_final_train"


FOLDING_CONFIG = FoldingConfig()


# ============================================================
# Utility functions
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_rna(seq: str) -> str:
    seq = str(seq).strip().upper().replace("T", "U")

    # For sequence modeling, the generator vocabulary is A/C/G/U only.
    # If your source file encodes modified U as "1", convert it to U.
    # The modified folding effect is handled by ViennaRNA parameter JSON.
    seq = seq.replace(FOLDING_CONFIG.modified_base_symbol, FOLDING_CONFIG.modified_base_replacement)

    bad = sorted(set(ch for ch in seq if ch not in BASES))
    if bad:
        raise ValueError(f"Invalid RNA sequence {seq!r}; invalid characters: {bad}")
    return seq


def prepare_sequence_for_folding(seq: str) -> str:
    seq = str(seq).strip().upper().replace("T", "U")
    symbol = FOLDING_CONFIG.modified_base_symbol
    replacement = FOLDING_CONFIG.modified_base_replacement
    if symbol and symbol != replacement:
        seq = seq.replace(symbol, replacement)
    bad = sorted(set(ch for ch in seq if ch not in BASES))
    if bad:
        raise ValueError(f"Invalid RNA sequence for folding {seq!r}; invalid characters: {bad}")
    return seq


def gc_fraction(seq: str) -> float:
    seq = normalize_rna(seq)
    return sum(ch in {"G", "C"} for ch in seq) / len(seq) if seq else 0.0


def maybe_float(row: Dict[str, str], *names: str) -> Optional[float]:
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except Exception:
                pass
    return None


def seq_to_ids(seq: str, fixed_len: int) -> List[int]:
    seq = normalize_rna(seq)
    ids = [BASE_TO_ID[ch] for ch in seq[:fixed_len]]
    if len(ids) < fixed_len:
        ids += [PAD_ID] * (fixed_len - len(ids))
    return ids


def ids_to_seq(ids: Sequence[int]) -> str:
    return "".join(ID_TO_BASE[int(x)] for x in ids if int(x) != PAD_ID)


def normalize_structure(struct: Optional[str], fixed_len: int) -> str:
    struct = (struct or "").strip()
    out = []
    for ch in struct[:fixed_len]:
        out.append(ch if ch in STRUCT_TO_ID else "?")
    if len(out) < fixed_len:
        out += ["?"] * (fixed_len - len(out))
    return "".join(out)


def structure_to_ids(struct: Optional[str], fixed_len: int) -> List[int]:
    struct = normalize_structure(struct, fixed_len)
    return [STRUCT_TO_ID.get(ch, STRUCT_TO_ID["?"]) for ch in struct]


def set_global_folding_config(
    modification_json: Optional[str] = None,
    modified_base_symbol: str = "1",
    modified_base_replacement: str = "U",
) -> None:
    FOLDING_CONFIG.modification_json = modification_json or None
    FOLDING_CONFIG.modified_base_symbol = (modified_base_symbol or "1").upper()
    FOLDING_CONFIG.modified_base_replacement = (modified_base_replacement or "U").upper()
    _load_vienna_params.cache_clear()


@lru_cache(maxsize=8)
def _load_vienna_params(modification_json: str) -> bool:
    if RNA is None:
        raise RuntimeError(
            "ViennaRNA Python module is not available. Install/import RNA or provide CSV columns for both MFE and structure."
        )
    if not modification_json:
        return True
    if not os.path.exists(modification_json):
        raise FileNotFoundError(f"ViennaRNA modification JSON not found: {modification_json}")

    last_error = None
    for loader_name in ("params_load", "read_parameter_file"):
        loader = getattr(RNA, loader_name, None)
        if loader is None:
            continue
        try:
            loader(modification_json)
            return True
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"Failed to load ViennaRNA JSON {modification_json!r}: {last_error}")
    raise RuntimeError("This ViennaRNA build does not expose params_load or read_parameter_file.")


def mfe_and_structure(seq: str) -> Tuple[float, str]:
    if RNA is None:
        raise RuntimeError(
            "ViennaRNA Python module is not available, and MFE/structure must be computed. "
            "Either install/import RNA or include mfe and structure columns in the CSV."
        )

    seq = prepare_sequence_for_folding(seq)
    if FOLDING_CONFIG.modification_json:
        _load_vienna_params(FOLDING_CONFIG.modification_json)

    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    return float(mfe), struct


# ============================================================
# Data loading
# ============================================================

def load_training_rows(csv_path: str, seq_len: int) -> List[Tuple[str, Dict[str, object]]]:
    """
    Loads examples for the RL+GC+MFE+structure conditional generator.

    If MFE/structure columns are missing, they are computed with mfe_and_structure(),
    which applies the modification JSON if --modification_json is supplied.
    """
    rows: List[Tuple[str, Dict[str, object]]] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        seq_col = "sequence" if "sequence" in fieldnames else ("utr" if "utr" in fieldnames else None)
        if seq_col is None:
            raise ValueError("CSV must contain either a 'sequence' or 'utr' column.")

        for i, row in enumerate(reader, start=1):
            raw_seq = row[seq_col]
            seq = normalize_rna(raw_seq)

            if len(seq) < seq_len:
                seq = seq + ("A" * (seq_len - len(seq)))
            else:
                seq = seq[:seq_len]

            rl = maybe_float(row, "target_rl", "rl", "RL", "mrl", "MRL")
            if rl is None:
                rl = 0.0

            gc = maybe_float(row, "target_gc", "gc", "GC")
            if gc is None:
                gc = gc_fraction(seq)

            mfe = maybe_float(row, "target_mfe", "mfe", "MFE")
            struct = row.get("target_structure") or row.get("structure") or None

            if mfe is None or struct in ("", None):
                folded_mfe, folded_struct = mfe_and_structure(seq)
                if mfe is None:
                    mfe = folded_mfe
                if struct in ("", None):
                    struct = folded_struct

            if mfe is None or math.isnan(float(mfe)):
                mfe = 0.0

            rows.append((
                seq,
                {
                    "rl": float(rl),
                    "gc": float(gc),
                    "mfe": float(mfe),
                    "structure": normalize_structure(struct, seq_len),
                },
            ))

            if i % 5000 == 0:
                print(f"Loaded/prepared {i} rows...")

    if not rows:
        raise RuntimeError(f"No rows loaded from {csv_path}")

    return rows


class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[str, Dict[str, object]]],
        cfg: GeneratorConfig,
    ):
        self.rows = list(rows)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq, meta = self.rows[idx]
        target_structure = normalize_structure(str(meta["structure"]), self.cfg.seq_len)

        return {
            "tokens": torch.tensor(seq_to_ids(seq, self.cfg.seq_len), dtype=torch.long),
            "target_rl": torch.tensor(float(meta["rl"]), dtype=torch.float32),
            "target_gc": torch.tensor(float(meta["gc"]), dtype=torch.float32),
            "target_mfe": torch.tensor(float(meta["mfe"]), dtype=torch.float32),
            "target_structure_ids": torch.tensor(structure_to_ids(target_structure, self.cfg.seq_len), dtype=torch.long),
            "presence_flags": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        }


def split_rows(
    rows: List[Tuple[str, Dict[str, object]]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[str, Dict[str, object]]], List[Tuple[str, Dict[str, object]]]]:
    if val_fraction <= 0:
        return rows, []

    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(len(rows) * val_fraction)))
    val_idx = set(idx[:n_val])

    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    return train_rows, val_rows


# ============================================================
# Model
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConditionEmbedder(nn.Module):
    """
    Embeds scalar conditions RL, GC, MFE plus presence flags.
    The fourth presence flag means structure conditioning is present.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.rl_proj = nn.Linear(1, d_model)
        self.gc_proj = nn.Linear(1, d_model)
        self.mfe_proj = nn.Linear(1, d_model)
        self.presence_proj = nn.Linear(4, d_model)
        self.final = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        target_rl: torch.Tensor,
        target_gc: torch.Tensor,
        target_mfe: torch.Tensor,
        presence_flags: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [
            self.rl_proj(target_rl.reshape(-1, 1)),
            self.gc_proj(target_gc.reshape(-1, 1)),
            self.mfe_proj(target_mfe.reshape(-1, 1)),
            self.presence_proj(presence_flags),
        ]
        return self.final(torch.cat(pieces, dim=-1))


class StructureConditionEncoder(nn.Module):
    """
    Converts target dot-bracket structure into a per-position conditioning signal.
    """
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(STRUCT_VOCAB_SIZE, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)

    def forward(self, struct_ids: torch.Tensor) -> torch.Tensor:
        return self.pos(self.embed(struct_ids))


class HybridStructureConditionalGenerator(nn.Module):
    """
    Conditional Transformer generator trained to reconstruct input UTR sequence
    from RL + GC + MFE + target dot-bracket structure conditions.
    """
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.seq_len)
        self.cond = ConditionEmbedder(cfg.d_model)
        self.struct_encoder = StructureConditionEncoder(cfg.d_model, max_len=cfg.seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, VOCAB_SIZE)

    def forward(
        self,
        tokens: torch.Tensor,
        target_rl: torch.Tensor,
        target_gc: torch.Tensor,
        target_mfe: torch.Tensor,
        presence_flags: torch.Tensor,
        target_structure_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pos_enc(self.token_embed(tokens))
        x = x + self.cond(target_rl, target_gc, target_mfe, presence_flags).unsqueeze(1)
        x = x + self.struct_encoder(target_structure_ids)
        x = self.encoder(x)
        return self.lm_head(x)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        target_rl: float,
        target_gc: float,
        target_mfe: float,
        target_structure: str,
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Included so this trained checkpoint can be reused later by your generation script.
        This training-only script does not call sample().
        """
        self.eval()
        device = next(self.parameters()).device
        seq_len = self.cfg.seq_len

        tokens = torch.full((n_samples, seq_len), PAD_ID, dtype=torch.long, device=device)
        presence = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * n_samples, dtype=torch.float32, device=device)
        trl = torch.tensor([target_rl] * n_samples, dtype=torch.float32, device=device)
        tgc = torch.tensor([target_gc] * n_samples, dtype=torch.float32, device=device)
        tmfe = torch.tensor([target_mfe] * n_samples, dtype=torch.float32, device=device)
        struct_ids = torch.tensor(
            [structure_to_ids(target_structure, seq_len)] * n_samples,
            dtype=torch.long,
            device=device,
        )

        for pos in range(seq_len):
            logits = self.forward(tokens, trl, tgc, tmfe, presence, struct_ids)
            probs = F.softmax(logits[:, pos, :4] / max(temperature, 1e-6), dim=-1)
            tokens[:, pos] = torch.multinomial(probs, num_samples=1).squeeze(1)

        return [ids_to_seq(row.tolist()) for row in tokens.cpu()]


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
) -> float:
    model.train()
    losses: List[float] = []

    for batch in loader:
        tokens = batch["tokens"].to(device)
        logits = model(
            tokens,
            batch["target_rl"].to(device),
            batch["target_gc"].to(device),
            batch["target_mfe"].to(device),
            batch["presence_flags"].to(device),
            batch["target_structure_ids"].to(device),
        )

        loss = F.cross_entropy(
            logits[:, :, :4].reshape(-1, 4),
            tokens.reshape(-1),
            ignore_index=PAD_ID,
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    if loader is None:
        return float("nan")

    model.eval()
    losses: List[float] = []

    for batch in loader:
        tokens = batch["tokens"].to(device)
        logits = model(
            tokens,
            batch["target_rl"].to(device),
            batch["target_gc"].to(device),
            batch["target_mfe"].to(device),
            batch["presence_flags"].to(device),
            batch["target_structure_ids"].to(device),
        )

        loss = F.cross_entropy(
            logits[:, :, :4].reshape(-1, 4),
            tokens.reshape(-1),
            ignore_index=PAD_ID,
        )
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: GeneratorConfig,
    folding_config: FoldingConfig,
    epoch_losses: List[Dict[str, float]],
    best_val_loss: float,
    best_epoch: Optional[int],
    train_csv: str,
    n_train: int,
    n_val: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "generator_config": asdict(cfg),
            "folding_config": asdict(folding_config),
            "epoch_losses": epoch_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_csv": train_csv,
            "n_train": n_train,
            "n_val": n_val,
            "hp": {
                "batch_size": cfg.batch_size,
                "dropout": cfg.dropout,
                "ff_dim": cfg.ff_dim,
                "gen_d_model": cfg.d_model,
                "gen_epochs": cfg.epochs,
                "gen_layers": cfg.n_layers,
                "lr": cfg.lr,
            },
            "feature_set": ["rl", "gc", "mfe", "structure"],
            "note": "Final training checkpoint for hybrid RL+GC+MFE+structure generator.",
        },
        path,
    )


def train_final_model(args: argparse.Namespace) -> None:
    set_global_folding_config(
        modification_json=args.modification_json,
        modified_base_symbol=args.modified_base_symbol,
        modified_base_replacement=args.modified_base_replacement,
    )
    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GeneratorConfig(
        seq_len=args.seq_len,
        d_model=64,
        n_heads=args.n_heads,
        n_layers=3,
        ff_dim=256,
        dropout=0.15,
        lr=1e-4,
        batch_size=32,
        epochs=8,
        device=device,
        method_name="hybrid_rl_gc_mfe_structure_besthp_final_train",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Training hybrid+structure generator with best hyper-parameters")
    print(json.dumps(asdict(cfg), indent=2))
    print("Folding config:")
    print(json.dumps(asdict(FOLDING_CONFIG), indent=2))
    print("=" * 80)

    rows = load_training_rows(args.train_csv, seq_len=cfg.seq_len)
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)

    train_dataset = GeneratorDataset(train_rows, cfg)
    val_dataset = GeneratorDataset(val_rows, cfg) if val_rows else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )

    model = HybridStructureConditionalGenerator(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    epoch_rows: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    best_model_path = os.path.join(args.output_dir, "best_val_" + args.out_model)

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        val_loss = evaluate_loss(model, val_loader, device) if val_loader is not None else float("nan")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        epoch_rows.append(row)

        print(
            f"[epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                best_model_path,
                model,
                cfg,
                FOLDING_CONFIG,
                epoch_rows,
                best_val_loss,
                best_epoch,
                args.train_csv,
                n_train=len(train_rows),
                n_val=len(val_rows),
            )

    runtime_sec = time.time() - start_time

    final_model_path = os.path.join(args.output_dir, args.out_model)
    save_checkpoint(
        final_model_path,
        model,
        cfg,
        FOLDING_CONFIG,
        epoch_rows,
        best_val_loss if val_loader is not None else float("nan"),
        best_epoch,
        args.train_csv,
        n_train=len(train_rows),
        n_val=len(val_rows),
    )

    # CSV losses
    losses_csv = os.path.join(args.output_dir, "training_losses.csv")
    with open(losses_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(epoch_rows)

    summary = {
        "train_csv": args.train_csv,
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "val_fraction": args.val_fraction,
        "final_model_path": final_model_path,
        "best_val_model_path": best_model_path if val_loader is not None else None,
        "best_val_loss": best_val_loss if val_loader is not None else None,
        "best_epoch": best_epoch,
        "final_train_loss": epoch_rows[-1]["train_loss"],
        "final_val_loss": epoch_rows[-1]["val_loss"],
        "runtime_sec": runtime_sec,
        "generator_config": asdict(cfg),
        "folding_config": asdict(FOLDING_CONFIG),
        "feature_set": ["rl", "gc", "mfe", "structure"],
        "loss_csv": losses_csv,
        "note": (
            "This retrains the hybrid+structure generator using the supplied best hyper-parameters. "
            "The dot-bracket structure is used as an additional positional conditioning feature."
        ),
    }

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    save_json(summary_path, summary)

    print("=" * 80)
    print("Saved final model:", final_model_path)
    if val_loader is not None:
        print("Saved best validation model:", best_model_path)
    print("Saved losses:", losses_csv)
    print("Saved summary:", summary_path)
    print("=" * 80)


def inspect_checkpoint(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    keys = sorted(list(ckpt.keys()))
    print("Checkpoint keys:")
    print(json.dumps(keys, indent=2))

    for key in ["generator_config", "hp", "fold_id", "hp_id", "best_val_loss", "best_epoch", "folding_config", "feature_set"]:
        if key in ckpt:
            print(f"\n{key}:")
            print(json.dumps(ckpt[key], indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final hybrid RL+GC+MFE+structure generator using best hyper-parameters."
    )

    parser.add_argument("--mode", choices=["train", "inspect_checkpoint"], default="train")
    parser.add_argument("--checkpoint", default=None, help="Used only with --mode inspect_checkpoint")

    parser.add_argument("--train_csv", default="benchmark_sequences.csv")
    parser.add_argument("--output_dir", default="hybrid_struct_besthp_training")
    parser.add_argument("--out_model", default="hybrid_struct_besthp_final.pt")

    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--val_fraction", type=float, default=0.0,
                        help="0.0 trains on all data. Use e.g. 0.1 for a validation check.")

    parser.add_argument("--modification_json", default=None,
                        help="Path to ViennaRNA modified-base parameter JSON, e.g. rna_mod_n1methylpseudouridine_parameters.json")
    parser.add_argument("--modified_base_symbol", default="1")
    parser.add_argument("--modified_base_replacement", default="U")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "inspect_checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode inspect_checkpoint")
        inspect_checkpoint(args.checkpoint)
        return

    train_final_model(args)


if __name__ == "__main__":
    main()
