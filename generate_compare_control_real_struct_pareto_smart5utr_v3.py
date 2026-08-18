#!/usr/bin/env python3
"""
Generate 500 sequences per scenario with a trained hybrid RL+GC+MFE+structure generator,
run a Smart5UTR-only baseline for the same scenarios, and aggregate comparison statistics.

This script is intentionally mode-based. You do NOT need to comment/uncomment main blocks.

Typical usage:

1) Show all scenarios:
   python generate_compare_control_real_hybrid_struct_pareto_smart5utr.py --mode list_scenarios --scenario_set full

2) Run one scenario by SLURM_ARRAY_TASK_ID:
   python generate_compare_control_real_hybrid_struct_pareto_smart5utr.py \
     --mode run_scenario \
     --scenario_set full \
     --generator_model hybrid_struct_besthp_training/hybrid_struct_besthp_model.pt \
     --train_csv benchmark_sequences.csv \
     --smart5utr_model ../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5 \
     --smart5utr_scaler ../models/egfp_m1pseudo2.scaler \
     --modification_json rna_mod_n1methylpseudouridine_parameters.json \
     --n_results 500 \
     --out_dir scenario_outputs_struct_besthp

3) Aggregate after all array tasks finish:
   python generate_compare_control_real_hybrid_struct_pareto_smart5utr.py \
     --mode aggregate \
     --scenario_set full \
     --out_dir scenario_outputs_struct_besthp
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")

try:
    import RNA
except Exception:
    RNA = None

from Smart5UTR.train import load_MTAE

BASES = ["A", "C", "G", "U"]
BASE_TO_ID = {b: i for i, b in enumerate(BASES)}
ID_TO_BASE = {i: b for b, i in BASE_TO_ID.items()}
PAD_ID = 4
VOCAB_SIZE = 5

STRUCT_VOCAB = [".", "(", ")", "?"]
STRUCT_TO_ID = {ch: i for i, ch in enumerate(STRUCT_VOCAB)}
STRUCT_PAD_ID = len(STRUCT_VOCAB)
STRUCT_VOCAB_SIZE = STRUCT_PAD_ID + 1

TARGET_RL_DEFAULT = 7.5
DEFAULT_CDS_START_CONTEXT = "AUGGCUAUGGCGGC"
CONTROL_EXPERIMENTS = ["uAUG", "uORF", "kozak", "accessibility", "top_like", "tisu"]



@dataclass
class FoldingConfig:
    modification_json: Optional[str] = None
    modified_base_symbol: str = "1"
    modified_base_replacement: str = "U"


@dataclass
class GeneratorConfig:
    seq_len: int = 50
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
    method_name: str = "hybrid_rl_gc_mfe_structure_besthp"
    n_initial_samples: int = 16
    temperature: float = 1.0


@dataclass
class DesignTargets:
    target_rl: Optional[float] = None
    target_gc: Optional[float] = None
    target_mfe: Optional[float] = None
    target_structure: Optional[str] = None
    required_motifs: Optional[List[str]] = None
    forbidden_motifs: Optional[List[str]] = None


@dataclass
class ObjectiveWeights:
    w_rl_value: float = 0.0
    w_rl_target: float = 4.0
    w_gc: float = 0.1
    w_mfe: float = 0.5
    w_structure: float = 0.5
    w_motif: float = 0.25
    w_edit_prior: float = 0.0


@dataclass
class OptimizerConfig:
    num_steps: int = 60
    candidates_per_step: int = 12
    num_mutations_per_candidate: int = 4
    success_tolerance_rl: float = 0.1
    beam_size: int = 3
    adaptive_mutation: bool = True
    early_mutations: int = 4
    middle_mutations: int = 2
    late_mutations: int = 1
    final_refine: bool = True
    final_refine_top_k: int = 3
    final_refine_positions_per_round: int = 8


@dataclass
class ParetoParticleConfig:
    rounds: int = 25
    population_size: int = 24
    elite_size: int = 12
    offspring_per_elite: int = 4
    edit_positions_early: int = 6
    edit_positions_middle: int = 4
    edit_positions_late: int = 2
    random_edit_fraction: float = 0.30
    transformer_edit_fraction: float = 0.70
    crowding_keep_fraction: float = 0.50
    success_tolerance_rl: float = 0.1
    final_pick_from_front_only: bool = True


@dataclass
class ParticleRecord:
    seq: str
    rl: float
    metrics: Dict[str, float]
    objectives: Tuple[float, ...]
    rank: int = -1
    crowding: float = 0.0


@dataclass
class ResultRow:
    method: str
    scenario: str
    design_id: int
    initial_seq: str
    initial_rl: float
    generated_best_seq: str
    generated_best_rl: float
    optimized_seq: str
    optimized_rl: float
    target_rl: float
    target_gc: float
    target_mfe: float
    target_structure: str
    required_motifs: str
    forbidden_motifs: str
    final_gc: float
    final_mfe: float
    final_structure: str
    rl_abs_error: float
    rl_sq_error: float
    gc_abs_error: float
    gc_sq_error: float
    mfe_abs_error: float
    mfe_sq_error: float
    structure_match: float
    structure_error: float
    structure_sq_error: float
    required_motif_success: float
    forbidden_motif_success: float
    motif_success: float
    motif_score: float
    total_score: float
    oracle_calls: int
    runtime_sec: float
    success_rl_tol: int


FOLDING_CONFIG = FoldingConfig()


def set_global_folding_config(modification_json: Optional[str], modified_base_symbol: str, modified_base_replacement: str) -> None:
    FOLDING_CONFIG.modification_json = modification_json or None
    FOLDING_CONFIG.modified_base_symbol = (modified_base_symbol or "1").upper()
    FOLDING_CONFIG.modified_base_replacement = (modified_base_replacement or "U").upper()
    _load_vienna_params.cache_clear()


@lru_cache(maxsize=8)
def _load_vienna_params(modification_json: str) -> bool:
    if RNA is None:
        return False
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
    raise RuntimeError(f"Failed to load ViennaRNA JSON with this RNA module: {last_error}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_rna(seq: str, allow_mod_symbol: bool = False) -> str:
    seq = str(seq).strip().upper().replace("T", "U")
    allowed = set(BASES)
    if allow_mod_symbol and FOLDING_CONFIG.modified_base_symbol:
        allowed.add(FOLDING_CONFIG.modified_base_symbol)
    bad = sorted(set(ch for ch in seq if ch not in allowed))
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


def mfe_and_structure(seq: str) -> Tuple[float, str]:
    if RNA is None:
        return float("nan"), ""
    seq = prepare_sequence_for_folding(seq)
    if FOLDING_CONFIG.modification_json:
        _load_vienna_params(FOLDING_CONFIG.modification_json)
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    return float(mfe), struct


def normalize_structure(struct: Optional[str], fixed_len: int) -> str:
    struct = (struct or "").strip()
    out = []
    for ch in struct[:fixed_len]:
        out.append(ch if ch in STRUCT_TO_ID else "?")
    if len(out) < fixed_len:
        out += ["?"] * (fixed_len - len(out))
    return "".join(out)


def structure_to_ids(struct: Optional[str], fixed_len: int) -> List[int]:
    struct = normalize_structure(struct or ("?" * fixed_len), fixed_len)
    return [STRUCT_TO_ID.get(ch, STRUCT_TO_ID["?"]) for ch in struct]


def dotbracket_match_fraction(pred: str, target: Optional[str]) -> float:
    if not target or not pred or len(pred) != len(target):
        return float("nan")
    return sum(a == b for a, b in zip(pred, target)) / len(target)


def seq_to_ids(seq: str, fixed_len: int) -> List[int]:
    seq = normalize_rna(seq)[:fixed_len]
    ids = [BASE_TO_ID[ch] for ch in seq]
    if len(ids) < fixed_len:
        ids += [PAD_ID] * (fixed_len - len(ids))
    return ids


def ids_to_seq(ids: Sequence[int]) -> str:
    return "".join(ID_TO_BASE[int(x)] for x in ids if int(x) != PAD_ID)


def random_mutation(seq: str, num_mut: int) -> str:
    seq_list = list(normalize_rna(seq))
    for _ in range(num_mut):
        idx = random.randint(0, len(seq_list) - 1)
        old = seq_list[idx]
        seq_list[idx] = random.choice([b for b in BASES if b != old])
    return "".join(seq_list)


def generate_unique_mutants(seq: str, n: int, num_mut: int) -> List[str]:
    out, seen = [], {seq}
    tries = 0
    while len(out) < n and tries < max(50, n * 20):
        cand = random_mutation(seq, num_mut)
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
        tries += 1
    return out


def count_motif_occurrences(seq: str, motif: str) -> int:
    seq = normalize_rna(seq)
    motif = normalize_rna(motif)
    count, start = 0, 0
    while True:
        idx = seq.find(motif, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1


def motif_metrics(seq: str, targets: DesignTargets) -> Tuple[float, float, float, float]:
    required = targets.required_motifs or []
    forbidden = targets.forbidden_motifs or []
    req_ok = all(count_motif_occurrences(seq, m) > 0 for m in required) if required else True
    forb_ok = all(count_motif_occurrences(seq, m) == 0 for m in forbidden) if forbidden else True
    motif_success = req_ok and forb_ok
    score = 0.0
    for m in required:
        score += 1.0 if count_motif_occurrences(seq, m) > 0 else -1.0
    for m in forbidden:
        score += 0.5 if count_motif_occurrences(seq, m) == 0 else -1.0
    return float(req_ok), float(forb_ok), float(motif_success), float(score)


def safe_float(x: Optional[float]) -> float:
    return float("nan") if x is None else float(x)


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
    def __init__(self, d_model: int):
        super().__init__()
        self.rl_proj = nn.Linear(1, d_model)
        self.gc_proj = nn.Linear(1, d_model)
        self.mfe_proj = nn.Linear(1, d_model)
        self.presence_proj = nn.Linear(4, d_model)
        self.final = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, target_rl, target_gc, target_mfe, presence_flags):
        return self.final(torch.cat([
            self.rl_proj(target_rl.reshape(-1, 1)),
            self.gc_proj(target_gc.reshape(-1, 1)),
            self.mfe_proj(target_mfe.reshape(-1, 1)),
            self.presence_proj(presence_flags),
        ], dim=-1))


class StructureConditionEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(STRUCT_VOCAB_SIZE, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)

    def forward(self, struct_ids: torch.Tensor) -> torch.Tensor:
        return self.pos(self.embed(struct_ids))


class HybridStructureConditionalGenerator(nn.Module):
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

    def forward(self, tokens, target_rl, target_gc, target_mfe, presence_flags, target_structure_ids):
        x = self.pos_enc(self.token_embed(tokens))
        x = x + self.cond(target_rl, target_gc, target_mfe, presence_flags).unsqueeze(1)
        x = x + self.struct_encoder(target_structure_ids)
        x = self.encoder(x)
        return self.lm_head(x)

    @torch.no_grad()
    def sample(self, n_samples: int, seq_len: int, targets: DesignTargets, temperature: float) -> List[str]:
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.full((n_samples, seq_len), PAD_ID, dtype=torch.long, device=device)
        presence = torch.tensor([[
            1.0 if targets.target_rl is not None else 0.0,
            1.0 if targets.target_gc is not None else 0.0,
            1.0 if targets.target_mfe is not None else 0.0,
            1.0 if targets.target_structure is not None else 0.0,
        ]] * n_samples, dtype=torch.float32, device=device)
        trl = torch.tensor([0.0 if targets.target_rl is None else targets.target_rl] * n_samples, dtype=torch.float32, device=device)
        tgc = torch.tensor([0.0 if targets.target_gc is None else targets.target_gc] * n_samples, dtype=torch.float32, device=device)
        tmfe = torch.tensor([0.0 if targets.target_mfe is None else targets.target_mfe] * n_samples, dtype=torch.float32, device=device)
        struct_ids = torch.tensor([structure_to_ids(targets.target_structure, seq_len)] * n_samples, dtype=torch.long, device=device)
        for pos in range(seq_len):
            logits = self.forward(tokens, trl, tgc, tmfe, presence, struct_ids)
            probs = F.softmax(logits[:, pos, :4] / max(temperature, 1e-6), dim=-1)
            tokens[:, pos] = torch.multinomial(probs, num_samples=1).squeeze(1)
        return [ids_to_seq(row.tolist()) for row in tokens.cpu()]


class Smart5UTRScorer:
    def __init__(self, model_path: str, scaler_path: str, input_length: int = 50, batch_size: int = 256):
        self.model, self.scaler = load_MTAE(model_path=model_path, scaler_path=scaler_path)
        self.input_length = input_length
        self.batch_size = batch_size
        self.oracle_calls = 0

    def reset_oracle_counter(self):
        self.oracle_calls = 0

    def _encode(self, seqs: Sequence[str]) -> np.ndarray:
        mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
        X = np.zeros((len(seqs), self.input_length, 4), dtype=np.float32)
        for i, seq in enumerate(seqs):
            seq = normalize_rna(seq)[:self.input_length]
            for j, ch in enumerate(seq):
                X[i, j, mapping[ch]] = 1.0
        return X

    def predict_rl(self, seqs: Sequence[str]) -> np.ndarray:
        seqs = [normalize_rna(s) for s in seqs]
        if not seqs:
            return np.asarray([], dtype=np.float32)
        self.oracle_calls += len(seqs)
        outputs = []
        for start in range(0, len(seqs), self.batch_size):
            batch = seqs[start:start + self.batch_size]
            raw = self.model.predict(self._encode(batch), verbose=0)
            if isinstance(raw, (list, tuple)):
                pred = None
                for item in raw:
                    arr = np.asarray(item)
                    if arr.ndim == 1 and arr.shape[0] == len(batch):
                        pred = arr
                        break
                    if arr.ndim == 2 and arr.shape[0] == len(batch) and arr.shape[1] == 1:
                        pred = arr[:, 0]
                        break
                if pred is None:
                    raise ValueError(f"Could not identify scalar Smart5UTR output. Shapes: {[np.asarray(x).shape for x in raw]}")
            else:
                arr = np.asarray(raw)
                if arr.ndim == 1 and arr.shape[0] == len(batch):
                    pred = arr
                elif arr.ndim == 2 and arr.shape[0] == len(batch) and arr.shape[1] == 1:
                    pred = arr[:, 0]
                else:
                    raise ValueError(f"Unexpected Smart5UTR output shape: {arr.shape}")
            if self.scaler is not None:
                pred = self.scaler.inverse_transform(np.asarray(pred).reshape(-1, 1)).reshape(-1)
            outputs.append(np.asarray(pred, dtype=np.float32))
        return np.concatenate(outputs, axis=0)


def load_generator_model(path: str, device: str) -> Tuple[HybridStructureConditionalGenerator, GeneratorConfig]:
    ckpt = torch.load(path, map_location=device)
    raw_cfg = ckpt.get("generator_config", ckpt.get("config", {}))
    allowed = set(GeneratorConfig.__dataclass_fields__.keys())
    clean = {k: v for k, v in raw_cfg.items() if k in allowed}
    cfg = GeneratorConfig(**clean)
    cfg.device = device
    model = HybridStructureConditionalGenerator(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def compute_total_score(seq: str, rl: float, targets: DesignTargets, weights: ObjectiveWeights, reference_seq: Optional[str] = None) -> Dict[str, float]:
    gc = gc_fraction(seq)
    mfe, struct = mfe_and_structure(seq)
    req_ok, forb_ok, motif_success, motif_score = motif_metrics(seq, targets)
    total = weights.w_rl_value * rl + weights.w_motif * motif_score

    rl_abs = rl_sq = float("nan")
    if targets.target_rl is not None:
        rl_abs = abs(rl - targets.target_rl)
        rl_sq = (rl - targets.target_rl) ** 2
        total -= weights.w_rl_target * rl_sq

    gc_abs = gc_sq = float("nan")
    if targets.target_gc is not None:
        gc_abs = abs(gc - targets.target_gc)
        gc_sq = (gc - targets.target_gc) ** 2
        total -= weights.w_gc * gc_abs

    mfe_abs = mfe_sq = float("nan")
    if targets.target_mfe is not None and not math.isnan(mfe):
        mfe_abs = abs(mfe - targets.target_mfe)
        mfe_sq = (mfe - targets.target_mfe) ** 2
        total -= weights.w_mfe * mfe_abs

    struct_match = struct_err = struct_sq = float("nan")
    if targets.target_structure:
        struct_match = dotbracket_match_fraction(struct, targets.target_structure)
        if not math.isnan(struct_match):
            struct_err = 1.0 - struct_match
            struct_sq = struct_err ** 2
            total -= weights.w_structure * struct_err

    if reference_seq is not None and weights.w_edit_prior:
        total -= weights.w_edit_prior * sum(a != b for a, b in zip(seq, reference_seq))

    return {
        "gc": gc, "mfe": mfe, "structure": struct,
        "rl_abs_error": rl_abs, "rl_sq_error": rl_sq,
        "gc_abs_error": gc_abs, "gc_sq_error": gc_sq,
        "mfe_abs_error": mfe_abs, "mfe_sq_error": mfe_sq,
        "structure_match": struct_match, "structure_error": struct_err, "structure_sq_error": struct_sq,
        "required_motif_success": req_ok, "forbidden_motif_success": forb_ok,
        "motif_success": motif_success, "motif_score": motif_score,
        "total_score": float(total),
    }


def get_num_mutations(step: int, cfg: OptimizerConfig) -> int:
    if not cfg.adaptive_mutation:
        return cfg.num_mutations_per_candidate
    if step < cfg.num_steps * 0.33:
        return cfg.early_mutations
    if step < cfg.num_steps * 0.75:
        return cfg.middle_mutations
    return cfg.late_mutations


def rank_candidates(candidates: Sequence[str], scorer: Smart5UTRScorer, targets: DesignTargets, weights: ObjectiveWeights, reference_seq: str):
    if not candidates:
        return []
    seqs = [normalize_rna(s) for s in candidates]
    rls = scorer.predict_rl(seqs)
    ranked = []
    for seq, rl in zip(seqs, rls):
        metrics = compute_total_score(seq, float(rl), targets, weights, reference_seq)
        ranked.append((seq, float(rl), metrics))
    ranked.sort(key=lambda x: x[2]["total_score"], reverse=True)
    return ranked


def dedup_ranked(ranked, top_k: int):
    out, seen = [], set()
    for seq, rl, metrics in ranked:
        if seq in seen:
            continue
        seen.add(seq)
        out.append((seq, rl, metrics))
        if len(out) >= top_k:
            break
    return out


def local_refine_single_base(beam, scorer, targets, weights, reference_seq, positions_per_round: int, top_k: int):
    beam = beam[:top_k]
    candidates = []
    for seq, _, _ in beam:
        positions = list(range(len(seq)))
        random.shuffle(positions)
        for pos in positions[:min(positions_per_round, len(seq))]:
            current = seq[pos]
            for base in BASES:
                if base != current:
                    candidates.append(seq[:pos] + base + seq[pos+1:])
    ranked = rank_candidates(candidates, scorer, targets, weights, reference_seq)
    merged = beam + ranked
    merged.sort(key=lambda x: x[2]["total_score"], reverse=True)
    return dedup_ranked(merged, top_k=len(beam))


def refine_with_optimizer(initial_seq: str, scorer: Smart5UTRScorer, targets: DesignTargets, weights: ObjectiveWeights, cfg: OptimizerConfig):
    initial_seq = normalize_rna(initial_seq)
    initial_rl = float(scorer.predict_rl([initial_seq])[0])
    initial_metrics = compute_total_score(initial_seq, initial_rl, targets, weights, reference_seq=initial_seq)
    beam = [(initial_seq, initial_rl, initial_metrics)]
    for step in range(cfg.num_steps):
        num_mut = get_num_mutations(step, cfg)
        candidates = []
        for seq, _, _ in beam:
            candidates.extend(generate_unique_mutants(seq, cfg.candidates_per_step, num_mut))
        candidates.extend([seq for seq, _, _ in beam])
        beam = dedup_ranked(rank_candidates(candidates, scorer, targets, weights, initial_seq), cfg.beam_size)
    if cfg.final_refine:
        beam = local_refine_single_base(beam, scorer, targets, weights, initial_seq, cfg.final_refine_positions_per_round, cfg.final_refine_top_k)
    return beam[0]


def make_result_row(method: str, scenario: str, design_id: int, initial_seq: str, initial_rl: float, generated_best_seq: str, generated_best_rl: float, optimized_seq: str, optimized_rl: float, targets: DesignTargets, metrics: Dict[str, float], oracle_calls: int, runtime: float, opt_cfg: OptimizerConfig) -> ResultRow:
    rl_abs = metrics["rl_abs_error"]
    success = int(targets.target_rl is not None and not math.isnan(rl_abs) and rl_abs <= opt_cfg.success_tolerance_rl)
    return ResultRow(
        method=method, scenario=scenario, design_id=design_id,
        initial_seq=initial_seq, initial_rl=float(initial_rl),
        generated_best_seq=generated_best_seq, generated_best_rl=float(generated_best_rl),
        optimized_seq=optimized_seq, optimized_rl=float(optimized_rl),
        target_rl=safe_float(targets.target_rl), target_gc=safe_float(targets.target_gc), target_mfe=safe_float(targets.target_mfe),
        target_structure=targets.target_structure or "",
        required_motifs=";".join(targets.required_motifs or []),
        forbidden_motifs=";".join(targets.forbidden_motifs or []),
        final_gc=float(metrics["gc"]), final_mfe=float(metrics["mfe"]), final_structure=str(metrics["structure"]),
        rl_abs_error=float(metrics["rl_abs_error"]), rl_sq_error=float(metrics["rl_sq_error"]),
        gc_abs_error=float(metrics["gc_abs_error"]), gc_sq_error=float(metrics["gc_sq_error"]),
        mfe_abs_error=float(metrics["mfe_abs_error"]), mfe_sq_error=float(metrics["mfe_sq_error"]),
        structure_match=float(metrics["structure_match"]), structure_error=float(metrics["structure_error"]), structure_sq_error=float(metrics["structure_sq_error"]),
        required_motif_success=float(metrics["required_motif_success"]), forbidden_motif_success=float(metrics["forbidden_motif_success"]), motif_success=float(metrics["motif_success"]),
        motif_score=float(metrics["motif_score"]), total_score=float(metrics["total_score"]),
        oracle_calls=int(oracle_calls), runtime_sec=float(runtime), success_rl_tol=success,
    )


def run_hybrid_one(design_id: int, scenario: str, targets: DesignTargets, generator, gen_cfg, scorer, weights, opt_cfg) -> ResultRow:
    t0 = time.time()
    scorer.reset_oracle_counter()
    generated = generator.sample(gen_cfg.n_initial_samples, gen_cfg.seq_len, targets, gen_cfg.temperature)
    ranked_gen = dedup_ranked(rank_candidates(generated, scorer, targets, weights, generated[0]), opt_cfg.beam_size)
    generated_best_seq, generated_best_rl, _ = ranked_gen[0]
    refined = [refine_with_optimizer(seq, scorer, targets, weights, opt_cfg) for seq, _, _ in ranked_gen]
    refined.sort(key=lambda x: x[2]["total_score"], reverse=True)
    best_seq, best_rl, best_metrics = refined[0]
    return make_result_row("hybrid_struct", scenario, design_id, generated_best_seq, generated_best_rl, generated_best_seq, generated_best_rl, best_seq, best_rl, targets, best_metrics, scorer.oracle_calls, time.time() - t0, opt_cfg)


def load_seed_sequences(train_csv: str, seq_len: int) -> List[str]:
    seqs = []
    with open(train_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        seq_col = "sequence" if "sequence" in fieldnames else ("utr" if "utr" in fieldnames else None)
        if seq_col is None:
            raise ValueError("train_csv must contain sequence or utr column for Smart5UTR baseline seeds.")
        for row in reader:
            try:
                seq = normalize_rna(row[seq_col])
            except Exception:
                continue
            if len(seq) < seq_len:
                seq = seq + "A" * (seq_len - len(seq))
            else:
                seq = seq[:seq_len]
            seqs.append(seq)
    if not seqs:
        raise RuntimeError(f"No usable seed sequences found in {train_csv}")
    return seqs


def run_baseline_one(design_id: int, scenario: str, targets: DesignTargets, seed_seq: str, scorer, weights, opt_cfg) -> ResultRow:
    t0 = time.time()
    scorer.reset_oracle_counter()
    initial_rl = float(scorer.predict_rl([seed_seq])[0])
    best_seq, best_rl, best_metrics = refine_with_optimizer(seed_seq, scorer, targets, weights, opt_cfg)
    return make_result_row("smart5utr_baseline", scenario, design_id, seed_seq, initial_rl, seed_seq, initial_rl, best_seq, best_rl, targets, best_metrics, scorer.oracle_calls, time.time() - t0, opt_cfg)



# ============================================================
# Real control-element experiment scoring
# ============================================================

def count_substring_overlapping(seq: str, motif: str) -> int:
    seq = normalize_rna(seq)
    motif = normalize_rna(motif)
    count = 0
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1


def count_uaugs(seq: str) -> int:
    return count_substring_overlapping(seq, "AUG")


def count_upstream_orfs(seq: str) -> int:
    seq = normalize_rna(seq)
    stops = {"UAA", "UAG", "UGA"}
    total = 0
    for i in range(0, len(seq) - 2):
        if seq[i:i + 3] == "AUG":
            for j in range(i + 3, len(seq) - 2, 3):
                if seq[j:j + 3] in stops:
                    total += 1
                    break
    return total


def normalize_degenerate_pattern(pattern: str) -> str:
    """Normalize an IUPAC/degenerated motif pattern without rejecting symbols such as R or S."""
    pattern = str(pattern).strip().upper().replace("T", "U")
    allowed = set("ACGURYSWKMBDHVN")
    bad = sorted(set(ch for ch in pattern if ch not in allowed))
    if bad:
        raise ValueError(f"Invalid degenerate motif pattern {pattern!r}; invalid characters: {bad}")
    return pattern


def score_degenerate_pattern(seq: str, pattern: str) -> float:
    # seq is a real RNA window and must contain only A/C/G/U.
    # pattern may contain IUPAC degenerate motif symbols, e.g. R in Kozak and S in TISU.
    seq = normalize_rna(seq)
    pattern = normalize_degenerate_pattern(pattern)
    code = {
        "A": {"A"}, "C": {"C"}, "G": {"G"}, "U": {"U"},
        "R": {"A", "G"}, "Y": {"C", "U"}, "S": {"C", "G"}, "W": {"A", "U"},
        "K": {"G", "U"}, "M": {"A", "C"},
        "B": {"C", "G", "U"}, "D": {"A", "G", "U"},
        "H": {"A", "C", "U"}, "V": {"A", "C", "G"},
        "N": {"A", "C", "G", "U"},
    }
    if len(seq) != len(pattern):
        return 0.0
    return sum(1 for s_ch, p_ch in zip(seq, pattern) if s_ch in code[p_ch]) / len(pattern)


def pattern_fully_satisfied(seq: str, pattern: str) -> bool:
    return score_degenerate_pattern(seq, pattern) >= 0.999999


def boundary_window(utr_seq: str, cds_start_context: str, left_offset: int, right_offset: int) -> str:
    utr_seq = normalize_rna(utr_seq)
    cds = normalize_rna(cds_start_context)
    full = utr_seq + cds
    start = len(utr_seq)
    left = start + left_offset
    right = start + right_offset
    if left < 0 or right > len(full) or left >= right:
        return ""
    return full[left:right]


def kozak_window(utr_seq: str, cds_start_context: str = DEFAULT_CDS_START_CONTEXT) -> str:
    return boundary_window(utr_seq, cds_start_context, -6, 4)


def tisu_window(utr_seq: str, cds_start_context: str = DEFAULT_CDS_START_CONTEXT) -> str:
    return boundary_window(utr_seq, cds_start_context, -5, 7)


def kozak_score_full(utr_seq: str, cds_start_context: str = DEFAULT_CDS_START_CONTEXT) -> float:
    window = kozak_window(utr_seq, cds_start_context)
    return score_degenerate_pattern(window, "GCCRCCAUGG") if window else 0.0


def kozak_success(utr_seq: str, cds_start_context: str = DEFAULT_CDS_START_CONTEXT) -> float:
    window = kozak_window(utr_seq, cds_start_context)
    return float(bool(window) and pattern_fully_satisfied(window, "GCCRCCAUGG"))


def tisu_score_full(utr_seq: str, cds_start_context: str = DEFAULT_CDS_START_CONTEXT) -> float:
    window = tisu_window(utr_seq, cds_start_context)
    return score_degenerate_pattern(window, "SAASAUGGCGGC") if window else 0.0


def tisu_success(utr_seq: str, cds_start_context: str = DEFAULT_CDS_START_CONTEXT) -> float:
    window = tisu_window(utr_seq, cds_start_context)
    return float(bool(window) and pattern_fully_satisfied(window, "SAASAUGGCGGC"))


def top_like_score(seq: str) -> float:
    seq = normalize_rna(seq)
    if not seq:
        return 0.0
    k = min(7, len(seq))
    window = seq[:k]
    starts_c = 1.0 if window and window[0] == "C" else 0.0
    pyr_frac = sum(nt in {"C", "U"} for nt in window) / max(1, len(window))
    return float(0.4 * starts_c + 0.6 * pyr_frac)


def top_like_success(seq: str) -> float:
    seq = normalize_rna(seq)
    k = min(7, len(seq))
    window = seq[:k]
    if not window:
        return 0.0
    pyr_frac = sum(nt in {"C", "U"} for nt in window) / len(window)
    return float(window[0] == "C" and pyr_frac >= 0.75)


def accessibility_score(utr_seq: str, cds_start_context: str = DEFAULT_CDS_START_CONTEXT) -> float:
    utr_seq = normalize_rna(utr_seq)
    cds = normalize_rna(cds_start_context)
    full = utr_seq + cds
    start_idx = len(utr_seq)
    left = max(0, start_idx - 15)
    right = min(len(full), start_idx + 15)
    region = full[left:right]
    if not region:
        return 0.0
    if RNA is not None:
        try:
            fc = RNA.fold_compound(region)
            struct, _ = fc.mfe()
            local_aug = start_idx - left
            s_left = max(0, local_aug - 6)
            s_right = min(len(struct), local_aug + 6)
            sub = struct[s_left:s_right]
            return float(sub.count(".") / max(1, len(sub)))
        except Exception:
            pass
    return float(sum(nt in {"A", "U"} for nt in region) / len(region))


def control_experiment_metrics(seq: str, pred_rl: float, experiment: str, target_rl: float, cds_start_context: str, accessibility_threshold: float) -> Dict[str, object]:
    seq = normalize_rna(seq)
    n_uaug = count_uaugs(seq)
    n_uorf = count_upstream_orfs(seq)
    kozak = kozak_score_full(seq, cds_start_context)
    kozak_ok = kozak_success(seq, cds_start_context)
    tisu = tisu_score_full(seq, cds_start_context)
    tisu_ok = tisu_success(seq, cds_start_context)
    top_s = top_like_score(seq)
    top_ok = top_like_success(seq)
    access = accessibility_score(seq, cds_start_context)
    rl_abs = abs(float(pred_rl) - float(target_rl))
    rl_s = 1.0 / (1.0 + rl_abs)
    if experiment == "uAUG":
        motif_score = 1.0 / (1.0 + n_uaug)
        supports = float(n_uaug == 0)
        primary = motif_score
    elif experiment == "uORF":
        motif_score = 1.0 / (1.0 + n_uorf)
        supports = float(n_uorf == 0)
        primary = motif_score
    elif experiment == "kozak":
        motif_score = kozak
        supports = kozak_ok
        primary = kozak
    elif experiment == "tisu":
        motif_score = tisu
        supports = tisu_ok
        primary = tisu
    elif experiment == "top_like":
        motif_score = top_s
        supports = top_ok
        primary = top_s
    elif experiment == "accessibility":
        motif_score = access
        supports = float(access >= accessibility_threshold)
        primary = access
    else:
        raise ValueError(f"Unknown control experiment: {experiment}")
    return {
        "control_experiment": experiment,
        "control_supports_constraint": float(supports),
        "control_rank_primary": float(primary),
        "control_final_score": float(rl_s + motif_score),
        "control_rl_score": float(rl_s),
        "n_uAUG": float(n_uaug),
        "n_uORF": float(n_uorf),
        "kozak_window": kozak_window(seq, cds_start_context),
        "kozak_score": float(kozak),
        "kozak_exact_match": float(kozak_ok),
        "tisu_window": tisu_window(seq, cds_start_context),
        "tisu_score": float(tisu),
        "tisu_exact_match": float(tisu_ok),
        "top_like_score": float(top_s),
        "top_like_exact_support": float(top_ok),
        "accessibility_score": float(access),
        "accessibility_support": float(access >= accessibility_threshold),
    }


def make_control_objective(experiment: str, target_rl: float, cds_start_context: str, accessibility_threshold: float):
    def _score(seq: str, rl: float, targets: DesignTargets, weights: ObjectiveWeights, reference_seq: Optional[str] = None) -> Dict[str, float]:
        base = ORIGINAL_COMPUTE_TOTAL_SCORE(seq, rl, DesignTargets(target_rl=target_rl), weights, reference_seq)
        ctrl = control_experiment_metrics(seq, rl, experiment, target_rl, cds_start_context, accessibility_threshold)
        base["motif_score"] = float(ctrl["control_rank_primary"])
        base["motif_success"] = float(ctrl["control_supports_constraint"])
        base["required_motif_success"] = float(ctrl["control_supports_constraint"])
        base["forbidden_motif_success"] = float(ctrl["control_supports_constraint"])
        base["total_score"] = float(ctrl["control_final_score"])
        return base
    return _score


# ============================================================
# Pareto-guided refinement updated for structure-conditioned generator
# ============================================================

def build_pareto_objectives(seq: str, rl: float, metrics: Dict[str, float], targets: DesignTargets) -> Tuple[float, ...]:
    vals: List[float] = []
    if targets.target_rl is not None and not math.isnan(float(metrics["rl_abs_error"])):
        vals.append(-float(metrics["rl_abs_error"]))
    else:
        vals.append(float(rl))
    if targets.target_gc is not None and not math.isnan(float(metrics["gc_abs_error"])):
        vals.append(-float(metrics["gc_abs_error"]))
    if targets.target_mfe is not None and not math.isnan(float(metrics["mfe_abs_error"])):
        vals.append(-float(metrics["mfe_abs_error"]))
    if targets.target_structure:
        vals.append(float(metrics["structure_match"]) if not math.isnan(float(metrics["structure_match"])) else -1e9)
    vals.append(float(metrics["motif_score"]))
    vals.append(float(metrics["total_score"]))
    return tuple(vals)


def evaluate_particles(seqs: Sequence[str], scorer: Smart5UTRScorer, targets: DesignTargets, weights: ObjectiveWeights, reference_seq: Optional[str]) -> List[ParticleRecord]:
    if not seqs:
        return []
    seqs = [normalize_rna(s) for s in seqs]
    rls = scorer.predict_rl(seqs)
    records = []
    for seq, rl in zip(seqs, rls):
        metrics = compute_total_score(seq, float(rl), targets, weights, reference_seq)
        records.append(ParticleRecord(seq=seq, rl=float(rl), metrics=metrics, objectives=build_pareto_objectives(seq, float(rl), metrics, targets)))
    return records


def dominates(a: ParticleRecord, b: ParticleRecord) -> bool:
    return all(av >= bv for av, bv in zip(a.objectives, b.objectives)) and any(av > bv for av, bv in zip(a.objectives, b.objectives))


def fast_non_dominated_sort(population: List[ParticleRecord]) -> List[List[int]]:
    n = len(population)
    dominates_set = [set() for _ in range(n)]
    dominated_count = [0] * n
    fronts: List[List[int]] = [[]]
    for p_idx in range(n):
        for q_idx in range(n):
            if p_idx == q_idx:
                continue
            if dominates(population[p_idx], population[q_idx]):
                dominates_set[p_idx].add(q_idx)
            elif dominates(population[q_idx], population[p_idx]):
                dominated_count[p_idx] += 1
        if dominated_count[p_idx] == 0:
            population[p_idx].rank = 0
            fronts[0].append(p_idx)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in dominates_set[p_idx]:
                dominated_count[q_idx] -= 1
                if dominated_count[q_idx] == 0:
                    population[q_idx].rank = i + 1
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def assign_crowding_distance(population: List[ParticleRecord], front: List[int]) -> None:
    if not front:
        return
    m = len(population[front[0]].objectives)
    for idx in front:
        population[idx].crowding = 0.0
    if len(front) <= 2:
        for idx in front:
            population[idx].crowding = float("inf")
        return
    for obj_idx in range(m):
        ordered = sorted(front, key=lambda i: population[i].objectives[obj_idx])
        population[ordered[0]].crowding = float("inf")
        population[ordered[-1]].crowding = float("inf")
        lo = population[ordered[0]].objectives[obj_idx]
        hi = population[ordered[-1]].objectives[obj_idx]
        if hi == lo:
            continue
        for j in range(1, len(ordered) - 1):
            population[ordered[j]].crowding += (population[ordered[j + 1]].objectives[obj_idx] - population[ordered[j - 1]].objectives[obj_idx]) / (hi - lo)


def select_population(population: List[ParticleRecord], keep_n: int) -> List[ParticleRecord]:
    if len(population) <= keep_n:
        return population
    fronts = fast_non_dominated_sort(population)
    selected: List[ParticleRecord] = []
    for front in fronts:
        assign_crowding_distance(population, front)
        members = [population[i] for i in front]
        if len(selected) + len(members) <= keep_n:
            selected.extend(members)
        else:
            members.sort(key=lambda x: (x.crowding, x.metrics["total_score"]), reverse=True)
            selected.extend(members[:max(0, keep_n - len(selected))])
            break
    return selected


def get_front(population: List[ParticleRecord], rank: int = 0) -> List[ParticleRecord]:
    fronts = fast_non_dominated_sort(population)
    if rank >= len(fronts):
        return []
    assign_crowding_distance(population, fronts[rank])
    return [population[i] for i in fronts[rank]]


def get_pareto_edit_count(round_idx: int, cfg: ParetoParticleConfig) -> int:
    if round_idx < cfg.rounds * 0.33:
        return cfg.edit_positions_early
    if round_idx < cfg.rounds * 0.75:
        return cfg.edit_positions_middle
    return cfg.edit_positions_late


@torch.no_grad()
def structure_generator_position_logits(generator: HybridStructureConditionalGenerator, seq: str, targets: DesignTargets) -> torch.Tensor:
    device = next(generator.parameters()).device
    tokens = torch.tensor([seq_to_ids(seq, fixed_len=generator.cfg.seq_len)], dtype=torch.long, device=device)
    presence = torch.tensor([[
        1.0 if targets.target_rl is not None else 0.0,
        1.0 if targets.target_gc is not None else 0.0,
        1.0 if targets.target_mfe is not None else 0.0,
        1.0 if targets.target_structure else 0.0,
    ]], dtype=torch.float32, device=device)
    trl = torch.tensor([0.0 if targets.target_rl is None else targets.target_rl], dtype=torch.float32, device=device)
    tgc = torch.tensor([0.0 if targets.target_gc is None else targets.target_gc], dtype=torch.float32, device=device)
    tmfe = torch.tensor([0.0 if targets.target_mfe is None else targets.target_mfe], dtype=torch.float32, device=device)
    struct_ids = torch.tensor([structure_to_ids(targets.target_structure, generator.cfg.seq_len)], dtype=torch.long, device=device)
    generator.eval()
    return generator(tokens, trl, tgc, tmfe, presence, struct_ids)[0, :, :4].detach().cpu()


def mutate_with_structure_generator(generator: HybridStructureConditionalGenerator, seq: str, targets: DesignTargets, num_positions: int, temperature: float, random_fraction: float) -> str:
    seq = normalize_rna(seq)
    seq_list = list(seq)
    logits = structure_generator_position_logits(generator, seq, targets)
    probs_all = F.softmax(logits, dim=-1)
    entropy = -(probs_all * torch.log(probs_all.clamp_min(1e-12))).sum(dim=-1)
    positions = torch.argsort(entropy, descending=True).tolist()[:min(num_positions, len(seq_list))]
    for pos in positions:
        current = seq_list[pos]
        if random.random() < random_fraction:
            seq_list[pos] = random.choice([b for b in BASES if b != current])
        else:
            probs = F.softmax(logits[pos] / max(temperature, 1e-6), dim=-1).numpy().astype(np.float64)
            probs[BASE_TO_ID[current]] *= 0.25
            probs = probs / probs.sum()
            seq_list[pos] = ID_TO_BASE[int(np.random.choice(np.arange(4), p=probs))]
    return "".join(seq_list)


def run_pareto_one(design_id: int, scenario: str, targets: DesignTargets, generator, gen_cfg, scorer, weights, particle_cfg: ParetoParticleConfig, opt_cfg: OptimizerConfig) -> ResultRow:
    t0 = time.time()
    scorer.reset_oracle_counter()
    initial_population = generator.sample(particle_cfg.population_size, gen_cfg.seq_len, targets, gen_cfg.temperature)
    population = select_population(evaluate_particles(initial_population, scorer, targets, weights, None), particle_cfg.population_size)
    generated_best = max(population, key=lambda r: r.metrics["total_score"])
    reference_seq = generated_best.seq
    for round_idx in range(particle_cfg.rounds):
        edit_count = get_pareto_edit_count(round_idx, particle_cfg)
        elites = get_front(population, 0) or select_population(population, particle_cfg.elite_size)
        elites.sort(key=lambda p: (p.crowding, p.metrics["total_score"]), reverse=True)
        elites = elites[:particle_cfg.elite_size]
        offspring_seqs = []
        seen = {p.seq for p in population}
        for elite in elites:
            for _ in range(particle_cfg.offspring_per_elite):
                cand = mutate_with_structure_generator(generator, elite.seq, targets, edit_count, max(gen_cfg.temperature, 0.8), particle_cfg.random_edit_fraction)
                if random.random() < (1.0 - particle_cfg.transformer_edit_fraction):
                    cand = random_mutation(cand, max(1, edit_count // 2))
                if cand not in seen:
                    seen.add(cand)
                    offspring_seqs.append(cand)
        if not offspring_seqs:
            break
        merged = population + evaluate_particles(offspring_seqs, scorer, targets, weights, reference_seq)
        merged.sort(key=lambda r: r.metrics["total_score"], reverse=True)
        prekeep = max(particle_cfg.population_size, int(len(merged) * particle_cfg.crowding_keep_fraction))
        population = select_population(merged[:prekeep], particle_cfg.population_size)
    final_candidates = get_front(population, 0) if particle_cfg.final_pick_from_front_only else population
    if not final_candidates:
        final_candidates = population
    best = max(final_candidates, key=lambda r: r.metrics["total_score"])
    return make_result_row("pareto_struct", scenario, design_id, generated_best.seq, generated_best.rl, generated_best.seq, generated_best.rl, best.seq, best.rl, targets, best.metrics, scorer.oracle_calls, time.time() - t0, opt_cfg)


def build_scenarios(scenario_set: str) -> List[Tuple[str, DesignTargets]]:
    fixed_structure = "....(((...)))......((....))......................."  # length 50
    base = [
        ("rl_only_rl8", DesignTargets(target_rl=8.0)),
        ("rl_only_rl7", DesignTargets(target_rl=7.0)),
        ("rl_only_rl5", DesignTargets(target_rl=5.0)),
        ("rl_only_rl3", DesignTargets(target_rl=3.0)),
        ("rl_gc_rl3_gc040", DesignTargets(target_rl=3.0, target_gc=0.40)),
        ("rl_gc_rl7_gc050", DesignTargets(target_rl=7.0, target_gc=0.50)),
        ("rl_gc_rl8_gc060", DesignTargets(target_rl=8.0, target_gc=0.60)),
        ("rl_mfe_rl8_mfe-5", DesignTargets(target_rl=8.0, target_mfe=-5.0)),
        ("rl_mfe_rl7_mfe-6", DesignTargets(target_rl=7.0, target_mfe=-6.0)),
        ("rl_mfe_rl3_mfe-10", DesignTargets(target_rl=3.0, target_mfe=-10.0)),
        ("rl_gc_mfe_rl8_gc050_mfe-10", DesignTargets(target_rl=8.0, target_gc=0.50, target_mfe=-10.0)),
    ]
    structure_grid = [
        ("rl_struct_rl8_fixed", DesignTargets(target_rl=8.0, target_structure=fixed_structure)),
        ("rl_struct_rl7_fixed", DesignTargets(target_rl=7.0, target_structure=fixed_structure)),
        ("rl_struct_rl5_fixed", DesignTargets(target_rl=5.0, target_structure=fixed_structure)),
        ("rl_struct_rl3_fixed", DesignTargets(target_rl=3.0, target_structure=fixed_structure)),
    ]
    motif_control = [
        ("motif_rl3", DesignTargets(target_rl=3.0, required_motifs=["AUG"])),
        ("motif_rl5", DesignTargets(target_rl=5.0, required_motifs=["AUG"])),
        ("motif_rl7", DesignTargets(target_rl=7.0, required_motifs=["AUG"])),
        ("control_panel_rl5", DesignTargets(target_rl=5.0, forbidden_motifs=["AUG", "UUU"])),
        ("control_panel_rl7", DesignTargets(target_rl=7.0, forbidden_motifs=["AUG", "UUU"])),
    ]
    real_control = [(f"real_{name}", DesignTargets(target_rl=TARGET_RL_DEFAULT)) for name in CONTROL_EXPERIMENTS]
    if scenario_set == "core15":
        return base + structure_grid
    if scenario_set == "original17":
        return base + [("structure_rl7", DesignTargets(target_rl=7.0, target_structure=fixed_structure))] + motif_control
    if scenario_set == "full":
        return base + structure_grid + motif_control
    if scenario_set == "control_real":
        return real_control
    if scenario_set == "full_plus_control_real":
        return base + structure_grid + motif_control + real_control
    raise ValueError(f"Unknown scenario_set: {scenario_set}")


def is_real_control_scenario(scenario_name: str) -> bool:
    return scenario_name.startswith("real_") and scenario_name.replace("real_", "", 1) in CONTROL_EXPERIMENTS


def control_name_from_scenario(scenario_name: str) -> str:
    return scenario_name.replace("real_", "", 1)


ORIGINAL_COMPUTE_TOTAL_SCORE = compute_total_score

def numeric_values(rows: List[Dict[str, object]], key: str) -> List[float]:
    vals = []
    for r in rows:
        try:
            v = float(r.get(key, float("nan")))
        except Exception:
            continue
        if not math.isnan(v):
            vals.append(v)
    return vals


def mean_key(rows, key):
    vals = numeric_values(rows, key)
    return float(np.mean(vals)) if vals else float("nan")


def median_key(rows, key):
    vals = numeric_values(rows, key)
    return float(np.median(vals)) if vals else float("nan")


def std_key(rows, key):
    vals = numeric_values(rows, key)
    return float(np.std(vals)) if vals else float("nan")


def rmse_from_sq(rows, sq_key):
    vals = numeric_values(rows, sq_key)
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")


def summarize_method(rows: List[Dict[str, object]], scenario: str, method: str, targets: DesignTargets) -> Dict[str, object]:
    mrows = [r for r in rows if r.get("method") == method]
    return {
        "scenario": scenario,
        "method": method,
        "n": len(mrows),
        "target_rl": targets.target_rl,
        "target_gc": targets.target_gc,
        "target_mfe": targets.target_mfe,
        "target_structure": targets.target_structure,
        "required_motifs": targets.required_motifs,
        "forbidden_motifs": targets.forbidden_motifs,
        "mean_optimized_rl": mean_key(mrows, "optimized_rl"),
        "median_optimized_rl": median_key(mrows, "optimized_rl"),
        "std_optimized_rl": std_key(mrows, "optimized_rl"),
        "mean_rl_abs_error": mean_key(mrows, "rl_abs_error"),
        "median_rl_abs_error": median_key(mrows, "rl_abs_error"),
        "rmse_rl_error": rmse_from_sq(mrows, "rl_sq_error"),
        "mean_gc_abs_error": mean_key(mrows, "gc_abs_error"),
        "median_gc_abs_error": median_key(mrows, "gc_abs_error"),
        "rmse_gc_error": rmse_from_sq(mrows, "gc_sq_error"),
        "mean_mfe_abs_error": mean_key(mrows, "mfe_abs_error"),
        "median_mfe_abs_error": median_key(mrows, "mfe_abs_error"),
        "rmse_mfe_error": rmse_from_sq(mrows, "mfe_sq_error"),
        "mean_structure_match": mean_key(mrows, "structure_match"),
        "mean_structure_error": mean_key(mrows, "structure_error"),
        "rmse_structure_error": rmse_from_sq(mrows, "structure_sq_error"),
        "required_motif_success_rate": mean_key(mrows, "required_motif_success"),
        "forbidden_motif_success_rate": mean_key(mrows, "forbidden_motif_success"),
        "motif_success_rate": mean_key(mrows, "motif_success"),
        "success_rl_tolerance_rate": mean_key(mrows, "success_rl_tol"),
        "mean_total_score": mean_key(mrows, "total_score"),
        "mean_runtime_sec": mean_key(mrows, "runtime_sec"),
        "mean_oracle_calls": mean_key(mrows, "oracle_calls"),
        "control_support_rate": mean_key(mrows, "control_supports_constraint"),
        "mean_control_rank_primary": mean_key(mrows, "control_rank_primary"),
        "mean_control_final_score": mean_key(mrows, "control_final_score"),
        "mean_n_uAUG": mean_key(mrows, "n_uAUG"),
        "mean_n_uORF": mean_key(mrows, "n_uORF"),
        "mean_kozak_score": mean_key(mrows, "kozak_score"),
        "kozak_exact_success_rate": mean_key(mrows, "kozak_exact_match"),
        "mean_tisu_score": mean_key(mrows, "tisu_score"),
        "tisu_exact_success_rate": mean_key(mrows, "tisu_exact_match"),
        "mean_top_like_score": mean_key(mrows, "top_like_score"),
        "top_like_success_rate": mean_key(mrows, "top_like_exact_support"),
        "mean_accessibility_score": mean_key(mrows, "accessibility_score"),
        "accessibility_success_rate": mean_key(mrows, "accessibility_support"),
    }


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_scenario(args) -> None:
    global compute_total_score

    set_seed(args.seed)
    set_global_folding_config(args.modification_json, args.modified_base_symbol, args.modified_base_replacement)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    scenarios = build_scenarios(args.scenario_set)
    task_id = args.scenario_id
    if task_id is None:
        env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task is None:
            raise ValueError("Provide --scenario_id or run with SLURM_ARRAY_TASK_ID.")
        task_id = int(env_task)
    if task_id < 0 or task_id >= len(scenarios):
        raise ValueError(f"Invalid scenario_id={task_id}. Expected 0..{len(scenarios)-1} for scenario_set={args.scenario_set}")
    scenario_name, targets = scenarios[task_id]
    print(f"Running scenario {task_id}/{len(scenarios)-1}: {scenario_name}")

    real_control = is_real_control_scenario(scenario_name)
    if real_control:
        exp_name = control_name_from_scenario(scenario_name)
        targets = DesignTargets(target_rl=args.target_rl)
        compute_total_score = make_control_objective(exp_name, args.target_rl, args.cds_start_context, args.accessibility_threshold)
    else:
        compute_total_score = ORIGINAL_COMPUTE_TOTAL_SCORE

    generator, gen_cfg = load_generator_model(args.generator_model, device)
    gen_cfg.n_initial_samples = args.n_initial_samples
    gen_cfg.temperature = args.temperature
    scorer = Smart5UTRScorer(args.smart5utr_model, args.smart5utr_scaler, input_length=gen_cfg.seq_len, batch_size=args.predict_batch_size)
    weights = ObjectiveWeights()
    opt_cfg = OptimizerConfig(
        num_steps=args.num_steps,
        candidates_per_step=args.candidates_per_step,
        beam_size=args.beam_size,
        final_refine=not args.no_final_refine,
    )
    particle_cfg = ParetoParticleConfig(
        rounds=args.pareto_rounds,
        population_size=args.pareto_population_size,
        elite_size=args.pareto_elite_size,
        offspring_per_elite=args.pareto_offspring_per_elite,
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed_methods = {"hybrid_struct", "pareto_struct", "smart5utr_baseline"}
    bad = sorted(set(methods) - allowed_methods)
    if bad:
        raise ValueError(f"Unknown method(s): {bad}; allowed: {sorted(allowed_methods)}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    scenario_dir = os.path.join(out_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    method_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}
    seed_pool = load_seed_sequences(args.train_csv, gen_cfg.seq_len) if "smart5utr_baseline" in methods else []

    for i in range(1, args.n_results + 1):
        hybrid_initial_seq = None

        if "hybrid_struct" in methods:
            print(f"[{scenario_name}] hybrid_struct {i}/{args.n_results}")
            h = run_hybrid_one(i, scenario_name, targets, generator, gen_cfg, scorer, weights, opt_cfg)
            hrow = asdict(h)
            if real_control:
                hrow.update(control_experiment_metrics(hrow["optimized_seq"], float(hrow["optimized_rl"]), exp_name, args.target_rl, args.cds_start_context, args.accessibility_threshold))
            method_rows["hybrid_struct"].append(hrow)
            rows.append(hrow)
            hybrid_initial_seq = h.initial_seq

        if "pareto_struct" in methods:
            print(f"[{scenario_name}] pareto_struct {i}/{args.n_results}")
            p = run_pareto_one(i, scenario_name, targets, generator, gen_cfg, scorer, weights, particle_cfg, opt_cfg)
            prow = asdict(p)
            if real_control:
                prow.update(control_experiment_metrics(prow["optimized_seq"], float(prow["optimized_rl"]), exp_name, args.target_rl, args.cds_start_context, args.accessibility_threshold))
            method_rows["pareto_struct"].append(prow)
            rows.append(prow)

        if "smart5utr_baseline" in methods:
            print(f"[{scenario_name}] smart5utr_baseline {i}/{args.n_results}")
            if args.baseline_seed_source == "hybrid_initial" and hybrid_initial_seq is not None:
                seed_seq = hybrid_initial_seq
            else:
                seed_seq = seed_pool[(i - 1) % len(seed_pool)] if args.deterministic_baseline_seeds else random.choice(seed_pool)
            b = run_baseline_one(i, scenario_name, targets, seed_seq, scorer, weights, opt_cfg)
            brow = asdict(b)
            if real_control:
                brow.update(control_experiment_metrics(brow["optimized_seq"], float(brow["optimized_rl"]), exp_name, args.target_rl, args.cds_start_context, args.accessibility_threshold))
            method_rows["smart5utr_baseline"].append(brow)
            rows.append(brow)

        if args.flush_every > 0 and i % args.flush_every == 0:
            for method, mr in method_rows.items():
                write_csv(os.path.join(scenario_dir, f"{scenario_name}_{method}_results.csv"), mr)
            write_csv(os.path.join(scenario_dir, f"{scenario_name}_combined_results.csv"), rows)

    for method, mr in method_rows.items():
        write_csv(os.path.join(scenario_dir, f"{scenario_name}_{method}_results.csv"), mr)
    write_csv(os.path.join(scenario_dir, f"{scenario_name}_combined_results.csv"), rows)

    summary_rows = [summarize_method(rows, scenario_name, method, targets) for method in methods]
    write_csv(os.path.join(scenario_dir, f"{scenario_name}_summary_comparison.csv"), summary_rows)
    with open(os.path.join(scenario_dir, f"{scenario_name}_summary_comparison.json"), "w") as f:
        json.dump({"scenario": scenario_name, "scenario_id": task_id, "methods": methods, "summaries": summary_rows}, f, indent=2)
    compute_total_score = ORIGINAL_COMPUTE_TOTAL_SCORE
    print(json.dumps(summary_rows, indent=2))


def aggregate(args) -> None:
    scenarios = build_scenarios(args.scenario_set)
    all_summary = []
    missing = []
    for scenario_name, targets in scenarios:
        path = os.path.join(args.out_dir, scenario_name, f"{scenario_name}_combined_results.csv")
        if not os.path.exists(path):
            missing.append(scenario_name)
            continue
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        methods = sorted(set(r.get("method", "") for r in rows if r.get("method", "")))
        for method in methods:
            all_summary.append(summarize_method(rows, scenario_name, method, targets))
    out_csv = os.path.join(args.out_dir, "all_scenarios_summary_comparison.csv")
    write_csv(out_csv, all_summary)
    with open(os.path.join(args.out_dir, "all_scenarios_summary_comparison.json"), "w") as f:
        json.dump({"scenario_set": args.scenario_set, "n_expected_scenarios": len(scenarios), "missing_scenarios": missing, "summaries": all_summary}, f, indent=2)
    print(f"Saved: {out_csv}")
    if missing:
        print("Missing scenarios:")
        for name in missing:
            print(f"  - {name}")
    else:
        print(f"All {len(scenarios)} scenarios were found and aggregated.")


def parse_args():
    p = argparse.ArgumentParser(description="Generate hybrid-structure and Smart5UTR baseline sequences for all scenarios.")
    p.add_argument("--mode", choices=["list_scenarios", "run_scenario", "aggregate", "run_all_serial"], required=True)
    p.add_argument("--scenario_set", choices=["core15", "original17", "full", "control_real", "full_plus_control_real"], default="full_plus_control_real")
    p.add_argument("--scenario_id", type=int, default=None)
    p.add_argument("--generator_model", default="hybrid_struct_besthp_training/hybrid_struct_besthp_model.pt")
    p.add_argument("--train_csv", default="benchmark_sequences.csv")
    p.add_argument("--smart5utr_model", default="../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5")
    p.add_argument("--smart5utr_scaler", default="../models/egfp_m1pseudo2.scaler")
    p.add_argument("--modification_json", default=None)
    p.add_argument("--modified_base_symbol", default="1")
    p.add_argument("--modified_base_replacement", default="U")
    p.add_argument("--out_dir", default="scenario_outputs_struct_besthp")
    p.add_argument("--n_results", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_initial_samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_steps", type=int, default=60)
    p.add_argument("--candidates_per_step", type=int, default=12)
    p.add_argument("--beam_size", type=int, default=3)
    p.add_argument("--no_final_refine", action="store_true")
    p.add_argument("--predict_batch_size", type=int, default=256)
    p.add_argument("--baseline_seed_source", choices=["training_random", "hybrid_initial"], default="training_random")
    p.add_argument("--deterministic_baseline_seeds", action="store_true", help="Cycle through training seeds deterministically instead of random choice.")
    p.add_argument("--flush_every", type=int, default=25)
    p.add_argument("--methods", default="hybrid_struct,pareto_struct,smart5utr_baseline", help="Comma-separated subset: hybrid_struct,pareto_struct,smart5utr_baseline")
    p.add_argument("--target_rl", type=float, default=TARGET_RL_DEFAULT, help="Target RL for real control-element scenarios.")
    p.add_argument("--accessibility_threshold", type=float, default=0.60)
    p.add_argument("--cds_start_context", default=DEFAULT_CDS_START_CONTEXT)
    p.add_argument("--pareto_rounds", type=int, default=25)
    p.add_argument("--pareto_population_size", type=int, default=24)
    p.add_argument("--pareto_elite_size", type=int, default=12)
    p.add_argument("--pareto_offspring_per_elite", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "list_scenarios":
        scenarios = build_scenarios(args.scenario_set)
        print(f"scenario_set={args.scenario_set}; n_scenarios={len(scenarios)}")
        for i, (name, targets) in enumerate(scenarios):
            print(f"{i:02d}\t{name}\t{asdict(targets)}")
        return
    if args.mode == "run_scenario":
        run_scenario(args)
        return
    if args.mode == "run_all_serial":
        scenarios = build_scenarios(args.scenario_set)
        for i in range(len(scenarios)):
            args.scenario_id = i
            run_scenario(args)
        aggregate(args)
        return
    if args.mode == "aggregate":
        aggregate(args)
        return


if __name__ == "__main__":
    main()
