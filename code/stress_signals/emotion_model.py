"""Step 4: multi-label emotion model on GoEmotions (training is RUN BY THE USER on the GPU of whichever machine runs the
notebook, else on the CPU; see utils.select_device).

Pipeline per seed (resume-safe; every stage writes to durable storage and is skipped if its output exists):
    official GoEmotions splits (fixed split file) -> 28-column label matrix -> fine-tune with HF Trainer (one sigmoid per
    label, BCEWithLogits, checkpoint every epoch, early stopping on validation macro AP, best model reloaded) ->
    best_model/ -> logits for validation + test -> DONE.json
Evaluation, per-label calibration, per-label thresholds, the Ekman collapse and the Dreaddit domain-shift check all read
saved logits or run inference only. Final bundle: <paths.models>/emotion/v<YYYYMMDD>/.

Run layout: trained_models/emotion/runs/<model_choice>/seed<k>/{best_model/, train_log.csv, train_summary.json,
            predictions_validation.parquet, predictions_test.parquet, DONE.json, calibration.json, thresholds.json,
            FINAL_TEST_REPORT.json, predictions_dreaddit.parquet, domain_shift_dreaddit.json}
Checkpoints: <checkpoint_root>/emotion/<model_choice>/seed<k>/ (never inside trained_models: the Trainer rotates them).

Scope note: GoEmotions labels say which emotions annotators read in a short Reddit COMMENT. They are not stress labels,
and a negative emotion is not stress. Outputs feed aggregate signals only.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import stress_model as SM
from .metrics import (MULTILABEL_KEYS, _prf, apply_label_calibration, bootstrap_multilabel, ece, fast_roc_auc,
                      fit_label_calibration, multilabel_ece, multilabel_metrics, plot_reliability, sigmoid,
                      tune_label_thresholds)
from .utils import (checkpoint_root, device_label, get_logger, model_version_dir, models_root, select_device, set_seed,
                    sha256_file, write_json, write_manifest)

LOG = get_logger("stress_signals.emotion_model")
TRAINING_BANNER = SM.TRAINING_BANNER
COMPONENT = "emotion"
SPLITS = ("train", "validation", "test")
DOMAIN_SHIFT_NOTE = (
    "DOMAIN SHIFT: GoEmotions is short Reddit COMMENTS (median ~18 roberta tokens, 2019, mixed subreddits); Dreaddit is "
    "longer POST segments (median ~99 tokens, 2017-2018, stress-prone subreddits). Probabilities calibrated on GoEmotions "
    "need not be calibrated on Dreaddit. NOT STRESS: a negative emotion is not the same as stress, and these numbers say "
    "nothing about any person; they describe language in sampled text only.")


# ================================================================ settings / paths
def settings(cfg: dict[str, Any], model_choice: str | None = None) -> dict[str, Any]:
    """emotion_model config with the chosen model resolved (`hf_id`)."""
    s = copy.deepcopy(cfg["emotion_model"])
    choice = model_choice or s["model_choice"]
    if choice not in s["model_options"]:
        raise ValueError(f"model_choice {choice!r} not in {list(s['model_options'])}")
    s["model_choice"], s["hf_id"] = choice, s["model_options"][choice]["hf_id"]
    return s


def runs_root(cfg: dict[str, Any]) -> Path:
    return models_root(cfg) / COMPONENT / "runs"


def run_dir_for(cfg: dict[str, Any], model_choice: str, seed: int) -> Path:
    return runs_root(cfg) / model_choice / f"seed{seed}"


def checkpoint_dir_for(cfg: dict[str, Any], run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    return checkpoint_root(cfg) / COMPONENT / run_dir.parent.name / run_dir.name


# ================================================================ data (4A)
def label_names(cfg: dict[str, Any]) -> list[str]:
    """The 28 label names in id order, from the fixed split file (in git), cross-checked against the official
    emotions.txt when it is on this machine and against the configured label count."""
    from .preprocess import split_path
    names = json.loads(split_path(cfg, "goemotions").read_text(encoding="utf-8"))["label_names"]
    src = cfg["sources"]["goemotions"]
    if len(names) != src["num_labels"] or len(names) != cfg["models"]["emotion"]["num_labels"]:
        raise ValueError(f"{len(names)} label names, config expects {src['num_labels']}")
    if len(set(names)) != len(names) or names[-1] != "neutral":
        raise ValueError("label names must be unique and end with 'neutral' (official order)")
    off = Path(cfg["_root"]) / src["official_dir"] / "emotions.txt"
    if off.is_file():
        official = [l.strip() for l in off.read_text(encoding="utf-8").splitlines() if l.strip()]
        if official != names:
            raise ValueError("split-file label names differ from the official emotions.txt")
    return list(names)


def label_matrix(label_lists: Sequence[Sequence[int]], n_labels: int) -> np.ndarray:
    """(n, n_labels) int8 multi-hot matrix. Every row needs >= 1 label id in [0, n_labels)."""
    m = np.zeros((len(label_lists), n_labels), dtype=np.int8)
    for i, ls in enumerate(label_lists):
        ids = [int(x) for x in ls]
        if not ids:
            raise ValueError(f"row {i} has no label")
        if min(ids) < 0 or max(ids) >= n_labels:
            raise ValueError(f"row {i} has a label id outside [0, {n_labels}): {ids}")
        m[i, ids] = 1
    return m


def resolve_source(cfg: dict[str, Any]) -> str:
    """'processed_official' (data/processed/goemotions.parquet, built in Step 2 from the official TSVs) if present,
    else 'hf_fallback' (google-research-datasets/go_emotions, simplified; Step 1 found it identical to the official
    files). The Kaggle copy was dropped by user decision in Step 1 (its contents were never verified)."""
    src = cfg["sources"]["goemotions"]
    if src.get("use_kaggle"):
        raise ValueError("sources.goemotions.use_kaggle is true, but the Kaggle copy is UNVERIFIED (Step 1 could not "
                         "download it and the user dropped it). Audit it with Step 1 before using it.")
    from .preprocess import processed_path
    return "processed_official" if processed_path(cfg, "goemotions").exists() else "hf_fallback"


def build_from_hf(cfg: dict[str, Any]) -> pd.DataFrame:
    """The same frame as the processed parquet, rebuilt from the HF fallback (record ids from the native comment id,
    text cleaned with the Step 2 cleaner). verify_split() then proves the ids match the fixed split file."""
    from datasets import load_dataset
    from .preprocess import clean_text, stable_record_id
    src = cfg["sources"]["goemotions"]
    ds = load_dataset(src["hf_fallback_id"], src["hf_fallback_config"])
    hf_names = ds["train"].features["labels"].feature.names
    if hf_names != label_names(cfg):
        raise ValueError("HF label names differ from the split-file label names")
    parts = []
    for split in SPLITS:
        d = ds[split].to_pandas()
        parts.append(pd.DataFrame({"record_id": d["id"].map(lambda i: stable_record_id("goemotions", i)),
                                   "text_clean": d["text"].map(lambda t: clean_text(t, cfg)),
                                   "labels": d["labels"].map(lambda a: [int(x) for x in a]),
                                   "split": split}))
    return pd.concat(parts, ignore_index=True)


def verify_split(cfg: dict[str, Any], df: pd.DataFrame) -> dict[str, int]:
    """Per-split record ids must equal the saved split file exactly (never regenerated here)."""
    from .preprocess import split_path
    saved = json.loads(split_path(cfg, "goemotions").read_text(encoding="utf-8"))["splits"]
    for name, ids in saved.items():
        if set(df.loc[df["split"] == name, "record_id"]) != set(ids):
            raise ValueError(f"GoEmotions split {name!r} differs from {split_path(cfg, 'goemotions').name}")
    return {k: len(v) for k, v in saved.items()}


def load_goemotions(cfg: dict[str, Any], source: str | None = None) -> dict[str, pd.DataFrame]:
    """train / validation / test frames (record_id, text_clean, labels, split); split file verified, labels validated."""
    source = source or resolve_source(cfg)
    if source == "processed_official":
        from .preprocess import processed_path
        df = pd.read_parquet(processed_path(cfg, "goemotions"), columns=["record_id", "text_clean", "labels", "split"])
        df["labels"] = df["labels"].map(lambda a: [int(x) for x in a])
    elif source == "hf_fallback":
        df = build_from_hf(cfg)
    else:
        raise ValueError(f"unknown GoEmotions source {source!r}")
    verify_split(cfg, df)
    label_matrix(df["labels"], len(label_names(cfg)))          # raises on empty rows / out-of-range ids
    LOG.info("GoEmotions loaded from %s", source)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}


def label_summary(data: dict[str, pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    """Per-label positive counts and shares per split (aggregate only)."""
    rows = {}
    for split, df in data.items():
        Y = label_matrix(df["labels"], len(names))
        rows[(split, "n_pos")] = Y.sum(0)
        rows[(split, "share")] = Y.mean(0).round(4)
    return pd.DataFrame(rows, index=list(names))


def structure_summary(data: dict[str, pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    """Rows, multi-label rate and labels-per-row distribution per split."""
    out = []
    for split, df in data.items():
        k = label_matrix(df["labels"], len(names)).sum(1)
        out.append({"split": split, "n": len(df), "multi_label_rate": round(float((k > 1).mean()), 4),
                    **{f"rows_with_{j}_labels": int((k == j).sum()) for j in range(1, int(k.max()) + 1)}})
    return pd.DataFrame(out)


# ================================================================ training pieces (4B)
def pos_weights(Y: np.ndarray, method: str = "inverse_sqrt", max_weight: float = 10.0) -> np.ndarray:
    """Optional BCE pos_weight per label from TRAIN counts. inverse_sqrt: sqrt(n_neg / n_pos), clipped to
    [1, max_weight]. OFF by default: it shifts every probability upward (worse calibration) while per-label thresholds
    are tuned on validation anyway, so the main use is a sensitivity run for rare emotions."""
    Y = np.asarray(Y)
    npos = Y.sum(0).astype(float)
    nneg = len(Y) - npos
    if np.any(npos == 0):
        raise ValueError("a label has no positive training example; pos_weight is undefined")
    if method == "inverse_sqrt":
        w = np.sqrt(nneg / npos)
    elif method == "inverse":
        w = nneg / npos
    else:
        raise ValueError(f"unknown class weighting method {method!r}")
    return np.clip(w, 1.0, float(max_weight))


def bce_loss(logits, labels, pos_weight=None):
    """Mean BCEWithLogits over all (row, label) elements, computed in fp32. Equals the loss transformers uses for
    problem_type='multi_label_classification' when pos_weight is None (checked in tests)."""
    import torch.nn.functional as F
    pw = None if pos_weight is None else pos_weight.to(device=logits.device, dtype=logits.float().dtype)
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float(), pos_weight=pw)


class MultiLabelDataset:
    """Pre-tokenised examples with a float multi-hot label row (dynamic padding happens in the collator)."""

    def __init__(self, encodings: dict[str, list], Y: np.ndarray):
        self.enc, self.Y = encodings, np.asarray(Y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.Y[i]
        return item


class MultiLabelCollator:
    """Pads token fields with the tokenizer and stacks `labels` as a float (batch, n_labels) tensor."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        tok = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tokenizer.pad(tok, return_tensors="pt")
        batch["labels"] = torch.tensor(np.stack([f["labels"] for f in features]), dtype=torch.float)
        return batch


@lru_cache(maxsize=1)
def trainer_divides_loss_by_ga() -> bool:
    """Does the INSTALLED Trainer divide a model-computed loss by gradient_accumulation_steps before backward()?
    Measured, not assumed: a tiny random 1-layer model (CPU, global RNG untouched), one micro-batch through
    Trainer.training_step with GA=1 and GA=2, compare gradient norms. No optimizer step, so no weight changes.
    (transformers 4.46.3 + RoBERTa: it does NOT divide, so the emotion loss divides itself; see EmotionTrainer.)"""
    import tempfile
    import torch
    import transformers as tr
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        conf = tr.RobertaConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=32, max_position_embeddings=24, num_labels=3,
                                problem_type="multi_label_classification", pad_token_id=1)
        model = tr.AutoModelForSequenceClassification.from_config(conf)
        inputs = {"input_ids": torch.randint(3, 64, (4, 8)), "attention_mask": torch.ones(4, 8, dtype=torch.long),
                  "labels": torch.randint(0, 2, (4, 3)).float()}
        norms = {}
        for ga in (1, 2):
            model.zero_grad(set_to_none=True)
            with tempfile.TemporaryDirectory() as tmp:
                args = tr.TrainingArguments(output_dir=tmp, gradient_accumulation_steps=ga, report_to="none", use_cpu=True)
                tr.Trainer(model=model, args=args).training_step(model, dict(inputs), None)
            norms[ga] = float(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm())
    ratio = norms[2] / norms[1]
    if not (np.isclose(ratio, 1.0, rtol=1e-3) or np.isclose(ratio, 0.5, rtol=1e-3)):
        raise RuntimeError(f"unexpected gradient ratio {ratio:.4f} for GA=2 vs GA=1; check the installed transformers")
    return bool(np.isclose(ratio, 0.5, rtol=1e-3))


def _trainer_classes():
    """EmotionTrainer + CSV logger, defined lazily so importing this module never requires transformers."""
    from transformers import Trainer
    _, CSVLogger = SM._trainer_classes()

    class EmotionTrainer(Trainer):
        """BCEWithLogits (optional pos_weight). If the installed Trainer does not scale the loss for gradient
        accumulation, the loss is divided by GA here during training, so every effective batch averages its gradient
        (same update as one big batch). Evaluation loss is never scaled."""

        def __init__(self, *args, pos_weight=None, ga_divide: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            import torch
            self._pos_weight = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float)
            self._ga_divide = bool(ga_divide)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # kwargs: num_items_in_batch (>=4.46)
            inputs = dict(inputs)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = bce_loss(outputs.logits, labels, self._pos_weight)
            if self._ga_divide and model.training and self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            return (loss, outputs) if return_outputs else loss

    class ScaledCSVLogger(CSVLogger):
        """CSV log whose training `loss` is multiplied back by `loss_scale` (undoes the GA division above in the log)."""

        def __init__(self, path: Path, loss_scale: float = 1.0):
            super().__init__(path)
            self.loss_scale = float(loss_scale)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and self.loss_scale != 1.0:
                logs = {**logs, "loss": logs["loss"] * self.loss_scale}
            return super().on_log(args, state, control, logs=logs, **kwargs)

    return EmotionTrainer, ScaledCSVLogger


def trainer_compute_metrics(eval_pred) -> dict[str, float]:
    """Validation metrics during training (uncalibrated, threshold 0.5 for F1). Trainer adds the `eval_` prefix.
    `macro_ap` (threshold-free) selects the best epoch (config emotion_model.metric_for_best_model)."""
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    m = multilabel_metrics(np.asarray(eval_pred.label_ids).round().astype(int), sigmoid(logits), 0.5)
    return {"macro_ap": m["ap_macro"], "f1_macro": m["f1_macro"], "f1_micro": m["f1_micro"],
            "roc_auc_macro": m["roc_auc_macro"]}


def _save_predictions(df: pd.DataFrame, Y: np.ndarray, Z: np.ndarray, names: Sequence[str], path: Path) -> None:
    """record_id + split + true labels (y__<name>) + logits (logit__<name>). No text."""
    out = pd.concat([df[["record_id", "split"]].reset_index(drop=True),
                     pd.DataFrame(np.asarray(Y, dtype=np.int8), columns=[f"y__{n}" for n in names]),
                     pd.DataFrame(np.asarray(Z, dtype=np.float32), columns=[f"logit__{n}" for n in names])], axis=1)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_predictions(run_dir: Path, split: str, names: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(ids frame, Y (n, L), logits Z (n, L)) from a run's saved predictions."""
    df = pd.read_parquet(Path(run_dir) / f"predictions_{split}.parquet")
    return (df[["record_id", "split"]], df[[f"y__{n}" for n in names]].to_numpy().astype(int),
            df[[f"logit__{n}" for n in names]].to_numpy(dtype=float))


# ================================================================ training (USER RUNS)
def train_emotion_seed(cfg: dict[str, Any], seed: int, model_choice: str | None = None,
                       data: dict[str, pd.DataFrame] | None = None, run_dir: Path | None = None,
                       epochs: int | None = None) -> Path:
    """Fine-tune one seed. Resume-safe: (1) DONE.json -> skip; (2) best_model/ exists -> only (re)write missing
    predictions; (3) otherwise resume from the last epoch checkpoint if any."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    s = settings(cfg, model_choice)
    if epochs is not None:
        s["epochs"] = int(epochs)
    run_dir = Path(run_dir or run_dir_for(cfg, s["model_choice"], seed))
    if (run_dir / "DONE.json").exists():
        LOG.info("seed %s: DONE.json present in %s, skipping", seed, run_dir)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    data = data or load_goemotions(cfg)
    names = label_names(cfg)
    id2label = dict(enumerate(names))
    ga_divide = not trainer_divides_loss_by_ga()          # measured before seeding (it does not touch the global RNG)
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    best_dir = run_dir / "best_model"
    tok = AutoTokenizer.from_pretrained(s["hf_id"], use_fast=True)
    cuda = torch.cuda.is_available()

    if not (best_dir / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            s["hf_id"], num_labels=len(names), problem_type=s["problem_type"], id2label=id2label,
            label2id={v: k for k, v in id2label.items()})
        tr, va = data["train"], data["validation"]
        Ytr, Yva = label_matrix(tr["labels"], len(names)), label_matrix(va["labels"], len(names))
        cw = s["class_weighting"]
        pw = pos_weights(Ytr, cw["method"], cw["max_pos_weight"]) if cw["enabled"] else None
        ds_tr = MultiLabelDataset(SM.encode(tok, tr["text_clean"], s["max_length"]), Ytr)
        ds_va = MultiLabelDataset(SM.encode(tok, va["text_clean"], s["max_length"]), Yva)
        bf16_ok = bool(cuda and torch.cuda.is_bf16_supported())
        ck = checkpoint_dir_for(cfg, run_dir)
        args = SM.build_training_args(s, ck, seed, cuda, bf16_ok)
        EmotionTrainer, ScaledCSVLogger = _trainer_classes()
        compat = SM.hf_compat()
        ga = s["gradient_accumulation_steps"]
        trainer = EmotionTrainer(
            model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, data_collator=MultiLabelCollator(tok),
            compute_metrics=trainer_compute_metrics, pos_weight=pw, ga_divide=ga_divide,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=s["early_stopping_patience"]),
                       ScaledCSVLogger(run_dir / "train_log.csv", ga if (ga_divide and ga > 1) else 1.0)],
            **{compat["tokenizer_key"]: tok})
        last = get_last_checkpoint(str(ck)) if ck.is_dir() else None
        LOG.info("seed %s: training %s (%s) on %s, batch %s x GA %s, ga_divide=%s, resume_from=%s", seed,
                 s["model_choice"], s["hf_id"], device_label(), s["train_batch_size"], ga, ga_divide, last)
        result = trainer.train(resume_from_checkpoint=last)
        trainer.save_model(str(best_dir))            # load_best_model_at_end=True -> this is the best epoch
        tok.save_pretrained(str(best_dir))
        write_json(run_dir / "train_summary.json", {
            "seed": seed, "model_choice": s["model_choice"], "hf_id": s["hf_id"], "settings": s, "label_names": names,
            "best_metric_validation_macro_ap": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint, "epochs_completed": trainer.state.epoch,
            "global_step": trainer.state.global_step, "train_runtime_s": result.metrics.get("train_runtime"),
            "checkpoint_dir": ck, "log_history": trainer.state.log_history, "transformers": compat["transformers"],
            "gradient_accumulation": {"steps": ga, "loss_divided_by_ga_in_compute_loss": ga_divide and ga > 1,
                                      "note": "log_history `loss` is the Trainer's report (divided by GA once more when "
                                              "ga_divide); train_log.csv has it rescaled to the true mean loss"},
            "pos_weight": None if pw is None else dict(zip(names, map(float, pw))),
            "n_train": len(tr), "n_validation": len(va), "device": device_label()})
        del trainer, model
        if cuda:
            torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
    device = select_device(cfg)
    for split in ("validation", "test"):
        out = run_dir / f"predictions_{split}.parquet"
        if out.exists() or split not in data:
            continue
        df = data[split]
        Z = SM.predict_logits(model, tok, df["text_clean"].tolist(), s["max_length"], s["eval_batch_size"], device)
        _save_predictions(df, label_matrix(df["labels"], len(names)), Z, names, out)
    write_json(run_dir / "DONE.json", {"seed": seed, "model_choice": s["model_choice"],
                                       "finished_utc": _dt.datetime.now(_dt.timezone.utc)})
    return run_dir


# ================================================================ evaluation (4C, reads saved logits)
def completed_seeds(cfg: dict[str, Any], model_choice: str | None = None) -> dict[int, Path]:
    s = settings(cfg, model_choice)
    return {k: run_dir_for(cfg, s["model_choice"], k) for k in s["seeds"]
            if (run_dir_for(cfg, s["model_choice"], k) / "DONE.json").exists()}


def readiness(cfg: dict[str, Any], model_choice: str | None = None) -> dict[str, Any]:
    """Evaluation waits for ALL seeds (the final seed and the once-only test report must not come from a partial set)."""
    s = settings(cfg, model_choice)
    done = completed_seeds(cfg, model_choice)
    missing = [k for k in s["seeds"] if k not in done]
    return {"seeds_done": sorted(done), "seeds_missing": missing, "all_seeds": not missing,
            "why_seeds": f"needs every emotion seed of {s['model_choice']} {s['seeds']}; missing {missing}. "
                         "They train in 4B.1 (an interrupted seed resumes there on the next Run All)."}


def _summary_only(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if k != "per_label"}


def evaluate_runs(cfg: dict[str, Any], model_choice: str | None = None) -> dict[str, Any]:
    """Uncalibrated metrics at threshold 0.5 on validation and test for every completed seed (point values, per-label
    tables) and mean +/- std across seeds. Bootstrap CIs are computed once, for the final model, in final_test_report."""
    s = settings(cfg, model_choice)
    names = label_names(cfg)
    seeds = completed_seeds(cfg, model_choice)
    if not seeds:
        raise FileNotFoundError(f"no completed emotion runs under {runs_root(cfg) / s['model_choice']}")
    res: dict[str, Any] = {"model_choice": s["model_choice"], "threshold": 0.5, "calibrated": False, "seeds": {},
                           "summary": {}}
    for seed, d in seeds.items():
        res["seeds"][seed] = {}
        for split in ("validation", "test"):
            _, Y, Z = load_predictions(d, split, names)
            res["seeds"][seed][split] = multilabel_metrics(Y, sigmoid(Z), 0.5, names)
    for split in ("validation", "test"):
        rows = pd.DataFrame([{k: res["seeds"][sd][split][k] for k in MULTILABEL_KEYS} for sd in res["seeds"]])
        res["summary"][split] = {k: {"mean": float(rows[k].mean()), "std": float(rows[k].std(ddof=1)) if len(rows) > 1 else 0.0,
                                     "n_runs": int(len(rows))} for k in MULTILABEL_KEYS}
    write_json(runs_root(cfg) / s["model_choice"] / "evaluation_uncalibrated.json", res)
    return res


def select_final_seed(evaluation: dict[str, Any]) -> int:
    """Deployable model = seed with the best VALIDATION macro AP (threshold-free; test never used for selection)."""
    return int(max(evaluation["seeds"], key=lambda k: evaluation["seeds"][k]["validation"]["ap_macro"]))


def per_label_table(m: dict[str, Any], ci: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-label metrics as a table (optionally with bootstrap CI columns for F1 and ROC-AUC)."""
    t = pd.DataFrame(m["per_label"])
    if ci is not None:
        t["f1_lo"] = t["label"].map(lambda n: ci["per_label"][n]["f1"]["lo"])
        t["f1_hi"] = t["label"].map(lambda n: ci["per_label"][n]["f1"]["hi"])
        t["roc_auc_lo"] = t["label"].map(lambda n: ci["per_label"][n]["roc_auc"]["lo"])
        t["roc_auc_hi"] = t["label"].map(lambda n: ci["per_label"][n]["roc_auc"]["hi"])
    return t


# ================================================================ calibration (4D) and thresholds (4C)
def calibrate(cfg: dict[str, Any], seed: int, model_choice: str | None = None) -> dict[str, Any]:
    """Per-label calibration fitted on VALIDATION logits only (config emotion_model.calibration.method).

    Why Platt (default) over temperature: dashboard aggregates average PROBABILITIES over many texts, so a label whose
    probabilities are shifted (systematically too low/high, common for rare labels under BCE) biases the aggregate.
    A per-label temperature only rescales the logit and keeps p = 0.5 at z = 0, so it cannot remove such an offset;
    Platt's a*z + b can. a > 0 is enforced (ranking, AUC and AP unchanged) and Platt's smoothed targets keep the
    2-parameter fit stable for labels with few validation positives (grief: 13). Temperature is a config switch."""
    s = settings(cfg, model_choice)
    names = label_names(cfg)
    d = run_dir_for(cfg, s["model_choice"], seed)
    _, Y, Z = load_predictions(d, "validation", names)
    c = s["calibration"]
    params = fit_label_calibration(Z, Y, c["method"], names)
    P0, P1 = sigmoid(Z), apply_label_calibration(Z, params)
    e0, e1 = multilabel_ece(Y, P0, c["ece_bins"]), multilabel_ece(Y, P1, c["ece_bins"])
    per = pd.DataFrame(params)
    per["n_pos_validation"] = Y.sum(0)
    per["ece_before"], per["ece_after"] = e0["per_label"], e1["per_label"]
    per["mean_p_before"], per["mean_p_after"], per["observed_rate"] = P0.mean(0), P1.mean(0), Y.mean(0)
    out = {"seed": seed, "model_choice": s["model_choice"], "fitted_on": "validation", "n_validation": int(len(Y)),
           "method": c["method"], "formula": "p = sigmoid(a * logit + b), per label", "params": params,
           "per_label": per.to_dict(orient="records"), "ece_bins": c["ece_bins"],
           "validation_ece_before": {k: v for k, v in e0.items() if k != "per_label"},
           "validation_ece_after": {k: v for k, v in e1.items() if k != "per_label"},
           "caveat": "fitted and evaluated on the same validation set; the test ECE in FINAL_TEST_REPORT.json is the honest check"}
    write_json(d / "calibration.json", out)
    plot_reliability({"uncalibrated (all labels pooled)": (Y.ravel(), P0.ravel()),
                      f"{c['method']} per label (pooled)": (Y.ravel(), P1.ravel())},
                     c["ece_bins"], f"Emotion reliability on validation: seed {seed}", path=str(d / "reliability_validation.png"))
    return out


def tune_thresholds(cfg: dict[str, Any], seed: int, model_choice: str | None = None) -> dict[str, Any]:
    """Per-label thresholds on VALIDATION calibrated probabilities (max F1 per label, ties -> closest to 0.5).
    Labels with fewer than eval.low_support_val_positives validation positives are flagged `low_support`; what is
    APPLIED for them follows thresholds.low_support_rule (tune | keep_0.5). Writes thresholds.json + the F1 curve."""
    s = settings(cfg, model_choice)
    names = label_names(cfg)
    d = run_dir_for(cfg, s["model_choice"], seed)
    cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    _, Y, Z = load_predictions(d, "validation", names)
    P = apply_label_calibration(Z, cal["params"])
    tab, curve = tune_label_thresholds(Y, P, s["thresholds"]["grid"], names)
    min_pos, rule = s["eval"]["low_support_val_positives"], s["thresholds"]["low_support_rule"]
    if rule not in ("tune", "keep_0.5"):
        raise ValueError(f"unknown thresholds.low_support_rule {rule!r}")
    tab["low_support"] = tab["n_pos"] < min_pos
    tab["applied"] = np.where(tab["low_support"] & (rule == "keep_0.5"), 0.5, tab["threshold"])
    applied = tab["applied"].to_numpy()
    out = {"seed": seed, "model_choice": s["model_choice"], "fitted_on": "validation (calibrated probabilities)",
           "criterion": "max F1 per label; ties -> closest to 0.5", "grid": s["thresholds"]["grid"],
           "low_support_val_positives": min_pos, "low_support_rule": rule,
           "thresholds": dict(zip(names, map(float, applied))),
           "per_label": tab.to_dict(orient="records"),
           "validation_at_0.5": _summary_only(multilabel_metrics(Y, P, 0.5, names)),
           "validation_at_applied": _summary_only(multilabel_metrics(Y, P, applied, names)),
           "caveat": "validation gains are optimistic: the thresholds were chosen on the same data"}
    write_json(d / "thresholds.json", out)
    curve.to_csv(d / "threshold_curves_validation.csv", index=False)
    return out


# ================================================================ Ekman collapse (4E)
def load_ekman_mapping(cfg: dict[str, Any], names: Sequence[str] | None = None) -> dict[str, list[str]]:
    """Official GoEmotions ekman_mapping.json (+ neutral as its own group if configured). Validated: every label
    except neutral is in exactly one group and every mapped name is a real label."""
    names = list(names or label_names(cfg))
    e = cfg["emotion_model"]["ekman"]
    mapping = json.loads((Path(cfg["_root"]) / e["mapping_file"]).read_text(encoding="utf-8"))
    seen = [m for ms in mapping.values() for m in ms]
    unknown = sorted(set(seen) - set(names))
    dup = sorted({m for m in seen if seen.count(m) > 1})
    missing = sorted(set(names) - set(seen) - {"neutral"})
    if unknown or dup or missing:
        raise ValueError(f"Ekman mapping invalid: unknown {unknown}, in two groups {dup}, unmapped {missing}")
    mapping = {g: list(ms) for g, ms in mapping.items()}
    if e.get("keep_neutral", True):
        mapping["neutral"] = ["neutral"]
    return mapping


def ekman_members(mapping: dict[str, list[str]], names: Sequence[str]) -> list[list[int]]:
    idx = {n: i for i, n in enumerate(names)}
    return [[idx[m] for m in ms] for ms in mapping.values()]


def ekman_probs(P: np.ndarray, members: Sequence[Sequence[int]], combine: str = "max") -> np.ndarray:
    """Group probability from member CALIBRATED probabilities.
    max (default): P(group) >= max_i p_i holds for ANY dependence between members (Frechet lower bound), so `max` is a
      conservative estimate that does not grow with group size; GoEmotions members co-occur strongly (anger+annoyance:
      269 train rows), which is exactly when independence-based rules overcount.
    noisy_or: 1 - prod_i(1 - p_i), exact only if members are independent; inflates large groups (joy has 12 members).
    Neither is calibrated at group level; FINAL_TEST_REPORT.json reports the group-level ECE of the chosen rule."""
    P = np.asarray(P, dtype=float)
    if combine == "max":
        return np.stack([P[:, m].max(1) for m in members], 1)
    if combine == "noisy_or":
        return np.stack([1 - np.prod(1 - P[:, m], 1) for m in members], 1)
    raise ValueError(f"unknown Ekman combine rule {combine!r}")


def ekman_any(B: np.ndarray, members: Sequence[Sequence[int]]) -> np.ndarray:
    """Group is on if ANY member is on (used for gold labels and for thresholded predictions)."""
    B = np.asarray(B).astype(bool)
    return np.stack([B[:, m].any(1) for m in members], 1)


def ekman_rule_text(combine: str) -> str:
    return {"max": "group probability = max of member calibrated probabilities (lower bound of P(any member), "
                   "no independence assumption); group predicted = any member above its own threshold; "
                   "group gold = any member labelled",
            "noisy_or": "group probability = 1 - prod(1 - member calibrated probabilities) (assumes independent "
                        "members); group predicted = any member above its own threshold; group gold = any member labelled"}[combine]


def ekman_metrics(Yg: np.ndarray, Pg: np.ndarray, Hg: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    """Per-group P/R/F1 (predictions = OR of member decisions), ROC-AUC of the group probability, macro/micro F1."""
    Yg, Hg = np.asarray(Yg).astype(bool), np.asarray(Hg).astype(bool)
    tp, fp, fn = (Yg & Hg).sum(0), (~Yg & Hg).sum(0), (Yg & ~Hg).sum(0)
    prec, rec, f1 = _prf(tp, fp, fn)
    auc = fast_roc_auc(Yg, Pg)
    return {"f1_macro": float(f1.mean()), "f1_micro": float(_prf(tp.sum(), fp.sum(), fn.sum())[2]),
            "roc_auc_macro": float(np.nanmean(auc)),
            "per_group": [{"group": g, "support": int(Yg[:, j].sum()), "precision": float(prec[j]), "recall": float(rec[j]),
                           "f1": float(f1[j]), "roc_auc": float(auc[j]), "predicted_positive_rate": float(Hg[:, j].mean()),
                           "mean_group_probability": float(np.mean(Pg[:, j])), "observed_rate": float(Yg[:, j].mean())}
                          for j, g in enumerate(groups)]}


# ================================================================ final test report (once)
def final_test_report(cfg: dict[str, Any], seed: int, model_choice: str | None = None, force: bool = False) -> dict[str, Any]:
    """Calibrated test metrics with the VALIDATION-tuned per-label thresholds, computed ONCE (FINAL_TEST_REPORT.json is
    returned unchanged afterwards). Includes bootstrap CIs, the same metrics at 0.5, uncalibrated 0.5, test ECE before
    and after calibration, a sensitivity check without test rows whose text also occurs in train/validation (Step 2
    flags), and the Ekman-level results."""
    s = settings(cfg, model_choice)
    names = label_names(cfg)
    d = run_dir_for(cfg, s["model_choice"], seed)
    path = d / "FINAL_TEST_REPORT.json"
    if path.exists() and not force:
        LOG.info("final emotion test report already exists (reported once): %s", path)
        return json.loads(path.read_text(encoding="utf-8"))
    cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
    t = np.array([thr["thresholds"][n] for n in names])
    ids, Y, Z = load_predictions(d, "test", names)
    P0, P = sigmoid(Z), apply_label_calibration(Z, cal["params"])
    ev = s["eval"]
    main = multilabel_metrics(Y, P, t, names)
    rep: dict[str, Any] = {
        "seed": seed, "model_choice": s["model_choice"], "calibration_method": cal["method"],
        "thresholds_fitted_on": "validation", "low_support_rule": thr["low_support_rule"],
        "test": main,
        "test_ci": bootstrap_multilabel(Y, P, t, ev["bootstrap_resamples"], cfg["primary_seed"], ev["ci_level"], names),
        "test_calibrated_at_0.5": multilabel_metrics(Y, P, 0.5, names),
        "test_uncalibrated_at_0.5": _summary_only(multilabel_metrics(Y, P0, 0.5, names)),
        "test_ece": {"uncalibrated": multilabel_ece(Y, P0, s["calibration"]["ece_bins"]),
                     "calibrated": multilabel_ece(Y, P, s["calibration"]["ece_bins"])},
    }
    from .preprocess import goemotions_overlap_path
    fp = goemotions_overlap_path(cfg)
    if fp.exists():
        flags = pd.read_parquet(fp)
        excl = set(flags.loc[flags["exclude_for_test_sensitivity"], "record_id"])
        keep = ~ids["record_id"].isin(excl).to_numpy()
        rep["test_sensitivity_without_cross_split_overlap"] = {
            "n_excluded": int((~keep).sum()), **_summary_only(multilabel_metrics(Y[keep], P[keep], t, names))}
    mapping = load_ekman_mapping(cfg, names)
    mem, groups, comb = ekman_members(mapping, names), list(mapping), s["ekman"]["combine"]
    Yg, Pg, Hg = ekman_any(Y, mem), ekman_probs(P, mem, comb), ekman_any(P >= t, mem)
    rep["ekman"] = {"rule": ekman_rule_text(comb), "groups": groups, "test": ekman_metrics(Yg, Pg, Hg, groups),
                    "test_group_ece": dict(zip(groups, (ece(Yg[:, j], Pg[:, j], s["calibration"]["ece_bins"])
                                                        for j in range(len(groups)))))}
    rep["created_utc"] = _dt.datetime.now(_dt.timezone.utc)
    write_json(path, rep)
    return json.loads(path.read_text(encoding="utf-8"))


# ================================================================ domain shift on Dreaddit (4F, inference only)
def predict_chunked(model, tokenizer, texts: Sequence[str], max_length: int, stride: int, max_chunks: int,
                    params: Sequence[dict[str, Any]], aggregation: str = "mean", batch_size: int = 16,
                    device: str = "cpu", half: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated per-text probabilities (n, L): sentence-aligned chunks of <= max_length tokens (preprocess.chunk_text),
    per-chunk calibrated probabilities, combined per text with preprocess.aggregate_chunk_scores. Returns
    (P, chunks used per text, full token count per text). Inference only."""
    from .preprocess import aggregate_chunk_scores, chunk_text
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    stride = min(int(stride), budget // 2)
    chunks = [chunk_text(t, tokenizer, max_length, stride, max_chunks) for t in texts]
    if any(len(c) == 0 for c in chunks):
        raise ValueError("a text produced no chunks (empty text); filter empty texts first")
    flat = [c.text for cs in chunks for c in cs]
    owner = np.repeat(np.arange(len(texts)), [len(cs) for cs in chunks])
    Pc = apply_label_calibration(SM.predict_logits(model, tokenizer, flat, max_length, batch_size, device, half), params)
    P = np.empty((len(texts), Pc.shape[1]))
    for i, cs in enumerate(chunks):
        rows = Pc[owner == i]
        w = [c.n_tokens for c in cs] if aggregation == "weighted_mean" else None
        P[i] = aggregate_chunk_scores(rows, aggregation, w)
    n_tok = np.array([len(tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts])
    return P, np.array([len(cs) for cs in chunks]), n_tok


def cluster_bootstrap_diff(V: np.ndarray, flag: np.ndarray, clusters: Sequence[Any], n_resamples: int = 1000,
                           seed: int = 42, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mean(V | flag) - mean(V | not flag) per column, with a percentile CI from resampling whole clusters (post_id:
    segments of one post are not independent). Returns (diff, lo, hi)."""
    V = np.asarray(V, dtype=float)
    f = np.asarray(flag).astype(bool)
    codes, _ = pd.factorize(pd.Series(np.asarray(clusters)).astype(str))
    C = codes.max() + 1
    S1, S0 = np.zeros((C, V.shape[1])), np.zeros((C, V.shape[1]))
    np.add.at(S1, codes[f], V[f])
    np.add.at(S0, codes[~f], V[~f])
    N1, N0 = np.bincount(codes[f], minlength=C).astype(float), np.bincount(codes[~f], minlength=C).astype(float)
    diff = S1.sum(0) / N1.sum() - S0.sum(0) / N0.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty((n_resamples, V.shape[1]))
    for b in range(n_resamples):
        w = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
        draws[b] = (w @ S1) / max(w @ N1, 1.0) - (w @ S0) / max(w @ N0, 1.0)
    a = (1 - level) / 2
    return diff, np.quantile(draws, a, axis=0), np.quantile(draws, 1 - a, axis=0)


def summarise_domain_shift(pred: pd.DataFrame, names: Sequence[str], mapping: dict[str, list[str]],
                           thresholds: Sequence[float], combine: str, k_min: int, max_length: int,
                           n_resamples: int = 1000, seed: int = 42, level: float = 0.95) -> dict[str, Any]:
    """Aggregate emotion distribution for Dreaddit stress (label 1) vs not-stress (label 0) segments: mean calibrated
    probability, share above the per-label threshold, difference with a post-clustered bootstrap CI, at label and Ekman
    level; per-community Ekman means with every cell of n < k_min suppressed. No text, no ids."""
    P = pred[[f"p__{n}" for n in names]].to_numpy(dtype=float)
    H = P >= np.asarray(thresholds, dtype=float)
    mem, groups = ekman_members(mapping, names), list(mapping)
    Pg, Hg = ekman_probs(P, mem, combine), ekman_any(H, mem)
    f = pred["label"].to_numpy() == 1

    def block(V: np.ndarray, B: np.ndarray, cols: Sequence[str]) -> list[dict[str, Any]]:
        diff, lo, hi = cluster_bootstrap_diff(V, f, pred["post_id"], n_resamples, seed, level)
        rdiff, rlo, rhi = cluster_bootstrap_diff(B.astype(float), f, pred["post_id"], n_resamples, seed, level)
        return [{"name": c, "mean_p_stress": float(V[f, j].mean()), "mean_p_not_stress": float(V[~f, j].mean()),
                 "diff_mean_p": float(diff[j]), "diff_lo": float(lo[j]), "diff_hi": float(hi[j]),
                 "rate_stress": float(B[f, j].mean()), "rate_not_stress": float(B[~f, j].mean()),
                 "diff_rate": float(rdiff[j]), "diff_rate_lo": float(rlo[j]), "diff_rate_hi": float(rhi[j])}
                for j, c in enumerate(cols)]

    by_comm = []
    for (comm, lab), g in pred.assign(_i=np.arange(len(pred))).groupby(["community", "label"]):
        row: dict[str, Any] = {"community": comm, "dreaddit_label": "stress" if lab == 1 else "not_stress", "n": int(len(g))}
        if len(g) < k_min:
            row["suppressed"] = True
        else:
            row["suppressed"] = False
            row.update({f"mean_p__{grp}": float(Pg[g["_i"].to_numpy(), j].mean()) for j, grp in enumerate(groups)})
        by_comm.append(row)
    return {"n": {"stress": int(f.sum()), "not_stress": int((~f).sum()), "posts": int(pred["post_id"].nunique())},
            "level": level, "n_resamples": n_resamples, "ci_resampling": "post_id clusters", "ekman_rule": ekman_rule_text(combine),
            "labels": block(P, H, names), "ekman": block(Pg, Hg, groups), "by_community_ekman": by_comm, "k_min": k_min,
            "chunking": {"share_longer_than_max_length": float((pred["n_tokens"] > max_length).mean()),
                         "mean_chunks": float(pred["n_chunks"].mean()), "max_chunks": int(pred["n_chunks"].max())},
            "note": DOMAIN_SHIFT_NOTE}


def domain_shift_dreaddit(cfg: dict[str, Any], seed: int, model_choice: str | None = None,
                          batch_size: int | None = None) -> dict[str, Any]:
    """INFERENCE ONLY: apply the final emotion model (calibration + thresholds from validation) to Dreaddit text and
    summarise emotions for stress vs not-stress segments. Writes predictions_dreaddit.parquet (ids + probabilities, no
    text; local, gitignored) and domain_shift_dreaddit.json (aggregates only)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    s = settings(cfg, model_choice)
    names = label_names(cfg)
    d = run_dir_for(cfg, s["model_choice"], seed)
    cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
    pp = d / "predictions_dreaddit.parquet"
    if pp.exists():
        pred = pd.read_parquet(pp)
    else:
        dd = SM.load_dreaddit(cfg)
        df = pd.concat([dd[k] for k in s["domain_shift"]["splits"]], ignore_index=True)
        tok = AutoTokenizer.from_pretrained(str(d / "best_model"))
        model = AutoModelForSequenceClassification.from_pretrained(str(d / "best_model"))
        device = select_device(cfg)
        bs = batch_size or (s["eval_batch_size"] if device == "cuda" else cfg["inference"]["batch_size"])
        ch = cfg["chunking"]
        P, n_chunks, n_tok = predict_chunked(model, tok, df["text_clean"].tolist(), s["max_length"], ch["stride"],
                                             ch["max_chunks_per_doc"], cal["params"], ch["aggregation"], bs, device,
                                             cfg["inference"]["half_precision"])
        pred = pd.concat([df[["record_id", "split", "community", "post_id", "label"]].reset_index(drop=True),
                          pd.DataFrame({"n_tokens": n_tok, "n_chunks": n_chunks}),
                          pd.DataFrame(P.astype(np.float32), columns=[f"p__{n}" for n in names])], axis=1)
        tmp = pp.with_suffix(".tmp.parquet")
        pred.to_parquet(tmp, index=False)
        os.replace(tmp, pp)
    out = summarise_domain_shift(pred, names, load_ekman_mapping(cfg, names), [thr["thresholds"][n] for n in names],
                                 s["ekman"]["combine"], cfg["privacy"]["k_min"], s["max_length"], s["eval"]["bootstrap_resamples"],
                                 cfg["primary_seed"], s["eval"]["ci_level"])
    out.update({"seed": seed, "model_choice": s["model_choice"], "splits": s["domain_shift"]["splits"],
                "calibration": "GoEmotions-validation Platt/temperature params (NOT re-fitted on Dreaddit)",
                "aggregation": cfg["chunking"]["aggregation"], "max_length": s["max_length"]})
    write_json(d / "domain_shift_dreaddit.json", out)
    return out


def write_handcheck_sample(cfg: dict[str, Any], seed: int, path: Path, model_choice: str | None = None) -> Path:
    """LOCAL hand-check sheet (CSV, never overwritten): n Dreaddit segments (half stress, half not, seeded) with the
    scrubbed text, top-k emotions, emotions above threshold and Ekman probabilities, plus blank columns to fill in.
    No author, no post id. Lives under data/ (gitignored); it is the only Step 4 file that holds text."""
    from .privacy import scrub_text
    path = Path(path)
    if path.exists():
        LOG.info("hand-check sheet exists, not overwritten: %s", path)
        return path
    s = settings(cfg, model_choice)
    names = label_names(cfg)
    d = run_dir_for(cfg, s["model_choice"], seed)
    thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    pred = pd.read_parquet(d / "predictions_dreaddit.parquet")
    n, k = s["domain_shift"]["handcheck_n"], s["domain_shift"]["handcheck_top_k"]
    smp = (pred.groupby("label", group_keys=False)[list(pred.columns)]
           .apply(lambda g: g.sample(min(n // 2, len(g)), random_state=cfg["primary_seed"]))
           .sample(frac=1.0, random_state=cfg["primary_seed"]).reset_index(drop=True))
    dd = pd.concat(SM.load_dreaddit(cfg).values(), ignore_index=True)[["record_id", "text_clean"]]
    smp = smp.merge(dd, on="record_id", how="left")
    P = smp[[f"p__{x}" for x in names]].to_numpy()
    mapping = load_ekman_mapping(cfg, names)
    Pg = ekman_probs(P, ekman_members(mapping, names), s["ekman"]["combine"])
    order = np.argsort(-P, axis=1)[:, :k]
    out = pd.DataFrame({
        "sample_id": np.arange(1, len(smp) + 1), "record_id": smp["record_id"], "community": smp["community"],
        "dreaddit_label": np.where(smp["label"] == 1, "stress", "not_stress"),
        "text": smp["text_clean"].map(lambda t: scrub_text(t, cfg)),
        "n_chunks": smp["n_chunks"],
        "top_emotions": [", ".join(f"{names[j]} ({P[i, j]:.2f})" for j in order[i]) for i in range(len(smp))],
        "emotions_above_threshold": [", ".join(n_ for j, n_ in enumerate(names) if P[i, j] >= thr[n_]) or "(none)"
                                     for i in range(len(smp))],
        **{f"ekman_{g}": Pg[:, j].round(3) for j, g in enumerate(mapping)},
        "hand_top_emotion_plausible (y/n/unsure)": "", "hand_missing_emotion": "", "hand_notes": "",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)
    return path


def plot_domain_shift(summary: dict[str, Any], path: str | None = None):
    """Difference in mean calibrated Ekman probability (stress minus not-stress Dreaddit segments) with the post-clustered
    95% CI: one series, dots + whiskers, zero line. Palette slot 1 (same hue as the reliability diagrams)."""
    import matplotlib.pyplot as plt
    rows = summary["ekman"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["diff_lo"], r["diff_hi"]], [yi, yi], color="#2a78d6", linewidth=2, solid_capstyle="round")
    ax.scatter([r["diff_mean_p"] for r in rows], y, s=40, color="#2a78d6", zorder=3, edgecolors="white", linewidths=1.5)
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.set_xlabel("difference in mean calibrated probability (stress − not stress)")
    ax.set_title("Ekman emotion language: Dreaddit stress vs not-stress segments (95% CI, post-clustered)",
                 loc="left", fontsize=10)
    ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


# ================================================================ bundle (4G)
def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_bundle(cfg: dict[str, Any], seed: int, model_choice: str | None = None, date: _dt.date | None = None,
                overwrite: bool = False) -> Path:
    """Write <paths.models>/emotion/v<YYYYMMDD>/: weights + tokenizer, emotion_config.json (label map, problem type,
    max length, per-label calibration, thresholds, Ekman rule, chunking), thresholds.json, ekman_mapping.json,
    metrics.json, model_card.md, manifest.json (SHA-256 of every file). Every number comes from saved run files.
    An existing bundle is refused unless overwrite=True (same file names overwritten in place, nothing deleted)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .preprocess import processed_path, split_path
    s = settings(cfg, model_choice)
    names = label_names(cfg)
    d = run_dir_for(cfg, s["model_choice"], seed)
    bundle = model_version_dir(cfg, COMPONENT, date)
    if (bundle / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"{bundle} already holds a bundle; pass another `date` or overwrite=True")
    bundle.mkdir(parents=True, exist_ok=True)
    cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
    final = json.loads((d / "FINAL_TEST_REPORT.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
    evaluation = _read_json(runs_root(cfg) / s["model_choice"] / "evaluation_uncalibrated.json")
    shift = _read_json(d / "domain_shift_dreaddit.json")
    mapping = load_ekman_mapping(cfg, names)
    model = AutoModelForSequenceClassification.from_pretrained(str(d / "best_model"))
    if model.config.problem_type != s["problem_type"] or model.config.num_labels != len(names):
        raise ValueError("best_model config is not the 28-label multi-label model this bundle describes")
    model.save_pretrained(str(bundle))
    AutoTokenizer.from_pretrained(str(d / "best_model")).save_pretrained(str(bundle))
    split_meta = json.loads(split_path(cfg, "goemotions").read_text(encoding="utf-8"))
    ch = cfg["chunking"]
    ecfg = {
        "component": COMPONENT, "model_choice": s["model_choice"], "base_model": s["hf_id"], "seed": seed,
        "problem_type": s["problem_type"], "activation": "sigmoid (one independent probability per label)",
        "label_map": dict(enumerate(names)), "num_labels": len(names), "max_length": s["max_length"],
        "input": "text_clean produced by stress_signals.preprocess.clean_text (Step 2)",
        "calibration": {"method": cal["method"], "formula": cal["formula"], "fitted_on": "validation",
                        "params": {p["label"]: {"a": p["a"], "b": p["b"]} for p in cal["params"]}},
        "thresholds": {"values": thr["thresholds"], "fitted_on": thr["fitted_on"], "criterion": thr["criterion"],
                       "low_support_rule": thr["low_support_rule"],
                       "low_support_labels": [r["label"] for r in thr["per_label"] if r["low_support"]]},
        "ekman": {"mapping_file": "ekman_mapping.json", "combine": s["ekman"]["combine"],
                  "rule": ekman_rule_text(s["ekman"]["combine"]), "groups": list(mapping)},
        "chunking": {"max_tokens": s["max_length"], "stride": min(ch["stride"], (s["max_length"] - 2) // 2),
                     "max_chunks_per_doc": ch["max_chunks_per_doc"], "aggregation": ch["aggregation"],
                     "rule": "sentence-aligned windows (preprocess.chunk_text); record = mean of calibrated chunk probabilities"},
        "training": {k: s[k] for k in ("learning_rate", "train_batch_size", "gradient_accumulation_steps", "epochs",
                                       "warmup_ratio", "weight_decay", "mixed_precision", "early_stopping_patience",
                                       "metric_for_best_model", "class_weighting")},
        "gradient_accumulation": summary.get("gradient_accumulation"),
        "epochs_completed": summary.get("epochs_completed"), "transformers_used_for_training": summary.get("transformers"),
        "data": {"dataset": "GoEmotions (official google-research files; HF go_emotions/simplified identical, Step 1)",
                 "split_file": split_path(cfg, "goemotions").name, "split_fingerprint": split_meta["data_fingerprint"],
                 "split_counts": split_meta["counts"]},
        "inference_defaults": {"precision": "fp32", "half_precision": False},
        "created_utc": _dt.datetime.now(_dt.timezone.utc),
    }
    write_json(bundle / "emotion_config.json", ecfg)
    write_json(bundle / "thresholds.json", thr)
    write_json(bundle / "ekman_mapping.json", {"source": cfg["emotion_model"]["ekman"]["mapping_file"],
                                               "combine": s["ekman"]["combine"], "rule": ekman_rule_text(s["ekman"]["combine"]),
                                               "groups": mapping})
    metrics = {"final_model": {"seed": seed, "calibration_validation": cal, "final_test_report": final},
               "all_seeds_uncalibrated": evaluation, "domain_shift_dreaddit": shift}
    write_json(bundle / "metrics.json", metrics)
    (bundle / "model_card.md").write_text(model_card(ecfg, metrics), encoding="utf-8")
    files = [p for p in sorted(bundle.iterdir()) if p.is_file() and p.name != "manifest.json"]
    gp = processed_path(cfg, "goemotions")
    write_manifest(bundle / "manifest.json", artifacts=files, seeds=s["seeds"],
                   dataset_versions={"goemotions_split_fingerprint": split_meta["data_fingerprint"],
                                     "goemotions_parquet_sha256": sha256_file(gp) if gp.exists() else None},
                   extra={"component": COMPONENT, "run_dir": d, "final_seed": seed}, root=bundle)
    return bundle


def _f(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def model_card(ec: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Short model card. Every number is read from `metrics` (saved run files)."""
    fin = metrics["final_model"]["final_test_report"]
    t, ci = fin["test"], fin["test_ci"]
    summ = (metrics.get("all_seeds_uncalibrated") or {}).get("summary", {}).get("test", {})
    seeds_line = (f"Across seeds (uncalibrated, threshold 0.5): macro-F1 {summ['f1_macro']['mean']:.3f} ± "
                  f"{summ['f1_macro']['std']:.3f}, macro AP {summ['ap_macro']['mean']:.3f} ± {summ['ap_macro']['std']:.3f} "
                  f"(n={summ['f1_macro']['n_runs']} runs)." if summ else "")
    per = sorted(t["per_label"], key=lambda r: r["f1"])
    weakest = ", ".join(f"{r['label']} {r['f1']:.2f} (n={r['support']})" for r in per[:5])
    ek = fin["ekman"]["test"]
    sens = fin.get("test_sensitivity_without_cross_split_overlap")
    shift = metrics.get("domain_shift_dreaddit")
    shift_line = ""
    if shift:
        top = sorted(shift["ekman"], key=lambda r: -abs(r["diff_mean_p"]))[:3]
        shift_line = ("Dreaddit check (inference only, n = {s} stress / {ns} not-stress segments): largest differences in mean "
                      "calibrated Ekman probability (stress − not stress): {d}. Plausibility hand-check: see the local sheet."
                      ).format(s=shift["n"]["stress"], ns=shift["n"]["not_stress"],
                               d="; ".join(f"{r['name']} {r['diff_mean_p']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]" for r in top))
    return f"""# Model card: emotion-language classifier ({ec['model_choice']}, seed {ec['seed']})

**What it is.** A multi-label classifier (28 independent sigmoid outputs: 27 emotions + neutral) of the emotions
*expressed in* short English Reddit-style text, fine-tuned from `{ec['base_model']}` on GoEmotions. Outputs feed
**aggregate, population-level** signals only.

**Not for diagnosis, and not a stress detector.** It does not assess any individual. A negative emotion is not stress,
and emotion language is not an emotional state. Do not use it for screening, profiling, ranking or moderation.

**Data.** GoEmotions (Demszky et al., 2020), official google-research files (Apache-2.0), Reddit comments, 2019.
Official splits {ec['data']['split_counts']}. Labels are rater judgements; many emotions are rare.

**Final test (calibrated: {ec['calibration']['method']} per label; per-label thresholds tuned on validation).**
macro-F1 {_f(t['f1_macro'])} [95% CI {ci['f1_macro']['lo']:.3f}–{ci['f1_macro']['hi']:.3f}],
micro-F1 {_f(t['f1_micro'])} [{ci['f1_micro']['lo']:.3f}–{ci['f1_micro']['hi']:.3f}], macro ROC-AUC {_f(t['roc_auc_macro'])},
macro AP {_f(t['ap_macro'])}, n = {t['n']}. Same calibrated probabilities at a global 0.5: macro-F1
{_f(fin['test_calibrated_at_0.5']['f1_macro'])}. Pooled ECE uncalibrated {_f(fin['test_ece']['uncalibrated']['pooled'])} →
calibrated {_f(fin['test_ece']['calibrated']['pooled'])}. {seeds_line}
Weakest labels (test F1): {weakest}. Low-support labels (threshold noisy): {', '.join(ec['thresholds']['low_support_labels']) or 'none'}.
{f"Without the {sens['n_excluded']} test rows whose text also occurs in train/validation: macro-F1 {_f(sens['f1_macro'])}." if sens else ""}

**Ekman view (dashboard).** {ec['ekman']['rule']}. Test macro-F1 over the 7 groups {_f(ek['f1_macro'])}, macro ROC-AUC
{_f(ek['roc_auc_macro'])}.

**Domain shift.** {shift_line or 'Not run yet (4F).'}
GoEmotions is short comments; the target corpora are longer posts (sentence-aligned chunks of ≤ {ec['max_length']} tokens,
mean of calibrated chunk probabilities). Calibration was fitted on GoEmotions validation and is not guaranteed on posts.

**Limitations.**
- Rare labels (e.g. grief, pride, relief, nervousness) have few examples; their F1 and thresholds are unstable (see CIs).
- Rater disagreement in GoEmotions is high for several emotions; labels are not ground truth about the writer.
- English only; 2019 Reddit comments; no demographic fairness audit is possible (no such attributes, by design).
- The Ekman group probability is a rule over member probabilities, not a separately calibrated model.
"""
