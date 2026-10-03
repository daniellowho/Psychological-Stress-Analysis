"""Step 3: binary stress-language classifier on Dreaddit (training is RUN BY THE USER on the laptop's NVIDIA GPU if it
has one, else on the CPU; see utils.select_device).

Pipeline per seed (resume-safe; every stage writes to durable storage and is skipped if its output exists):
    data guard (text only) -> fine-tune with HF Trainer (checkpoint every epoch, early stopping on validation
    f1_stress, best model reloaded at the end, CSV log) -> best_model/ -> logits for validation + test -> DONE.json
Evaluation, calibration (temperature scaling, threshold) and shortcut checks read the saved logits, so they can
run on any device without retraining. Final bundle: <paths.models>/stress/v<YYYYMMDD>/.

Run layout: <project root>/trained_models/stress/runs/<model_choice>/seed<k>/{best_model/, train_log.csv,
            train_summary.json, predictions_validation.parquet, predictions_test.parquet, DONE.json}.
Checkpoints: <checkpoint_root>/stress/<model_choice>/seed<k>/ (data/outputs/checkpoints; never inside trained_models,
            because the Trainer deletes old checkpoints).
"""

from __future__ import annotations

import copy
import csv
import datetime as _dt
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import (METRIC_KEYS, binary_metrics, bootstrap_metrics, choose_threshold, ece, fit_temperature,
                      mean_std, metrics_by_group, paired_bootstrap_gain, plot_reliability, positive_proba)
from .utils import (checkpoint_root, device_label, get_logger, library_versions, model_version_dir, models_root,
                    select_device, set_seed, sha256_file, write_json, write_manifest)

LOG = get_logger("stress_signals.stress_model")

# Dreaddit's precomputed columns (LIWC, DAL, syntax, social, sentiment). They cannot be reproduced on new Reddit
# text, so they are NEVER model inputs. Step 2 already dropped them; this guard re-drops them explicitly.
DREADDIT_FEATURE_PREFIXES = ("lex_liwc_", "lex_dal_", "lex_", "syntax_", "social_", "sentiment")
MODEL_INPUT_COLUMNS = ["record_id", "text_clean", "label", "confidence", "community", "post_id", "split", "official_split"]
TRAINING_BANNER = "# >>> USER RUNS THIS CELL. DO NOT EXECUTE. <<<"


# ================================================================ settings / paths
def settings(cfg: dict[str, Any], model_choice: str | None = None) -> dict[str, Any]:
    """stress_model config with the chosen model resolved (`hf_id`) and int label keys."""
    s = copy.deepcopy(cfg["stress_model"])
    choice = model_choice or s["model_choice"]
    if choice not in s["model_options"]:
        raise ValueError(f"model_choice {choice!r} not in {list(s['model_options'])}")
    s["model_choice"], s["hf_id"] = choice, s["model_options"][choice]["hf_id"]
    s["label_map"] = {int(k): str(v) for k, v in s["label_map"].items()}
    return s


def runs_root(cfg: dict[str, Any]) -> Path:
    """<project root>/trained_models/stress/runs (on whichever kernel runs the code)."""
    return models_root(cfg) / "stress" / "runs"


def run_dir_for(cfg: dict[str, Any], model_choice: str, seed: int) -> Path:
    return runs_root(cfg) / model_choice / f"seed{seed}"


def checkpoint_dir_for(cfg: dict[str, Any], run_dir: Path) -> Path:
    """Trainer output dir for a run: outside trained_models (see utils.checkpoint_root), unique per run (<group>/<run name>)."""
    run_dir = Path(run_dir)
    return checkpoint_root(cfg) / "stress" / run_dir.parent.name / run_dir.name


# ================================================================ data
def select_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Explicitly drop Dreaddit's precomputed feature columns and keep only the text + bookkeeping columns.
    The model sees `text_clean` (Step 2 cleaning of Dreaddit's `text`) and nothing else."""
    feats = [c for c in df.columns if str(c).startswith(DREADDIT_FEATURE_PREFIXES)]
    if feats:
        LOG.info("dropping %d precomputed feature columns (never model inputs)", len(feats))
        df = df.drop(columns=feats)
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"processed Dreaddit is missing columns {missing}; rerun Step 2")
    return df[MODEL_INPUT_COLUMNS].copy()


def verify_split(cfg: dict[str, Any], df: pd.DataFrame) -> dict[str, int]:
    """The parquet's `split` column must equal the saved fixed split file (never regenerated silently)."""
    from .preprocess import split_path
    saved = json.loads(split_path(cfg, "dreaddit").read_text(encoding="utf-8"))["splits"]
    for name, ids in saved.items():
        have = set(df.loc[df["split"] == name, "record_id"])
        if have != set(ids):
            raise ValueError(f"Dreaddit split {name!r} in the parquet differs from {split_path(cfg, 'dreaddit').name}")
    return {k: len(v) for k, v in saved.items()}


def load_dreaddit(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """train / validation / test frames from data/processed/dreaddit.parquet (split file verified, text-only guard)."""
    p = Path(cfg["_paths"]["processed"]) / "dreaddit.parquet"
    df = pd.read_parquet(p)
    verify_split(cfg, df)
    df = select_model_inputs(df)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test")}


def confidence_weights(confidence: Sequence[float], floor: float = 0.5) -> np.ndarray:
    """Optional sample weights from Dreaddit `confidence` (share of annotators agreeing with the label).

    Why it can help: low-agreement segments are the most likely to carry a noisy label, so down-weighting them
    reduces the influence of label noise. Why it is OFF by default: (1) it changes the effective training
    distribution (ambiguous texts are exactly the ones the model meets in the wild); (2) some rows have
    confidence 0.0, which would delete them; (3) the paper's evaluation does not use weights, so results stay
    comparable. The floor keeps every example in play; weights are renormalised to mean 1 (same loss scale)."""
    w = np.maximum(np.nan_to_num(np.asarray(confidence, dtype=float), nan=1.0), float(floor))
    return w / w.mean()


# ================================================================ transformers compatibility
def hf_compat() -> dict[str, Any]:
    """Argument names that changed across transformers versions, read from the INSTALLED version's signatures:
    `eval_strategy` (new) vs `evaluation_strategy` (old); Trainer `processing_class` (>=4.46) vs `tokenizer`."""
    import transformers
    from transformers import Trainer, TrainingArguments
    ta = inspect.signature(TrainingArguments.__init__).parameters
    tr = inspect.signature(Trainer.__init__).parameters
    return {"transformers": transformers.__version__,
            "eval_strategy_key": "eval_strategy" if "eval_strategy" in ta else "evaluation_strategy",
            "tokenizer_key": "processing_class" if "processing_class" in tr else "tokenizer",
            "training_args_params": set(ta)}


def precision_flags(setting: str, hf_id: str, cuda: bool, bf16_supported: bool) -> dict[str, bool]:
    """fp16/bf16 flags. fp16 only with CUDA (the CPU trains in fp32). DeBERTa-v3 is known to overflow in fp16, so `auto` uses bf16 when the
    GPU supports it (A100/L4) and fp32 otherwise (e.g. T4)."""
    off = {"fp16": False, "bf16": False}
    if not cuda or setting == "no":
        return off
    if setting == "fp16":
        return {"fp16": True, "bf16": False}
    if setting == "bf16":
        if not bf16_supported:
            raise ValueError("bf16 requested but this GPU does not support it")
        return {"fp16": False, "bf16": True}
    if setting != "auto":
        raise ValueError(f"unknown mixed_precision {setting!r}")
    if "deberta" in hf_id.lower():
        return {"fp16": False, "bf16": True} if bf16_supported else off
    return {"fp16": True, "bf16": False}


def build_training_args(s: dict[str, Any], output_dir: Path, seed: int, cuda: bool, bf16_supported: bool = False):
    """TrainingArguments for one run: evaluate + checkpoint every epoch, keep best (validation f1_stress), CSV-friendly
    logging, no external reporting. Keys unknown to the installed version are dropped (logged)."""
    from transformers import TrainingArguments
    compat = hf_compat()
    kw: dict[str, Any] = {
        "output_dir": str(output_dir), "num_train_epochs": s["epochs"], "learning_rate": s["learning_rate"],
        "per_device_train_batch_size": s["train_batch_size"], "per_device_eval_batch_size": s["eval_batch_size"],
        "gradient_accumulation_steps": s["gradient_accumulation_steps"], "weight_decay": s["weight_decay"],
        "warmup_ratio": s["warmup_ratio"], "lr_scheduler_type": "linear",
        compat["eval_strategy_key"]: "epoch", "save_strategy": "epoch", "logging_strategy": "steps", "logging_steps": 20,
        "load_best_model_at_end": True, "metric_for_best_model": s["metric_for_best_model"], "greater_is_better": True,
        "save_total_limit": s["save_total_limit"], "seed": seed, "data_seed": seed, "report_to": "none",
        "remove_unused_columns": False,   # keep `sample_weight` in the batch for the weighted loss
        "dataloader_num_workers": 0 if os.name == "nt" else s.get("dataloader_num_workers", 0),
        **precision_flags(s["mixed_precision"], s["hf_id"], cuda, bf16_supported),
    }
    dropped = [k for k in kw if k not in compat["training_args_params"]]
    if dropped:
        LOG.warning("TrainingArguments in transformers %s does not accept %s (dropped)", compat["transformers"], dropped)
    return TrainingArguments(**{k: v for k, v in kw.items() if k not in dropped})


# ================================================================ torch pieces (imported lazily)
def weighted_cross_entropy(logits, labels, weights=None):
    """Mean CE, or weight-normalised CE: sum(w_i * ce_i) / sum(w_i)."""
    import torch.nn.functional as F
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    if weights is None:
        return ce.mean()
    w = weights.to(ce.dtype)
    return (ce * w).sum() / w.sum().clamp_min(1e-12)


class EncodedDataset:
    """Pre-tokenised examples (dynamic padding happens in the collator)."""

    def __init__(self, encodings: dict[str, list], labels: Sequence[int], weights: Sequence[float] | None = None):
        self.enc, self.labels = encodings, list(map(int, labels))
        self.weights = list(map(float, weights)) if weights is not None else [1.0] * len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"], item["sample_weight"] = self.labels[i], self.weights[i]
        return item


class WeightedCollator:
    """Pads token fields with the tokenizer; stacks `labels` (long) and `sample_weight` (float)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        tok = [{k: v for k, v in f.items() if k not in ("labels", "sample_weight")} for f in features]
        batch = self.tokenizer.pad(tok, return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        batch["sample_weight"] = torch.tensor([f["sample_weight"] for f in features], dtype=torch.float)
        return batch


def encode(tokenizer, texts: Sequence[str], max_length: int) -> dict[str, list]:
    return dict(tokenizer(list(texts), truncation=True, max_length=max_length, padding=False))


def _trainer_classes():
    """WeightedTrainer + CSV logger, defined lazily so importing this module never requires transformers."""
    from transformers import Trainer, TrainerCallback

    class WeightedTrainer(Trainer):
        """Trainer whose loss honours per-example `sample_weight` (all 1.0 unless confidence weights are on)."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # kwargs: num_items_in_batch (>=4.46)
            inputs = dict(inputs)
            weights = inputs.pop("sample_weight", None)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_cross_entropy(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    class CSVLogger(TrainerCallback):
        """Appends every Trainer log entry to a long-format CSV (time, step, epoch, key, value); survives resumes."""

        def __init__(self, path: Path):
            self.path = Path(path)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.is_world_process_zero:
                return
            new = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["time_utc", "global_step", "epoch", "key", "value"])
                t = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        w.writerow([t, state.global_step, state.epoch, k, v])

    return WeightedTrainer, CSVLogger


def trainer_compute_metrics(eval_pred) -> dict[str, float]:
    """Validation metrics during training (threshold 0.5, uncalibrated). Keys get an `eval_` prefix from Trainer."""
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    m = binary_metrics(eval_pred.label_ids, positive_proba(logits), 0.5)
    return {k: m[k] for k in METRIC_KEYS}


# ================================================================ inference
def resolve_device(cfg: dict[str, Any]) -> str:
    """Inference device: config inference.device, else the NVIDIA GPU if there is one, else the CPU."""
    return select_device(cfg)


def predict_logits(model, tokenizer, texts: Sequence[str], max_length: int, batch_size: int = 16,
                   device: str = "cpu", half: bool = False) -> np.ndarray:
    """Logits (n, num_labels) in input order. No gradients, eval mode, fp32 unless `half` (keep False on Pascal).
    Texts are length-sorted for efficient batching and returned in the original order."""
    import torch
    model = model.to(device).eval()
    if half and device == "cuda":
        model = model.half()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = tokenizer([texts[j] for j in idx], truncation=True, max_length=max_length, padding=True,
                            return_tensors="pt").to(device)
            out[idx] = model(**enc).logits.float().cpu().numpy()
    return out


def _save_predictions(df: pd.DataFrame, logits: np.ndarray, path: Path) -> None:
    out = df[["record_id", "label", "confidence", "community", "post_id", "split"]].copy()
    out["logit_0"], out["logit_1"] = logits[:, 0], logits[:, 1]
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ================================================================ training (USER RUNS)
def train_stress_seed(cfg: dict[str, Any], seed: int, model_choice: str | None = None,
                      data: dict[str, pd.DataFrame] | None = None, run_dir: Path | None = None,
                      epochs: int | None = None) -> Path:
    """Fine-tune one seed. Resume-safe: (1) DONE.json -> skip; (2) best_model/ exists -> skip training, only
    (re)write missing predictions; (3) otherwise resume from the last epoch checkpoint if any.
    `data` may override the splits (used by leave-one-subreddit-out); its "test" frame is only predicted."""
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
    data = data or load_dreaddit(cfg)
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    best_dir = run_dir / "best_model"
    tok = AutoTokenizer.from_pretrained(s["hf_id"], use_fast=True)
    cuda = torch.cuda.is_available()

    if not (best_dir / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(
            s["hf_id"], num_labels=2, id2label=s["label_map"], label2id={v: k for k, v in s["label_map"].items()})
        tr, va = data["train"], data["validation"]
        w = confidence_weights(tr["confidence"], s["confidence_weights"]["floor"]) if s["confidence_weights"]["enabled"] else None
        ds_tr = EncodedDataset(encode(tok, tr["text_clean"], s["max_length"]), tr["label"], w)
        ds_va = EncodedDataset(encode(tok, va["text_clean"], s["max_length"]), va["label"])
        bf16_ok = bool(cuda and torch.cuda.is_bf16_supported())
        ck = checkpoint_dir_for(cfg, run_dir)        # outside trained_models: the Trainer rotates (deletes) checkpoints
        args = build_training_args(s, ck, seed, cuda, bf16_ok)
        WeightedTrainer, CSVLogger = _trainer_classes()
        compat = hf_compat()
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, data_collator=WeightedCollator(tok),
            compute_metrics=trainer_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=s["early_stopping_patience"]),
                       CSVLogger(run_dir / "train_log.csv")],
            **{compat["tokenizer_key"]: tok})
        last = get_last_checkpoint(str(ck)) if ck.is_dir() else None
        LOG.info("seed %s: training %s (%s) on %s, resume_from=%s", seed, s["model_choice"], s["hf_id"],
                 device_label(), last)
        result = trainer.train(resume_from_checkpoint=last)
        trainer.save_model(str(best_dir))          # load_best_model_at_end=True -> this is the best epoch
        tok.save_pretrained(str(best_dir))
        write_json(run_dir / "train_summary.json", {
            "seed": seed, "model_choice": s["model_choice"], "hf_id": s["hf_id"], "settings": s,
            "best_metric_validation_f1_stress": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint, "epochs_completed": trainer.state.epoch,
            "global_step": trainer.state.global_step, "train_runtime_s": result.metrics.get("train_runtime"),
            "checkpoint_dir": ck,
            "log_history": trainer.state.log_history, "transformers": compat["transformers"],
            "confidence_weights": s["confidence_weights"], "n_train": len(tr), "n_validation": len(va),
            "device": device_label()})
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
        logits = predict_logits(model, tok, df["text_clean"].tolist(), s["max_length"], s["eval_batch_size"], device)
        _save_predictions(df, logits, out)
    write_json(run_dir / "DONE.json", {"seed": seed, "model_choice": s["model_choice"],
                                       "finished_utc": _dt.datetime.now(_dt.timezone.utc)})
    return run_dir


# ================================================================ evaluation (reads saved logits)
def load_predictions(run_dir: Path, split: str, temperature: float = 1.0) -> pd.DataFrame:
    df = pd.read_parquet(Path(run_dir) / f"predictions_{split}.parquet")
    df["p"] = positive_proba(df[["logit_0", "logit_1"]].to_numpy(), temperature)
    return df


def completed_seeds(cfg: dict[str, Any], model_choice: str | None = None) -> dict[int, Path]:
    s = settings(cfg, model_choice)
    out = {}
    for seed in s["seeds"]:
        d = run_dir_for(cfg, s["model_choice"], seed)
        if (d / "DONE.json").exists():
            out[seed] = d
    return out


def readiness(cfg: dict[str, Any], model_choice: str | None = None) -> dict[str, Any]:
    """What exists on THIS kernel, so evaluation cells can skip with a reason instead of failing under "Run All".
    Evaluation waits for ALL seeds: the final seed and the once-only test report must not be chosen from a partial set."""
    s = settings(cfg, model_choice)
    done = completed_seeds(cfg, model_choice)
    missing = [k for k in s["seeds"] if k not in done]
    return {"baseline": (runs_root(cfg) / "baseline_tfidf_lr" / "DONE.json").exists(),
            "seeds_done": sorted(done), "seeds_missing": missing, "all_seeds": not missing,
            "why_seeds": f"needs every seed of {s['model_choice']} {s['seeds']}; missing {missing}. "
                         "They train in 3B.1 (a failed or interrupted seed resumes there on the next Run All).",
            "why_baseline": "needs the baseline, which trains in 3A.1."}


def evaluate_runs(cfg: dict[str, Any], model_choice: str | None = None, threshold: float = 0.5) -> dict[str, Any]:
    """Uncalibrated metrics (threshold 0.5) on validation and the official test for every completed seed:
    point metrics, cluster-bootstrap 95% CIs, the high-confidence subset, and mean +/- std across seeds."""
    s = settings(cfg, model_choice)
    seeds = completed_seeds(cfg, model_choice)
    if not seeds:
        raise FileNotFoundError(f"no completed runs under {runs_root(cfg) / s['model_choice']}; run the training cell first")
    ev, hc_thr = s["eval"], s["high_confidence_threshold"]
    res: dict[str, Any] = {"model_choice": s["model_choice"], "threshold": threshold, "seeds": {}, "summary": {}}
    for seed, d in seeds.items():
        r: dict[str, Any] = {}
        for split in ("validation", "test"):
            df = load_predictions(d, split)
            hc = df[df["confidence"] >= hc_thr]
            r[split] = {
                "all": binary_metrics(df["label"], df["p"], threshold),
                "ci": bootstrap_metrics(df["label"], df["p"], threshold, ev["bootstrap_resamples"],
                                        df[ev["bootstrap_cluster"]] if ev.get("bootstrap_cluster") else None,
                                        seed=cfg["primary_seed"], level=ev["ci_level"]),
                "high_confidence": {**binary_metrics(hc["label"], hc["p"], threshold), "confidence_threshold": hc_thr,
                                    "share_of_split": float(len(hc) / max(len(df), 1))},
            }
        res["seeds"][seed] = r
    for split in ("validation", "test"):
        res["summary"][split] = mean_std([res["seeds"][k][split]["all"] for k in res["seeds"]])
        res["summary"][f"{split}_high_confidence"] = mean_std([res["seeds"][k][split]["high_confidence"] for k in res["seeds"]])
    write_json(runs_root(cfg) / s["model_choice"] / "evaluation_uncalibrated.json", res)
    return res


def per_subreddit_errors(cfg: dict[str, Any], seed: int, split: str = "test", model_choice: str | None = None,
                         temperature: float = 1.0, threshold: float = 0.5) -> pd.DataFrame:
    """Per-community n, prevalence, accuracy, F1, ROC-AUC and FP/FN rates (aggregates only)."""
    s = settings(cfg, model_choice)
    df = load_predictions(run_dir_for(cfg, s["model_choice"], seed), split, temperature)
    return metrics_by_group(df, "community", threshold=threshold)


def select_final_seed(evaluation: dict[str, Any]) -> int:
    """Deployable model = seed with the best VALIDATION f1_stress (test is never used for selection)."""
    return max(evaluation["seeds"], key=lambda k: evaluation["seeds"][k]["validation"]["all"]["f1_stress"])


def calibrate(cfg: dict[str, Any], seed: int, model_choice: str | None = None) -> dict[str, Any]:
    """Fit temperature T and the decision threshold on VALIDATION ONLY; report ECE/Brier before/after on validation;
    save calibration.json + reliability diagram next to the run.

    Threshold rule (user decision): search only calibration.threshold_grid (0.40-0.60) for max validation F1(stress);
    APPLY the tuned threshold only if its validation gain over 0.5 is clear (decide_threshold); else apply 0.5."""
    s = settings(cfg, model_choice)
    d = run_dir_for(cfg, s["model_choice"], seed)
    va = load_predictions(d, "validation")
    logits = va[["logit_0", "logit_1"]].to_numpy()
    T = fit_temperature(logits, va["label"].to_numpy())
    p_cal = positive_proba(logits, T)
    c = s["calibration"]
    thr = choose_threshold(va["label"], p_cal, c["threshold_criterion"], c["threshold_grid"])
    dec = decide_threshold(va["label"], p_cal, thr["threshold"], c["apply_rule"],
                           va[s["eval"]["bootstrap_cluster"]] if s["eval"].get("bootstrap_cluster") else None,
                           seed=cfg["primary_seed"], level=s["eval"]["ci_level"])
    out = {"seed": seed, "model_choice": s["model_choice"], "fitted_on": "validation", "n_validation": int(len(va)),
           "temperature": T, "threshold_tuned": thr["threshold"], "threshold_search_grid": c["threshold_grid"],
           "threshold_criterion": thr["criterion"], "threshold_value_on_validation": thr["value_on_validation"],
           "threshold_decision": dec, "threshold": dec["applied_threshold"],
           "validation_before": {"ece": ece(va["label"], va["p"], c["ece_bins"]), **binary_metrics(va["label"], va["p"], 0.5)},
           "validation_after": {"ece": ece(va["label"], p_cal, c["ece_bins"]), **binary_metrics(va["label"], p_cal, 0.5)},
           "validation_after_tuned": binary_metrics(va["label"], p_cal, thr["threshold"]),
           "ece_bins": c["ece_bins"]}
    write_json(d / "calibration.json", out)
    thr["curve"].to_csv(d / "threshold_curve_validation.csv", index=False)
    plot_reliability({"uncalibrated": (va["label"], va["p"]), f"temperature-scaled (T={T:.2f})": (va["label"], p_cal)},
                     c["ece_bins"], f"Reliability on validation: {s['model_choice']} seed {seed}",
                     path=str(d / "reliability_validation.png"))
    return out


def decide_threshold(y_true: Sequence[int], p: Sequence[float], tuned: float, rule: dict[str, Any],
                     groups: Sequence[Any] | None = None, seed: int = 42, level: float = 0.95) -> dict[str, Any]:
    """Apply `tuned` instead of 0.5 only if the VALIDATION F1(stress) gain is clear: gain >= rule.min_gain and (if
    required) the paired cluster-bootstrap CI of the gain lies entirely above 0. The output records the caveat that the
    threshold was chosen on the same data, so the gain estimate is optimistic."""
    g = paired_bootstrap_gain(y_true, p, tuned, 0.5, "f1_stress", rule["bootstrap_resamples"], groups, seed, level)
    clear = g["gain"] >= rule["min_gain"] and (g["lo"] > 0 if rule.get("require_ci_above_zero", True) else True)
    if abs(tuned - 0.5) < 1e-9:
        clear, reason = False, "tuned threshold equals 0.5"
    elif clear:
        reason = f"gain {g['gain']:.4f} >= {rule['min_gain']} and CI [{g['lo']:.4f}, {g['hi']:.4f}] above 0"
    else:
        reason = (f"gain {g['gain']:.4f} not clear (min_gain {rule['min_gain']}, CI [{g['lo']:.4f}, {g['hi']:.4f}]); "
                  "keeping 0.5")
    return {"tuned_threshold": float(tuned), "applied_threshold": float(tuned) if clear else 0.5,
            "applied": "tuned" if clear else "0.5", "validation_gain_f1_stress": g, "rule": rule, "reason": reason,
            "caveat": "gain measured on the validation set that selected the threshold (optimistic)"}


def final_test_report(cfg: dict[str, Any], seed: int, model_choice: str | None = None, force: bool = False) -> dict[str, Any]:
    """Calibrated test metrics, computed ONCE with the validation-fitted T and threshold. If FINAL_TEST_REPORT.json
    exists it is returned unchanged (guards against tuning on test). `force` only after a documented change."""
    s = settings(cfg, model_choice)
    d = run_dir_for(cfg, s["model_choice"], seed)
    path = d / "FINAL_TEST_REPORT.json"
    if path.exists() and not force:
        LOG.info("final test report already exists (reported once): %s", path)
        return json.loads(path.read_text(encoding="utf-8"))
    cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    te = load_predictions(d, "test", cal["temperature"])
    hc = te[te["confidence"] >= s["high_confidence_threshold"]]
    ev = s["eval"]
    def block(t: float) -> dict[str, Any]:
        return {"threshold": t, "test": binary_metrics(te["label"], te["p"], t),
                "test_ci": bootstrap_metrics(te["label"], te["p"], t, ev["bootstrap_resamples"], te[ev["bootstrap_cluster"]],
                                             seed=cfg["primary_seed"], level=ev["ci_level"]),
                "test_high_confidence": binary_metrics(hc["label"], hc["p"], t)}

    at_05, at_tuned = block(0.5), block(cal["threshold_tuned"])     # user decision: report test at BOTH thresholds
    applied = at_tuned if cal["threshold_decision"]["applied"] == "tuned" else at_05
    rep = {"seed": seed, "model_choice": s["model_choice"], "temperature": cal["temperature"],
           "threshold": applied["threshold"], "threshold_applied": cal["threshold_decision"]["applied"],
           "threshold_decision_reason": cal["threshold_decision"]["reason"],
           "test": applied["test"], "test_ci": applied["test_ci"], "test_high_confidence": applied["test_high_confidence"],
           "at_threshold_0.5": at_05, "at_threshold_tuned": at_tuned,
           "test_ece": ece(te["label"], te["p"], s["calibration"]["ece_bins"]),
           "created_utc": _dt.datetime.now(_dt.timezone.utc)}
    write_json(path, rep)
    return json.loads(path.read_text(encoding="utf-8"))   # identical object on first and later calls


# ================================================================ baseline (USER RUNS)
def build_baseline(b: dict[str, Any], C: float, seed: int):
    """TF-IDF word (1-2) + char_wb (2-5) n-grams -> logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    feats = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=tuple(b["word_ngrams"]), min_df=b["min_df"],
                                 max_features=b["max_features_word"], sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=tuple(b["char_ngrams"]),
                                 min_df=b["min_df"], max_features=b["max_features_char"], sublinear_tf=True)),
    ])
    return Pipeline([("features", feats),
                     ("clf", LogisticRegression(C=C, max_iter=b["max_iter"], solver="liblinear", random_state=seed))])


def fit_baseline(cfg: dict[str, Any], data: dict[str, pd.DataFrame] | None = None, seed: int | None = None) -> Path:
    """Grid over C on VALIDATION f1_stress (fit on train only; ties -> smaller C), save model + val/test probs.
    Output: runs/baseline_tfidf_lr/. Skips if DONE.json exists."""
    import joblib
    s = settings(cfg)
    b, seed = s["baseline"], cfg["primary_seed"] if seed is None else seed
    out = runs_root(cfg) / "baseline_tfidf_lr"
    if (out / "DONE.json").exists():
        LOG.info("baseline already trained: %s", out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    data = data or load_dreaddit(cfg)
    tr, va, te = data["train"], data["validation"], data["test"]
    rows, best = [], None
    for C in sorted(b["C_grid"]):
        pipe = build_baseline(b, C, seed).fit(tr["text_clean"], tr["label"])
        m = binary_metrics(va["label"], pipe.predict_proba(va["text_clean"])[:, 1], 0.5)
        rows.append({"C": C, **{k: m[k] for k in METRIC_KEYS}})
        if best is None or m["f1_stress"] > best[1] + 1e-12:
            best = (pipe, m["f1_stress"], C)
    pipe, _, C = best
    pd.DataFrame(rows).to_csv(out / "c_selection_validation.csv", index=False)
    joblib.dump(pipe, out / "model.joblib")
    for split, df in (("validation", va), ("test", te)):
        p = pipe.predict_proba(df["text_clean"])[:, 1]
        o = df[["record_id", "label", "confidence", "community", "post_id", "split"]].copy()
        o["p"] = p
        o.to_parquet(out / f"predictions_{split}.parquet", index=False)
    write_json(out / "DONE.json", {"C": C, "seed": seed, "settings": b, "finished_utc": _dt.datetime.now(_dt.timezone.utc)})
    return out


def evaluate_baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    out = runs_root(cfg) / "baseline_tfidf_lr"
    s = settings(cfg)
    ev = s["eval"]
    res = {"C": json.loads((out / "DONE.json").read_text(encoding="utf-8"))["C"]}
    for split in ("validation", "test"):
        df = pd.read_parquet(out / f"predictions_{split}.parquet")
        hc = df[df["confidence"] >= s["high_confidence_threshold"]]
        res[split] = {"all": binary_metrics(df["label"], df["p"], 0.5),
                      "ci": bootstrap_metrics(df["label"], df["p"], 0.5, ev["bootstrap_resamples"], df[ev["bootstrap_cluster"]],
                                              seed=cfg["primary_seed"], level=ev["ci_level"]),
                      "high_confidence": binary_metrics(hc["label"], hc["p"], 0.5)}
    write_json(out / "evaluation.json", res)
    return res


def top_ngrams(cfg: dict[str, Any], k: int | None = None) -> pd.DataFrame:
    """Largest positive (-> stress) and negative (-> not stress) LR coefficients of the baseline, flagged when the
    n-gram contains a subreddit name / domain term from the shortcut list (a sign of topic shortcuts)."""
    import joblib
    s = settings(cfg)
    k = k or s["baseline"]["top_k_ngrams"]
    pipe = joblib.load(runs_root(cfg) / "baseline_tfidf_lr" / "model.joblib")
    names = pipe.named_steps["features"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    rx = mask_regex(s["shortcut"]["subreddit_names"] + s["shortcut"]["domain_terms"])
    order = np.argsort(coef)
    rows = []
    for direction, idx in (("stress", order[::-1][:k]), ("not_stress", order[:k])):
        for rank, i in enumerate(idx, 1):
            feat = str(names[i])
            kind, gram = feat.split("__", 1)
            rows.append({"direction": direction, "rank": rank, "type": kind, "ngram": gram, "coef": float(coef[i]),
                         "topic_term": bool(rx.search(gram)) if kind == "word" else None})
    return pd.DataFrame(rows)


# ================================================================ shortcut checks
def mask_regex(terms: Sequence[str]) -> re.Pattern[str]:
    """Case-insensitive whole-word match of any term, also in r/<term> form; longest terms first."""
    alts = sorted({re.escape(t.lower()) for t in terms if t}, key=len, reverse=True)
    return re.compile(r"(?i)(?<!\w)(?:/?r/)?(?:" + "|".join(alts) + r")(?!\w)")


def mask_texts(texts: Sequence[str], terms: Sequence[str], mask_token: str) -> tuple[list[str], list[int]]:
    """Replace every subreddit/domain term with `mask_token`. Returns masked texts and per-text mask counts."""
    rx = mask_regex(terms)
    out, counts = [], []
    for t in texts:
        new, n = rx.subn(mask_token, t)
        out.append(new)
        counts.append(n)
    return out, counts


WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']+\b")


def random_mask(texts: Sequence[str], counts: Sequence[int], mask_token: str, seed: int = 42,
                protect: re.Pattern[str] | None = None) -> list[str]:
    """Control condition: mask the SAME number of randomly chosen other words per text (words matching `protect`
    are never chosen). Separates 'the model lost information' from 'the model relied on topic words'."""
    rng = np.random.default_rng(seed)
    out = []
    for t, n in zip(texts, counts):
        spans = [m.span() for m in WORD_RE.finditer(t) if not (protect and protect.fullmatch(m.group(0)))]
        if n <= 0 or not spans:
            out.append(t)
            continue
        pick = sorted(rng.choice(len(spans), size=min(n, len(spans)), replace=False).tolist(), reverse=True)
        for i in pick:
            a, b = spans[i]
            t = t[:a] + mask_token + t[b:]
        out.append(t)
    return out


def shortcut_masking_eval(cfg: dict[str, Any], model_dir: Path, split: str = "test", temperature: float = 1.0,
                          threshold: float = 0.5, model_choice: str | None = None, batch_size: int | None = None) -> pd.DataFrame:
    """INFERENCE ONLY (no training): metrics on original text vs topic-masked vs random-masked control.
    Diagnostic only; nothing is tuned on these numbers."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    s = settings(cfg, model_choice)
    data = load_dreaddit(cfg)[split]
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    mtok = tok.mask_token or "<mask>"
    terms = s["shortcut"]["subreddit_names"] + s["shortcut"]["domain_terms"]
    masked, counts = mask_texts(data["text_clean"].tolist(), terms, mtok)
    sc = s["shortcut"]
    protect = mask_regex(terms)
    conditions = [("original", None, data["text_clean"].tolist()), ("topic_masked", None, masked)]
    for k in range(int(sc.get("random_control_draws", 1))):          # user decision: 5 draws
        draw_seed = int(sc["random_control_seed"]) + k
        conditions.append(("random_masked_control", draw_seed,
                           random_mask(data["text_clean"].tolist(), counts, mtok, draw_seed, protect)))
    device = resolve_device(cfg)
    bs = batch_size or (s["eval_batch_size"] if device == "cuda" else cfg["inference"]["batch_size"])
    rows = []
    for cond, draw_seed, texts in conditions:
        lg = predict_logits(model, tok, texts, s["max_length"], bs, device, cfg["inference"]["half_precision"])
        m = binary_metrics(data["label"], positive_proba(lg, temperature), threshold)
        rows.append({"condition": cond, "draw_seed": draw_seed, **{k: m[k] for k in METRIC_KEYS},
                     "predicted_positive_rate": m["predicted_positive_rate"]})
    res = pd.DataFrame(rows)
    res["texts_with_masked_terms"] = int(np.sum(np.asarray(counts) > 0))
    res["masked_terms_total"] = int(np.sum(counts))
    res["n"] = len(data)
    write_json(runs_root(cfg) / s["model_choice"] / "shortcut_masking.json",
               {"split": split, "temperature": temperature, "threshold": threshold, "model_dir": model_dir,
                "results": res.to_dict(orient="records"),
                "summary": summarise_masking(res).to_dict(orient="records")})
    return res


def summarise_masking(res: pd.DataFrame, metrics: Sequence[str] = ("f1_stress", "roc_auc")) -> pd.DataFrame:
    """Drop from `original` under topic masking vs the random control (mean, sd, min, max over the draws).
    Topic reliance is suggested when the topic drop exceeds the LARGEST control drop."""
    orig = res[res["condition"] == "original"].iloc[0]
    topic = res[res["condition"] == "topic_masked"].iloc[0]
    ctrl = res[res["condition"] == "random_masked_control"]
    rows = []
    for m in metrics:
        cd = orig[m] - ctrl[m]
        td = orig[m] - topic[m]
        rows.append({"metric": m, "original": float(orig[m]), "drop_topic_masked": float(td),
                     "drop_control_mean": float(cd.mean()), "drop_control_sd": float(cd.std(ddof=1)) if len(cd) > 1 else 0.0,
                     "drop_control_min": float(cd.min()), "drop_control_max": float(cd.max()), "n_control_draws": int(len(cd)),
                     "topic_drop_exceeds_all_control_draws": bool(td > cd.max())})
    return pd.DataFrame(rows)


def loso_data(data: dict[str, pd.DataFrame], heldout: str) -> dict[str, pd.DataFrame]:
    """Leave-one-subreddit-out: train/validation without `heldout`; ALL rows of `heldout` (never seen) as test."""
    allrows = pd.concat([data["train"], data["validation"], data["test"]], ignore_index=True)
    return {"train": data["train"][data["train"]["community"] != heldout].reset_index(drop=True),
            "validation": data["validation"][data["validation"]["community"] != heldout].reset_index(drop=True),
            "test": allrows[allrows["community"] == heldout].reset_index(drop=True)}


def train_loso(cfg: dict[str, Any], subreddits: Sequence[str] | None = None) -> dict[str, Path]:
    """USER RUNS. One model per held-out subreddit (config shortcut.loso_model / loso_epochs / loso_seed)."""
    s = settings(cfg)
    sc = s["shortcut"]
    data = load_dreaddit(cfg)
    subs = list(subreddits or sorted(pd.concat(data.values())["community"].unique()))
    out = {}
    for sub in subs:
        d = runs_root(cfg) / "loso" / f"{sc['loso_model']}_{sub}"
        out[sub] = train_stress_seed(cfg, sc["loso_seed"], sc["loso_model"], loso_data(data, sub), d, sc["loso_epochs"])
    return out


def evaluate_loso(cfg: dict[str, Any]) -> pd.DataFrame:
    """Held-out-subreddit metrics (threshold 0.5, uncalibrated) for every finished LOSO run."""
    rows = []
    for d in sorted((runs_root(cfg) / "loso").glob("*")):
        if not (d / "DONE.json").exists():
            continue
        df = load_predictions(d, "test")
        m = binary_metrics(df["label"], df["p"], 0.5)
        rows.append({"heldout_subreddit": df["community"].iloc[0], "run": d.name, **{k: m[k] for k in METRIC_KEYS},
                     "n": m["n"], "prevalence": m["prevalence"]})
    res = pd.DataFrame(rows)
    if len(res):
        write_json(runs_root(cfg) / "loso" / "loso_summary.json", {"threshold": 0.5, "results": res.to_dict(orient="records")})
    return res


# ================================================================ bundle
def save_bundle(cfg: dict[str, Any], seed: int, model_choice: str | None = None, date: _dt.date | None = None,
                overwrite: bool = False) -> Path:
    """Write <paths.models>/stress/v<YYYYMMDD>/: weights + tokenizer (save_pretrained), stress_config.json,
    metrics.json, model_card.md, manifest.json (SHA-256 of every file). Every number comes from saved run files.
    Never deletes: an existing bundle is refused unless overwrite=True, which overwrites the same file names in place."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .preprocess import split_path
    s = settings(cfg, model_choice)
    d = run_dir_for(cfg, s["model_choice"], seed)
    bundle = model_version_dir(cfg, "stress", date)
    if (bundle / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"{bundle} already holds a bundle; pass another `date` or overwrite=True (files are "
                              "overwritten in place, nothing is deleted)")
    bundle.mkdir(parents=True, exist_ok=True)
    cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    final = json.loads((d / "FINAL_TEST_REPORT.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
    rr = runs_root(cfg)
    evaluation = _read_json(rr / s["model_choice"] / "evaluation_uncalibrated.json")
    baseline = _read_json(rr / "baseline_tfidf_lr" / "evaluation.json")
    shortcut = _read_json(rr / s["model_choice"] / "shortcut_masking.json")
    loso = _read_json(rr / "loso" / "loso_summary.json")

    AutoModelForSequenceClassification.from_pretrained(str(d / "best_model")).save_pretrained(str(bundle))
    AutoTokenizer.from_pretrained(str(d / "best_model")).save_pretrained(str(bundle))
    ch = cfg["chunking"]
    split_meta = json.loads(split_path(cfg, "dreaddit").read_text(encoding="utf-8"))
    stress_cfg = {
        "component": "stress", "model_choice": s["model_choice"], "base_model": s["hf_id"], "seed": seed,
        "label_map": s["label_map"], "positive_label": s["positive_label"], "max_length": s["max_length"],
        "input": "text_clean produced by stress_signals.preprocess.clean_text (Step 2)",
        "chunking": {"max_tokens": s["max_length"], "stride": ch["stride"], "max_chunks_per_doc": ch["max_chunks_per_doc"],
                     "aggregation": ch["aggregation"],
                     "rule": "sentence-aligned windows (preprocess.chunk_text); record score = mean of calibrated chunk P(stress)"},
        "calibration": {"method": "temperature_scaling", "temperature": cal["temperature"], "fitted_on": "validation"},
        "threshold": {"value": cal["threshold"], "applied": cal["threshold_decision"]["applied"],
                      "tuned_value": cal["threshold_tuned"], "search_grid": cal["threshold_search_grid"],
                      "criterion": cal["threshold_criterion"], "apply_rule": cal["threshold_decision"]["rule"],
                      "decision_reason": cal["threshold_decision"]["reason"], "fitted_on": "validation"},
        "training": {k: s[k] for k in ("learning_rate", "train_batch_size", "epochs", "warmup_ratio", "weight_decay",
                                       "mixed_precision", "early_stopping_patience", "confidence_weights")},
        "epochs_completed": summary.get("epochs_completed"), "transformers_used_for_training": summary.get("transformers"),
        "data": {"dataset": "Dreaddit (andreagasparini/dreaddit)", "split_file": split_path(cfg, "dreaddit").name,
                 "split_fingerprint": split_meta["data_fingerprint"], "split_counts": split_meta["counts"]},
        "inference_defaults": {"precision": "fp32", "half_precision": False},
        "created_utc": _dt.datetime.now(_dt.timezone.utc),
    }
    write_json(bundle / "stress_config.json", stress_cfg)
    metrics = {"final_model": {"seed": seed, "calibration_validation": cal, "final_test_report": final},
               "all_seeds_uncalibrated": evaluation, "baseline_tfidf_lr": baseline,
               "shortcut_masking": shortcut, "leave_one_subreddit_out": loso}
    write_json(bundle / "metrics.json", metrics)
    (bundle / "model_card.md").write_text(model_card(stress_cfg, metrics), encoding="utf-8")
    files = [p for p in sorted(bundle.iterdir()) if p.is_file() and p.name != "manifest.json"]
    write_manifest(bundle / "manifest.json", artifacts=files, seeds=s["seeds"],
                   dataset_versions={"dreaddit_split_fingerprint": split_meta["data_fingerprint"],
                                     "dreaddit_parquet_sha256": sha256_file(Path(cfg["_paths"]["processed"]) / "dreaddit.parquet")},
                   extra={"component": "stress", "run_dir": d, "final_seed": seed}, root=bundle)
    return bundle


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fmt(m: dict[str, Any] | None, key: str) -> str:
    if not m or key not in m or m[key] is None:
        return "n/a"
    return f"{m[key]:.3f}"


def model_card(sc: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Short model card. Every number is read from `metrics` (saved run files); nothing is typed in by hand."""
    fin = metrics["final_model"]["final_test_report"]
    t, ci = fin["test"], fin["test_ci"]
    summ = (metrics.get("all_seeds_uncalibrated") or {}).get("summary", {}).get("test", {})
    base = (metrics.get("baseline_tfidf_lr") or {}).get("test", {}).get("all")
    seeds_line = (f"Across seeds (uncalibrated, threshold 0.5): F1(stress) {summ['f1_stress']['mean']:.3f} ± "
                  f"{summ['f1_stress']['std']:.3f} (n={summ['f1_stress']['n_runs']} runs)." if summ else "")
    return f"""# Model card: stress-language classifier ({sc['model_choice']}, seed {sc['seed']})

**What it is.** A binary classifier of *stress-related language* in short English Reddit-style text, fine-tuned from
`{sc['base_model']}` on Dreaddit text only. Its outputs feed **aggregate, population-level** signals.

**Not for diagnosis.** It does not detect mental illness or assess any individual. Do not use it for screening,
profiling, ranking, moderation or any decision about a person. Report results only as aggregates, with cells of n < K_MIN suppressed.

**Data.** Dreaddit (Turcan & McKeown, 2019): 2017–2018 posts from 10 subreddits in 5 domains (abuse, anxiety,
financial, PTSD, social). The labels are crowd-annotated segments. The HF card states no licence (UNVERIFIED). Splits: official test kept as the final
test; train/validation grouped by post_id. Label map: {sc['label_map']} (from the Dreaddit paper; the meaning is not in the HF card).

**Final test (calibrated, threshold {fin['threshold']:.2f} chosen on validation, T = {fin['temperature']:.3f}).**
F1(stress) {_fmt(t, 'f1_stress')} [95% CI {ci['f1_stress']['lo']:.3f}–{ci['f1_stress']['hi']:.3f}],
macro-F1 {_fmt(t, 'f1_macro')}, ROC-AUC {_fmt(t, 'roc_auc')}, PR-AUC {_fmt(t, 'pr_auc')}, accuracy {_fmt(t, 'accuracy')},
ECE {fin['test_ece']:.3f}, n = {t['n']}. {seeds_line}
Threshold: searched {sc['threshold']['search_grid'][0]:.2f}-{sc['threshold']['search_grid'][1]:.2f} on validation (tuned value
{sc['threshold']['tuned_value']:.2f}); applied = {sc['threshold']['applied']} ({sc['threshold']['decision_reason']}).
Test F1(stress) at 0.5: {_fmt(fin['at_threshold_0.5']['test'], 'f1_stress')}; at the tuned threshold: {_fmt(fin['at_threshold_tuned']['test'], 'f1_stress')}.
TF-IDF + LR baseline on the same test set (threshold 0.5): F1(stress) {_fmt(base, 'f1_stress')}, ROC-AUC {_fmt(base, 'roc_auc')}.

**Inference.** Input is `text_clean` from Step 2 cleaning; max {sc['max_length']} tokens. Longer posts are split into sentence-aligned chunks (stride
{sc['chunking']['stride']}), and the post score is the mean of calibrated chunk probabilities. fp32 by default.

**Limitations.**
- Domain shift: Dreaddit segments are short (median ~99 tokens) and come from 5 domains in 2017–2018. Target communities, years
  (2019–2022, COVID) and post lengths differ, so calibration and accuracy may not transfer. Re-check on a labelled sample.
- Topic shortcuts: see the masking and leave-one-subreddit-out results in metrics.json. The model may partly key on topic words
  (money, abuse) rather than stress expression.
- The labels are annotator judgements of whether a segment expresses stress. They are not clinical ground truth. Some segments have low annotator agreement.
- English only. No fairness audit across demographic groups was possible (no such attributes are available, by design).
"""
