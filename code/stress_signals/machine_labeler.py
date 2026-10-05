"""Machine labelling of the sheets that were meant for a human (user decision, Step 6).

READ THIS FIRST. These labels come from a zero-shot NLI model, not from a person. Consequences, which the reports repeat:
  * A metric computed against them measures AGREEMENT WITH THIS LABELLER, not correctness. They are never "gold".
  * Machine labels are written to their OWN files (never into the human sheets), together with a meta JSON that records the
    model, hypotheses and threshold. Config `label_source: machine` switches the downstream code to them; every report,
    model card and verdict then says "MACHINE-LABELLED".
  * The labeller is a different model family from everything it is used to check (Step 3 RoBERTa fine-tuned on Dreaddit,
    the SAD-trained stressor models), so the check is not circular, but it is one imperfect model's opinion. The 0.5
    threshold is fixed, not tuned (there is nothing to tune it on).
  * Inference only: no weights are updated. Texts stay on this machine; nothing is sent anywhere.

Hypothesis wording and the NLI model id are in config.yaml (machine_labeler). Model availability/size is UNVERIFIED until the
first run downloads it.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .audit import interim_dir
from .utils import get_logger, select_device, sha256_file, write_json

LOG = get_logger("stress_signals.machine_labeler")
MACHINE_BANNER = "MACHINE-LABELLED (zero-shot NLI), NOT a human annotation: results are agreement with that labeller, not correctness"


def settings(cfg: dict[str, Any]) -> dict[str, Any]:
    if "machine_labeler" not in cfg:
        raise KeyError("CFG has no 'machine_labeler' section: re-run cell 0.1 (or restart the kernel) after config.yaml changed")
    return cfg["machine_labeler"]


def entailment_indices(id2label: dict[int, str]) -> tuple[int, int]:
    """(contradiction_index, entailment_index) read from the model's own label map (never assumed). Raises if absent."""
    low = {int(k): str(v).strip().lower() for k, v in id2label.items()}
    ent = [k for k, v in low.items() if v == "entailment"]
    con = [k for k, v in low.items() if v == "contradiction"]
    if len(ent) != 1 or len(con) != 1:
        raise ValueError(f"model labels {low} do not contain exactly one 'entailment' and one 'contradiction'")
    return con[0], ent[0]


class NliLabeler:
    """P(hypothesis is entailed by the text) from an NLI model, as softmax over the (contradiction, entailment) logits."""

    def __init__(self, cfg: dict[str, Any]):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        s = settings(cfg)
        self.hf_id, self.max_length, self.batch = s["model"], int(s["max_length"]), int(cfg["inference"]["batch_size"])
        self.device = select_device(cfg)
        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_id).to(self.device).eval()
        self.con, self.ent = entailment_indices(self.model.config.id2label)
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def p_entail(self, premises: Sequence[str], hypothesis: str) -> np.ndarray:
        import torch
        out = np.zeros(len(premises))
        order = np.argsort([len(p) for p in premises], kind="stable")
        with torch.inference_mode():
            for i in range(0, len(premises), self.batch):
                idx = order[i:i + self.batch]
                enc = self.tok([premises[j] for j in idx], [hypothesis] * len(idx), truncation="only_first",
                               max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                z = self.model(**enc).logits.float()[:, [self.con, self.ent]]
                out[idx] = torch.softmax(z, dim=1)[:, 1].cpu().numpy()
        return out


def stressor_labels_from_scores(scores: dict[str, np.ndarray], other_generic: np.ndarray, threshold: float,
                                categories: Sequence[str], none_col: str = "none_unclear") -> pd.DataFrame:
    """0/1 label columns (categories + other_unclear + none_unclear) from per-category entailment scores.
    A category is marked when its score >= threshold. If no real category is marked: other_unclear when the generic
    'names a specific cause' hypothesis passes the threshold, else none_unclear. none_unclear never combines with another."""
    n = len(next(iter(scores.values())))
    out = pd.DataFrame({c: (scores[c] >= threshold).astype(int) for c in categories if c in scores})
    named = out.sum(axis=1) > 0 if len(out.columns) else pd.Series(False, index=range(n))
    out["other_unclear"] = ((~named) & (other_generic >= threshold)).astype(int)
    out[none_col] = ((~named) & (other_generic < threshold)).astype(int)
    return out[[*[c for c in categories if c in out.columns], "other_unclear", none_col]]


def _meta(cfg: dict[str, Any], lab: NliLabeler, what: str, extra: dict[str, Any]) -> dict[str, Any]:
    s = settings(cfg)
    return {"label_source": "machine", "banner": MACHINE_BANNER, "what": what, "model": lab.hf_id, "model_revision": lab.revision,
            "device": lab.device, "threshold": float(s["threshold"]), "threshold_note": "fixed at 0.5; not tuned (no ground truth)",
            "max_length_tokens": lab.max_length, "truncation": "text truncated to max_length tokens (long posts are labelled on their start)",
            "created_utc": _dt.datetime.now(_dt.timezone.utc), **extra}


# ================================================================ SenticNet hand-check sheet
def senticnet_machine_path(cfg: dict[str, Any]) -> Path:
    return interim_dir(cfg) / "senticnet_handcheck_machine.csv"


def label_senticnet_handcheck(cfg: dict[str, Any], force: bool = False) -> Path:
    """Fill the SenticNet hand-check by machine: file, row_index, auto_label, machine_label (1 = text expresses stress-related
    language), machine_score. No text is written. The human sheet is not touched. Idempotent unless force."""
    out = senticnet_machine_path(cfg)
    if out.exists() and not force:
        LOG.info("machine hand-check exists, not overwriting: %s", out)
        return out
    src = interim_dir(cfg) / "senticnet_handcheck.csv"
    df = pd.read_csv(src, encoding="utf-8-sig")
    s = settings(cfg)
    lab = NliLabeler(cfg)
    score = lab.p_entail(df["text"].astype(str).tolist(), s["senticnet_hypothesis"])
    res = pd.DataFrame({"file": df["file"], "row_index": df["row_index"], "auto_label": df["auto_label"],
                        "machine_label": (score >= float(s["threshold"])).astype(int), "machine_score": score.round(4)})
    tmp = out.with_suffix(".tmp.csv")
    res.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(out)
    write_json(out.with_suffix(".json"), _meta(cfg, lab, "SenticNet hand-check (label noise)",
                                               {"hypothesis": s["senticnet_hypothesis"], "n_rows": int(len(res)),
                                                "source_sheet_sha256": sha256_file(src)}))
    return out


# ================================================================ stressor gold sheet
def label_stressor_gold(cfg: dict[str, Any], tax: dict[str, Any], force: bool = False) -> Path:
    """Fill the Step 5 gold sheet by machine into data/outputs/gold/stressor_gold_v<k>_machine.csv (gold_id + 0/1 label
    columns + notes; no text). The human sheet is not touched. Idempotent unless force."""
    from . import stressor_model as ST
    f = ST.gold_files(cfg)
    out = f["machine_sheet"]
    if out.exists() and not force:
        LOG.info("machine gold sheet exists, not overwriting: %s", out)
        return out
    if not f["human_sheet"].exists():
        raise FileNotFoundError(f"{f['human_sheet']} not found: run Step 5B.1 first")
    sheet = pd.read_csv(f["human_sheet"], encoding="utf-8-sig", dtype=str, keep_default_na=False)
    s = settings(cfg)
    cats = [c for c in ST.category_ids(tax) if c != "other_unclear"]
    hyp = s["stressor_hypotheses"]
    missing = [c for c in cats if c not in hyp]
    if missing:
        raise KeyError(f"machine_labeler.stressor_hypotheses lacks {missing}")
    lab = NliLabeler(cfg)
    texts = sheet["text"].tolist()
    scores = {c: lab.p_entail(texts, hyp[c]) for c in cats}
    generic = lab.p_entail(texts, s["other_generic_hypothesis"])
    labels = stressor_labels_from_scores(scores, generic, float(s["threshold"]), cats, ST.NONE_COL)
    res = pd.DataFrame({"gold_id": sheet["gold_id"], **labels, "notes": "machine label"})
    for c in ST.gold_columns(tax):
        if c not in res.columns:
            res[c] = 0
    res = res[["gold_id", *ST.gold_columns(tax), "notes"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.csv")
    res.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(out)
    write_json(out.with_suffix(".json"), _meta(cfg, lab, "stressor gold set v%s" % cfg["stressor_model"]["gold"]["version"],
                                               {"hypotheses": {**{c: hyp[c] for c in cats}, "other_generic": s["other_generic_hypothesis"]},
                                                "n_rows": int(len(res)), "positives": {c: int(res[c].sum()) for c in ST.gold_columns(tax)},
                                                "rule": "category marked if entailment >= threshold; none marked -> other_unclear if the "
                                                        "generic 'names a cause' hypothesis passes, else none_unclear"}))
    return out
