"""Step 6: TensiStrength auxiliary signal (6A) and SenticNet external validation (6B).

Everything here is INFERENCE or evaluation, except `train_cross` (optional SenticNet -> Dreaddit fine-tuning), which is
USER-RUN and off by default (config external_validation.cross_training.enabled).

6A  TensiStrength (Java jar) is an independent lexicon signal. It runs on SAMPLES only, in ONE Java process (stdin mode),
    and only after the user confirms academic use in config. We report Spearman correlation with the calibrated
    Step 3 stress probability. A correlation between two scorers is agreement, not validity: neither is ground truth.
6B  SenticNet labels come from an automated DNN-based strategy (SenticNet README), so every number is AGREEMENT with those
    labels, never accuracy. A hand-check (data/processed/_cache/senticnet_handcheck.csv, filled by the user) estimates the
    label noise and from it the best agreement ANY classifier could reach (the "ceiling").

No raw text is written by this module: score files hold record ids and numbers only. Texts are scrubbed (Step 2) and exist
only in memory, or in a pipe to the Java process.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import stress_model as SM
from .audit import interim_dir, project_path
from .metrics import binary_metrics, ece, fit_temperature, plot_reliability, positive_proba, reliability_table
from .preprocess import dedupe_key
from .utils import (get_logger, java_version, library_versions, models_root, select_device, sha256_file, write_json,
                    write_manifest)

LOG = get_logger("stress_signals.external_validation")
TRAINING_BANNER = SM.TRAINING_BANNER
LICENCE_NOTE = ("TensiStrength: free for academic use, GBP 1000 for commercial use (data/raw/tensistrength/README.md). "
                "The jar is run only if external_validation.tensistrength.academic_use_confirmed is true.")
LABEL_NOTE = ("SenticNet labels were produced by an AUTOMATED DNN-based strategy (SenticNet README), not by human "
              "annotators. All numbers are AGREEMENT with those labels, not accuracy.")
METRIC_KEYS = ("f1_stress", "roc_auc", "pr_auc", "ece", "brier", "precision_stress", "recall_stress")
MACHINE_NOTE = ("The hand-check was labelled by a zero-shot NLI model (machine_labeler), NOT by a person. The 'label noise' and the "
                "ceiling below are therefore agreement between two automated systems, not an estimate of true label error.")
LIMITATIONS = [
    "SenticNet labels are automated. Agreement with them measures similarity to another model, not correctness.",
    "The label-noise estimate rests on a small hand-check by one annotator; its confidence intervals are wide.",
    "F1 and PR-AUC depend on label prevalence, which differs between Dreaddit and each SenticNet file. ROC-AUC is the "
    "least prevalence-sensitive number here.",
    "Differences from Dreaddit test mix domain shift, label noise, prevalence and text-length differences; this "
    "analysis cannot separate them.",
    "Calibration (ECE) against noisy labels is inflated even for a perfect model; read it as 'probabilities may need "
    "recalibration on this domain', not as a measured error rate.",
    "TensiStrength is a lexicon tool with no labelled validation data in its repository. A correlation shows the two "
    "signals move together, not that either measures psychological stress.",
    "Nothing here assesses any individual. All results are over sets of texts.",
]


# ================================================================ paths / small helpers
def settings(cfg: dict[str, Any]) -> dict[str, Any]:
    if "external_validation" not in cfg:
        raise KeyError("CFG has no 'external_validation' section: it was loaded before Step 6 was added to config.yaml. "
                       "Re-run cell 0.1 (or restart the kernel), then run this cell again.")
    return cfg["external_validation"]


def out_dir(cfg: dict[str, Any]) -> Path:
    d = project_path(cfg, settings(cfg)["out_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _atomic_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def latest_stress_bundle(cfg: dict[str, Any]) -> Path:
    """Newest <models>/stress/v<YYYYMMDD>/ that holds a manifest and stress_config.json (Step 3F)."""
    root = models_root(cfg) / "stress"
    cands = sorted(p for p in root.glob("v*") if (p / "manifest.json").is_file() and (p / "stress_config.json").is_file())
    if not cands:
        raise FileNotFoundError(f"no stress bundle under {root}: run Step 3F first")
    return cands[-1]


def bundle_info(bundle: Path) -> dict[str, Any]:
    return json.loads((Path(bundle) / "stress_config.json").read_text(encoding="utf-8"))


def decision_threshold(cfg: dict[str, Any], info: dict[str, Any]) -> float:
    t = settings(cfg).get("threshold")
    return float(t) if t is not None else float(info["threshold"]["value"])


def platform_of(name: str) -> str:
    return "reddit" if str(name).lower().startswith("reddit") else "twitter"


# ================================================================ metrics with bootstrap (adds ECE)
def _point(y: np.ndarray, p: np.ndarray, thr: float, bins: int) -> dict[str, float]:
    m = binary_metrics(y, p, thr)
    return {**{k: float(m[k]) for k in METRIC_KEYS if k != "ece"}, "ece": float(ece(y, p, bins))}


def bootstrap_draws(y: Sequence[int], p: Sequence[float], thr: float, groups: Sequence[Any] | None = None,
                    n: int = 1000, seed: int = 42, bins: int = 15) -> dict[str, np.ndarray]:
    """Per-resample metric values (percentile bootstrap; whole groups are resampled when `groups` is given).
    Resamples where a metric is undefined (one class only) are NaN for it."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    members = None
    if groups is not None:
        codes, uniq = pd.factorize(pd.Series(np.asarray(groups)).astype(str))
        members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    out = {k: np.full(n, np.nan) for k in METRIC_KEYS}
    for b in range(n):
        if members is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        for k, v in _point(y[idx], p[idx], thr, bins).items():
            out[k][b] = v
    return out


def _ci(d: np.ndarray, level: float) -> list[float]:
    d = d[~np.isnan(d)]
    a = (1 - level) / 2
    return [float(np.quantile(d, a)), float(np.quantile(d, 1 - a))] if d.size else [float("nan"), float("nan")]


def evaluate_block(y: Sequence[int], p: Sequence[float], thr: float, groups: Sequence[Any] | None = None,
                   n_boot: int = 1000, seed: int = 42, level: float = 0.95, bins: int = 15
                   ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Point metrics + bootstrap CIs. Returns (block, draws); `draws` are used for paired-free differences."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    draws = bootstrap_draws(y, p, thr, groups, n_boot, seed, bins)
    pt = _point(y, p, thr, bins)
    return ({"n": int(len(y)), "threshold": float(thr), "prevalence_label_1": float(y.mean()),
             "predicted_positive_rate": float((p >= thr).mean()), "point": pt,
             "ci": {k: _ci(draws[k], level) for k in METRIC_KEYS}, "ci_level": level, "n_resamples": int(n_boot),
             "resampling": "whole groups" if groups is not None else "rows"}, draws)


def delta_block(point_a: dict[str, float], draws_a: dict[str, np.ndarray], point_b: dict[str, float],
                draws_b: dict[str, np.ndarray], level: float = 0.95) -> dict[str, Any]:
    """a - b for each metric, CI from the difference of two INDEPENDENT bootstrap distributions (different datasets)."""
    out = {}
    for k in METRIC_KEYS:
        d = draws_a[k] - draws_b[k]
        out[k] = {"delta": float(point_a[k] - point_b[k]), "ci": _ci(d, level)}
    return out


# ================================================================ 6A: TensiStrength
def tensistrength_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    src = cfg["sources"]["tensistrength"]
    d = project_path(cfg, src["local_dir"])
    return d / src["jar_name"], d / "TensiStrength_Data"


def tensistrength_gate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Can the jar be used? Needs: user-confirmed academic use, Java, the jar and its resource folder."""
    t = settings(cfg)["tensistrength"]
    jar, data = tensistrength_paths(cfg)
    java = java_version()
    why: list[str] = []
    if not cfg["sources"]["tensistrength"].get("enabled", True):
        why.append("sources.tensistrength.enabled is false")
    if not t["academic_use_confirmed"]:
        why.append("external_validation.tensistrength.academic_use_confirmed is false (set it to true only if your use is "
                   "academic; commercial use needs a paid licence)")
    if java is None:
        why.append("Java not found on PATH or JAVA_HOME")
    if not jar.is_file():
        why.append(f"jar missing: {jar}")
    if not data.is_dir():
        why.append(f"resource folder missing: {data} (unzip TensiStrength_Data.zip; Step 1 does this)")
    return {"usable": not why, "reasons": why, "java_version": java, "jar": jar.name, "licence": LICENCE_NOTE,
            "academic_use_confirmed": bool(t["academic_use_confirmed"])}


def _java_exe() -> str:
    import os
    exe = shutil.which("java")
    if exe is None and os.environ.get("JAVA_HOME"):
        cand = Path(os.environ["JAVA_HOME"], "bin", "java.exe" if os.name == "nt" else "java")
        exe = str(cand) if cand.exists() else None
    if exe is None:
        raise FileNotFoundError("java not found")
    return exe


def _resource_dir(cfg: dict[str, Any]) -> Path:
    """TensiStrength's manual says the data folder path must contain no spaces. If ours does (e.g. 'TA-BDA Project'),
    use a copy of the (text-free) resource folder in the system temp dir."""
    _, data = tensistrength_paths(cfg)
    if " " not in str(data):
        return data
    dest = Path(tempfile.gettempdir()) / "stress_signals_tensistrength_data"
    if " " in str(dest):
        raise RuntimeError(f"neither {data} nor {dest} is free of spaces; copy TensiStrength_Data to a path without spaces "
                           "and point sources.tensistrength.local_dir at its parent")
    if not dest.is_dir():
        tmp = dest.with_name(dest.name + "_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(data, tmp)
        tmp.rename(dest)
    return dest


_SCORE_RE = re.compile(r"^\s*([+-]?\d+)[\s,]+([+-]?\d+)(?:\s|$)")


def parse_tensistrength_line(line: str) -> tuple[int, int] | None:
    """'4 -1' -> (4, -1): (relaxation, stress). Returns None if the line does not start with two integers."""
    m = _SCORE_RE.match(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_tensistrength_output(stdout: str, n_expected: int) -> pd.DataFrame:
    """One output line per input line, in order. Observed scale (Step 1 audit, inferred from outputs, not documented in the
    manual): first number = relaxation 1..5, second = stress -1..-5; 1 / -1 mean none. Raises on any surprise rather than
    guessing: wrong line count (alignment lost), unparsable line, or values outside the observed scale."""
    lines = stdout.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) != n_expected:
        raise ValueError(f"TensiStrength returned {len(lines)} lines for {n_expected} inputs; refusing to guess alignment. "
                         f"First lines: {lines[:3]}")
    rows = []
    for i, ln in enumerate(lines):
        r = parse_tensistrength_line(ln)
        if r is None:
            raise ValueError(f"line {i}: cannot parse scores from {ln[:60]!r}")
        rows.append(r)
    relax = np.array([r[0] for r in rows])
    stress = np.array([r[1] for r in rows])
    if relax.min() < 1 or relax.max() > 5 or stress.max() > -1 or stress.min() < -5:
        raise ValueError(f"scores outside the observed scale (relaxation 1..5, stress -1..-5): relax {relax.min()}..{relax.max()}, "
                         f"stress {stress.min()}..{stress.max()}")
    return pd.DataFrame({"relaxation": relax, "stress_strength": -stress})


def _run_stdin(cfg: dict[str, Any], texts: Sequence[str], utf8: bool, timeout: int) -> str:
    jar, _ = tensistrength_paths(cfg)
    cmd = [_java_exe(), "-jar", str(jar), "stdin", "sentidata", _resource_dir(cfg).as_posix() + "/"] + (["UTF8"] if utf8 else [])
    clean = [" ".join(str(t).split()) or "." for t in texts]       # one line per text; never an empty line
    res = subprocess.run(cmd, input=("\n".join(clean) + "\n").encode("utf-8"), capture_output=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"java exited with {res.returncode}: {res.stderr.decode('utf-8', 'replace')[:400]}")
    return res.stdout.decode("utf-8", "replace")


def tensistrength_batch(cfg: dict[str, Any], texts: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score many texts in ONE Java process (stdin mode). Returns (scores, timing). The gate must pass first."""
    gate = tensistrength_gate(cfg)
    if not gate["usable"]:
        raise RuntimeError("TensiStrength not usable: " + "; ".join(gate["reasons"]))
    timeout = int(settings(cfg)["tensistrength"]["timeout_s"])
    t0 = time.time()
    used_utf8 = True
    try:
        out = _run_stdin(cfg, texts, True, timeout)
        scores = parse_tensistrength_output(out, len(texts))
    except (RuntimeError, ValueError) as exc:
        LOG.warning("stdin mode with UTF8 failed (%s); retrying without the UTF8 flag", str(exc)[:200])
        used_utf8 = False
        out = _run_stdin(cfg, texts, False, timeout)
        scores = parse_tensistrength_output(out, len(texts))
    dt = time.time() - t0
    return scores, {"n_texts": len(texts), "seconds": round(dt, 2), "texts_per_second": round(len(texts) / dt, 1) if dt else None,
                    "utf8_flag_used": used_utf8}


def tensistrength_probe(cfg: dict[str, Any]) -> dict[str, Any]:
    """Format + speed check on the SYNTHETIC probe sentences: one batch call vs one Java launch per text."""
    probes = list(settings(cfg)["tensistrength"]["probe_texts"])
    scores, timing = tensistrength_batch(cfg, probes)
    singles = []
    for t in probes:
        t0 = time.time()
        _run_stdin(cfg, [t], timing["utf8_flag_used"], int(settings(cfg)["tensistrength"]["timeout_s"]))
        singles.append(time.time() - t0)
    return {"probe_texts": probes, "scores": scores.to_dict(orient="records"), "batch_timing": timing,
            "seconds_per_single_launch": [round(s, 3) for s in singles],
            "mean_seconds_per_single_launch": round(float(np.mean(singles)), 3),
            "scale": "relaxation 1..5, stress strength 1..5 (= minus the second printed number); 1 = none. Inferred from outputs."}


WHY_NOT_AT_SCALE = [
    "Java dependency: every machine that runs the pipeline needs a compatible JRE, and Spark workers would each need one too.",
    "Process startup: a JVM launch per record (or per Spark task) costs far more than the lexicon lookup itself. The probe "
    "cell measures seconds per single launch against texts per second in one batch on THIS machine; those two numbers "
    "(in external_validation.json) are the evidence, not an assumption.",
    "Licence: free for academic use only (GBP 1000 commercial), so it should not be a hard dependency of the pipeline.",
    "No labelled validation data ships with it, so its scores cannot be verified, only compared to other signals.",
    "A lexicon tool adds a second, differently-biased signal at the cost of a second runtime; for millions of records the "
    "Step 3 model and, if wanted, a Python re-implementation of a lexicon feature would be the scalable route.",
]


def _senticnet_sample(cfg: dict[str, Any]) -> pd.DataFrame:
    df, _ = load_senticnet_eval(cfg)
    n = int(settings(cfg)["tensistrength"]["senticnet_sample_per_platform"])
    parts = [g.sample(n=min(n, len(g)), random_state=settings(cfg)["seed"]) for _, g in df.groupby("platform")]
    return pd.concat(parts, ignore_index=True)


def run_tensistrength_sample(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """6A.1: probe, then score the Dreaddit test split and a seeded SenticNet sample. Saves ids + numbers only.
    Skips (and says why) if the gate fails. Idempotent."""
    d = out_dir(cfg)
    scores_path, run_path = d / "tensistrength_scores.parquet", d / "tensistrength_run.json"
    gate = tensistrength_gate(cfg)
    if not gate["usable"]:
        res = {"status": "not_run", "gate": gate, "why_not_at_scale": WHY_NOT_AT_SCALE}
        write_json(run_path, res)
        LOG.warning("TensiStrength skipped: %s", "; ".join(gate["reasons"]))
        return res
    if scores_path.exists() and run_path.exists() and not force:
        return _read_json(run_path)
    probe = tensistrength_probe(cfg)
    t = settings(cfg)["tensistrength"]
    dre = pd.read_parquet(Path(cfg["_paths"]["processed"]) / "dreaddit.parquet")
    dre = dre[dre["split"] == t["dreaddit_split"]]
    sen = _senticnet_sample(cfg)
    parts, timings = [], {}
    for name, frame, grp in (("dreaddit_" + t["dreaddit_split"], dre, dre["community"]), ("senticnet", sen, sen["community"])):
        s, tm = tensistrength_batch(cfg, frame["text_clean"].tolist())
        parts.append(pd.concat([pd.DataFrame({"record_id": frame["record_id"].to_numpy(), "dataset": name,
                                              "group": grp.to_numpy()}), s], axis=1))
        timings[name] = tm
    _atomic_parquet(pd.concat(parts, ignore_index=True), scores_path)
    res = {"status": "done", "gate": gate, "probe": probe, "timings": timings, "why_not_at_scale": WHY_NOT_AT_SCALE,
           "sample": {"dreaddit_split": t["dreaddit_split"], "n_dreaddit": int(len(dre)), "n_senticnet": int(len(sen)),
                      "senticnet_sample_per_platform": int(t["senticnet_sample_per_platform"]), "seed": settings(cfg)["seed"]},
           "scores_file": scores_path.name, "scores_sha256": sha256_file(scores_path),
           "created_utc": _dt.datetime.now(_dt.timezone.utc)}
    write_json(run_path, res)
    return res


def spearman_ci(x: Sequence[float], y: Sequence[float], groups: Sequence[Any] | None = None, n: int = 1000, seed: int = 42,
                level: float = 0.95) -> dict[str, Any]:
    """Spearman rho (average ranks for ties) with a percentile bootstrap CI (whole groups if given). NaN if a side is constant."""
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def rho(ix: np.ndarray) -> float:
        a, b = x[ix], y[ix]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

    full = np.arange(len(x))
    rng = np.random.default_rng(seed)
    if groups is not None:
        codes, uniq = pd.factorize(pd.Series(np.asarray(groups)).astype(str))
        members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    draws = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, len(x), len(x)) if groups is None else \
            np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        draws[b] = rho(idx)
    return {"rho": rho(full), "ci": _ci(draws, level), "n": int(len(x)), "share_ties_x": float(1 - len(np.unique(x)) / len(x)),     # x = the model's P(stress)
            "share_ties_y": float(1 - len(np.unique(y)) / len(y)), "resampling": "whole groups" if groups is not None else "rows"}


def tensistrength_correlations(cfg: dict[str, Any]) -> dict[str, Any]:
    """6A.2: Spearman(calibrated Step 3 P(stress), TensiStrength stress strength) on Dreaddit test and on the SenticNet
    sample, plus context (lexicon coverage, mean strength by label). Needs 6A.1; the SenticNet part needs 6B.1."""
    d = out_dir(cfg)
    sp = d / "tensistrength_scores.parquet"
    if not sp.exists():
        gate = (_read_json(d / "tensistrength_run.json") or {}).get("gate") or tensistrength_gate(cfg)
        res = {"status": "not_run", "reason": gate["reasons"] or ["6A.1 has not produced scores"]}
        write_json(d / "tensistrength_result.json", res)
        return res
    ev, seed, n = settings(cfg), settings(cfg)["seed"], settings(cfg)["bootstrap_resamples"]
    ts = pd.read_parquet(sp)
    bundle = latest_stress_bundle(cfg)
    info = bundle_info(bundle)
    ref = dreaddit_test_reference(cfg, info)
    res: dict[str, Any] = {"status": "done", "scale": "stress_strength 1..5 (1 = none), relaxation 1..5; see tensistrength_run.json",
                           "bundle": bundle.name, "datasets": {}}

    def one(name: str, df: pd.DataFrame, groups: Sequence[Any] | None, label_name: str) -> dict[str, Any]:
        from sklearn.metrics import roc_auc_score
        blk = {"n": int(len(df)),
               "spearman_stress_vs_p_stress": spearman_ci(df["p"], df["stress_strength"], groups, n, seed, ev["ci_level"]),
               "spearman_relaxation_vs_p_stress": spearman_ci(df["p"], df["relaxation"], groups, n, seed, ev["ci_level"]),
               "share_texts_with_any_stress_term": float((df["stress_strength"] > 1).mean()),
               "mean_stress_strength_by_label": {str(k): float(v) for k, v in df.groupby(label_name)["stress_strength"].mean().items()}}
        blk["roc_auc_of_stress_strength_vs_" + label_name] = (float(roc_auc_score(df[label_name], df["stress_strength"]))
                                                              if df[label_name].nunique() == 2 else None)
        return blk

    dsets = [k for k in ts["dataset"].unique() if k.startswith("dreaddit")]
    for k in dsets:
        j = ts[ts["dataset"] == k].merge(ref[["record_id", "p", "label", "post_id"]], on="record_id", how="inner")
        res["datasets"][k] = one(k, j, j["post_id"], "label")
        res["datasets"][k]["n_scored"], res["datasets"][k]["n_joined"] = int((ts["dataset"] == k).sum()), int(len(j))
    scp = d / f"senticnet_scores_{bundle.name}.parquet"
    if scp.exists():
        sc = pd.read_parquet(scp)[["record_id", "p", "label_auto", "platform"]]
        j = ts[ts["dataset"] == "senticnet"].merge(sc, on="record_id", how="inner")
        for plat, g in j.groupby("platform"):
            res["datasets"][f"senticnet_{plat}"] = one(plat, g, None, "label_auto")
            res["datasets"][f"senticnet_{plat}"]["note"] = "label_auto is an automated label, so this AUC is agreement, not accuracy"
    else:
        res["senticnet_note"] = "SenticNet scores (6B.1) not found: SenticNet correlations not computed"
    res["interpretation_guard"] = ("Spearman between two scorers = agreement. A lexicon with no labelled validation data cannot "
                                   "confirm the model. Many texts get stress_strength 1 (ties), which lowers rho.")
    write_json(d / "tensistrength_result.json", res)
    return res


# ================================================================ 6B: SenticNet evaluation frame + scoring
def load_senticnet_eval(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 2's senticnet_validation.parquet restricted to rows the English model can be judged on. Step 2 already removed
    Dreaddit-contained rows, within-file duplicates and conflicting-label groups; this re-checks duplicates per PLATFORM."""
    df = pd.read_parquet(Path(cfg["_paths"]["processed"]) / "senticnet_validation.parquet")
    info: dict[str, Any] = {"rows_in_file": int(len(df))}
    cols = ["record_id", "community", "platform", "text_clean", "label_auto", "lang", "is_non_advert"]
    df = df[[c for c in cols if c in df.columns]].copy()
    if settings(cfg)["english_only"]:
        keep = df["lang"].eq("en")
        info["excluded_non_english_or_unknown"] = int((~keep).sum())
        info["excluded_by_lang"] = {str(k): int(v) for k, v in df.loc[~keep, "lang"].fillna("und").value_counts().head(8).items()}
        df = df[keep]
    df = df[df["text_clean"].str.len() > 0]
    df["_k"] = df["text_clean"].map(dedupe_key)
    conflict = df.groupby(["platform", "_k"])["label_auto"].transform("nunique") > 1
    info["dropped_conflicting_duplicate_rows"] = int(conflict.sum())
    df = df[~conflict]
    dup = df.duplicated(["platform", "_k"])
    info["dropped_duplicate_rows"] = int(dup.sum())
    df = df[~dup].drop(columns="_k").reset_index(drop=True)
    info["rows_evaluated"] = int(len(df))
    info["by_file"] = {k: {"n": int(len(g)), "prevalence_label_1": float(g["label_auto"].mean())} for k, g in df.groupby("community")}
    info["by_platform"] = {k: {"n": int(len(g)), "prevalence_label_1": float(g["label_auto"].mean())} for k, g in df.groupby("platform")}
    return df, info


def dreaddit_test_reference(cfg: dict[str, Any], info: dict[str, Any]) -> pd.DataFrame:
    """Dreaddit test predictions of the bundled model: saved logits / T from the final seed's run (no model call).
    Same numbers as FINAL_TEST_REPORT.json (checked)."""
    run_dir = SM.run_dir_for(cfg, info["model_choice"], info["seed"])
    te = SM.load_predictions(run_dir, "test", info["calibration"]["temperature"])
    final = _read_json(run_dir / "FINAL_TEST_REPORT.json")
    if final:
        thr = float(final["threshold"])
        got = binary_metrics(te["label"], te["p"], thr)["f1_stress"]
        if abs(got - final["test"]["f1_stress"]) > 1e-6:
            LOG.warning("Dreaddit test F1 recomputed from saved logits (%.6f) != FINAL_TEST_REPORT (%.6f): bundle and run differ?",
                        got, final["test"]["f1_stress"])
    return te


def score_senticnet(cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """6B.1: calibrated Step 3 P(stress) for every evaluated SenticNet row (zero-shot: no weights change). Cached per file
    under out_dir/senticnet_scores_<bundle>/; the merged table holds ids + numbers only. Resumable per file."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    bundle = latest_stress_bundle(cfg)
    info = bundle_info(bundle)
    merged = out_dir(cfg) / f"senticnet_scores_{bundle.name}.parquet"
    if merged.exists() and not force:
        return pd.read_parquet(merged)
    df, _ = load_senticnet_eval(cfg)
    cache = out_dir(cfg) / f"senticnet_scores_{bundle.name}"
    tok = model = None
    device = select_device(cfg)
    inf, ch = cfg["inference"], info["chunking"]
    parts = []
    for comm, g in df.groupby("community", sort=True):
        f = cache / f"{comm}.parquet"
        if f.exists() and not force:
            cached = pd.read_parquet(f)
            if set(cached["record_id"]) == set(g["record_id"]):
                parts.append(cached)
                continue
        if model is None:
            tok = AutoTokenizer.from_pretrained(str(bundle))
            model = AutoModelForSequenceClassification.from_pretrained(str(bundle)).to(device).eval()
        LOG.info("scoring %s (%d rows) on %s", comm, len(g), device)
        p, nch = SM.predict_chunked_proba(model, tok, g["text_clean"].tolist(), info["max_length"], ch["stride"],
                                          ch["max_chunks_per_doc"], info["calibration"]["temperature"], ch["aggregation"],
                                          int(inf["batch_size"]), device, bool(inf["half_precision"]))
        part = g.drop(columns=["text_clean", "lang"]).assign(p=p, n_chunks=nch).reset_index(drop=True)
        _atomic_parquet(part, f)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    _atomic_parquet(out, merged)
    return out


def senticnet_zero_shot(cfg: dict[str, Any]) -> dict[str, Any]:
    """6B.3: zero-shot agreement of the Step 3 model with SenticNet labels, per platform (primary) and per file, with
    bootstrap CIs and the difference to Dreaddit test. Reads the 6B.1 scores. Saves senticnet_result.json + reliability PNG."""
    ev = settings(cfg)
    n, seed, lvl, bins = ev["bootstrap_resamples"], ev["seed"], ev["ci_level"], ev["ece_bins"]
    bundle = latest_stress_bundle(cfg)
    info = bundle_info(bundle)
    thr = decision_threshold(cfg, info)
    scores = score_senticnet(cfg)
    _, frame_info = load_senticnet_eval(cfg)
    ref = dreaddit_test_reference(cfg, info)
    ref_block, ref_draws = evaluate_block(ref["label"], ref["p"], thr, ref["post_id"], n, seed, lvl, bins)
    res: dict[str, Any] = {"status": "done", "bundle": bundle.name, "model": {"base": info["base_model"], "seed": info["seed"],
                           "temperature": info["calibration"]["temperature"], "threshold": thr, "no_finetuning_on_senticnet": True},
                           "label_note": LABEL_NOTE, "frame": frame_info, "dreaddit_test": ref_block,
                           "share_texts_multi_chunk": float((scores["n_chunks"] > 1).mean()), "platforms": {}, "files": {}}
    curves = {"Dreaddit test": (ref["label"], ref["p"])}

    def run(name: str, g: pd.DataFrame) -> dict[str, Any]:
        blk, dr = evaluate_block(g["label_auto"], g["p"], thr, None, n, seed, lvl, bins)
        blk["delta_vs_dreaddit_test"] = delta_block(blk["point"], dr, ref_block["point"], ref_draws, lvl)
        blk["note_prevalence"] = ("PR-AUC of a random scorer equals the label prevalence "
                                  f"({blk['prevalence_label_1']:.3f} here, {ref_block['prevalence_label_1']:.3f} on Dreaddit test)")
        return blk

    for plat, g in scores.groupby("platform"):
        res["platforms"][plat] = run(plat, g)
        curves[f"SenticNet {plat}"] = (g["label_auto"], g["p"])
    for comm, g in scores.groupby("community"):
        res["files"][comm] = run(comm, g)
    if "is_non_advert" in scores.columns:
        g = scores[(scores["community"] == "Twitter_Full") & scores["is_non_advert"].fillna(False).astype(bool)]
        if len(g) > 50:
            res["files"]["Twitter_Full (Non-Advert rows only)"] = run("non_advert", g)
    png = out_dir(cfg) / "reliability_external.png"
    plot_reliability(curves, bins, "Reliability: Dreaddit test vs SenticNet (automated labels)", path=str(png))
    res["reliability_png"] = png.name
    res["reliability_tables"] = {k: reliability_table(y, p, bins).round(5).to_dict(orient="records") for k, (y, p) in curves.items()}
    write_json(out_dir(cfg) / "senticnet_result.json", res)
    return res


# ================================================================ 6B: label noise and the ceiling
def wilson(k: int, n: int, level: float = 0.95) -> list[float]:
    """Wilson score interval for a proportion k/n."""
    from scipy.stats import norm
    if n == 0:
        return [float("nan"), float("nan")]
    z = float(norm.ppf(1 - (1 - level) / 2))
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def noise_ceiling(ppv1: float, q0: float, prevalence_auto1: float) -> dict[str, float]:
    """Best agreement with the AUTOMATED labels that a classifier reproducing the TRUE labels perfectly would reach.
    ppv1 = P(true=1 | auto=1), q0 = P(true=1 | auto=0) (both from the hand-check, which is stratified by auto label),
    prevalence_auto1 = share of auto label 1 in the evaluated data. With a perfect binary scorer:
    TPR = ppv1, FPR = q0, AUC = (1 + TPR - FPR) / 2; accuracy and F1 follow from the same four cell probabilities."""
    pi = prevalence_auto1
    tp = pi * ppv1
    predpos = pi * ppv1 + (1 - pi) * q0
    prec = tp / predpos if predpos > 0 else float("nan")
    rec = ppv1
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and (prec + rec) > 0 else float("nan")
    return {"roc_auc": 0.5 * (1 + ppv1 - q0), "accuracy": pi * ppv1 + (1 - pi) * (1 - q0), "f1_stress": f1}


def handcheck_noise(cfg: dict[str, Any], prevalence: dict[str, float] | None = None) -> dict[str, Any]:
    """6B.2: label-noise estimate from the user's hand-check (rows with human_label 0/1; blanks = unsure are ignored), per platform,
    and the resulting ceiling on agreement. Status is PENDING until min_rows_per_platform rows are filled and every
    (platform, auto label) cell has at least 3. Saves senticnet_noise.json."""
    hc = settings(cfg)["handcheck"]
    source = hc.get("label_source", "human")
    if source not in ("human", "machine"):
        raise ValueError(f"external_validation.handcheck.label_source must be human or machine, got {source!r}")
    p = interim_dir(cfg) / ("senticnet_handcheck_machine.csv" if source == "machine" else "senticnet_handcheck.csv")
    lvl, seed = settings(cfg)["ci_level"], settings(cfg)["seed"]
    if prevalence is None:
        _, fi = load_senticnet_eval(cfg)
        prevalence = {k: v["prevalence_label_1"] for k, v in fi["by_platform"].items()}
    base: dict[str, Any] = {"source": f"{p.name} (stratified by file x auto label)", "label_source": source,
                            "min_rows_per_platform": hc["min_rows_per_platform"], "per_platform": {}}
    if source == "machine":
        base["banner"] = MACHINE_NOTE
    if not p.exists():
        res = {**base, "status": "pending",
               "reason": ("machine hand-check not found: run the machine-labelling cell (6B.1b)" if source == "machine"
                          else "hand-check sheet not found; run Step 1 (make_senticnet_handcheck)")}
        write_json(out_dir(cfg) / "senticnet_noise.json", res)
        return res
    df = pd.read_csv(p, encoding="utf-8-sig")
    if source == "machine":
        df = df.rename(columns={"machine_label": "human_label"})      # same arithmetic; provenance is carried in `label_source`
    total = len(df)
    df = df[pd.to_numeric(df["human_label"], errors="coerce").isin([0, 1])].copy()
    df["human_label"], df["platform"] = df["human_label"].astype(int), df["file"].map(platform_of)
    base.update({"n_rows_in_sheet": int(total), "n_filled": int(len(df))})
    rng = np.random.default_rng(seed)
    complete = True
    for plat, g in df.groupby("platform"):
        a, h = g["auto_label"].to_numpy(), g["human_label"].to_numpy()
        cell = lambda aa, hh: int(((a == aa) & (h == hh)).sum())   # noqa: E731
        n1, n0 = int((a == 1).sum()), int((a == 0).sum())
        agree = int((a == h).sum())
        blk: dict[str, Any] = {"n": int(len(g)), "n_auto1": n1, "n_auto0": n0,
                               "counts": {"auto1_human1": cell(1, 1), "auto1_human0": cell(1, 0),
                                          "auto0_human1": cell(0, 1), "auto0_human0": cell(0, 0)},
                               "agreement": {"point": agree / len(g), "ci": wilson(agree, len(g), lvl)},
                               "estimated_label_noise": 1 - agree / len(g)}
        ok = len(g) >= hc["min_rows_per_platform"] and n1 >= 3 and n0 >= 3 and plat in prevalence
        if ok:
            ppv1, q0 = cell(1, 1) / n1, cell(0, 1) / n0
            blk["ppv_auto1"], blk["share_true1_among_auto0"] = {"point": ppv1, "ci": wilson(cell(1, 1), n1, lvl)}, \
                {"point": q0, "ci": wilson(cell(0, 1), n0, lvl)}
            ceil = noise_ceiling(ppv1, q0, prevalence[plat])
            draws = {k: [] for k in ceil}
            ia, ib = np.flatnonzero(a == 1), np.flatnonzero(a == 0)
            for _ in range(hc["bootstrap_resamples"]):
                ra, rb = rng.choice(ia, len(ia)), rng.choice(ib, len(ib))
                c = noise_ceiling(float(h[ra].mean()), float(h[rb].mean()), prevalence[plat])
                for k in draws:
                    draws[k].append(c[k])
            blk["ceiling"] = {k: {"point": float(v), "ci": _ci(np.asarray(draws[k], float), lvl)} for k, v in ceil.items()}
            blk["ceiling"]["prevalence_auto1_used"] = prevalence[plat]
        else:
            complete = False
            blk["ceiling"] = None
            blk["pending_reason"] = (f"needs >= {hc['min_rows_per_platform']} filled rows and >= 3 per auto label; "
                                     f"have {len(g)} (auto1 {n1}, auto0 {n0})")
        base["per_platform"][plat] = blk
    if not len(df):
        base["reason"] = "no human_label filled in yet (n_filled = 0)"
    base["status"] = "complete" if complete and len(base["per_platform"]) >= 2 else "pending"
    if len(df):
        try:
            from sklearn.metrics import cohen_kappa_score
            base["cohen_kappa_all"] = float(cohen_kappa_score(df["auto_label"], df["human_label"]))
        except Exception:   # noqa: BLE001
            pass
    base["implication_template"] = ("No classifier, however good, should be expected to exceed the ceiling on agreement with these "
                                    "labels; a result at the ceiling is the best possible. The ceiling assumes the human label "
                                    "is the truth, which one annotator cannot guarantee.")
    write_json(out_dir(cfg) / "senticnet_noise.json", base)
    return base


# ================================================================ 6B: optional SenticNet -> Dreaddit cross-training (USER RUNS)
def cross_run_dir(cfg: dict[str, Any], platform: str, seed: int) -> Path:
    """<models>/stress/external_validation/cross_senticnet_<platform>/seed<k>. The folder name before seed<k> keeps the
    Trainer checkpoint dir (checkpoint_root/stress/<that name>/seed<k>) separate from the Step 3 runs."""
    return models_root(cfg) / "stress" / "external_validation" / f"cross_senticnet_{platform}" / f"seed{seed}"


def cross_training_data(cfg: dict[str, Any], platform: str) -> dict[str, pd.DataFrame]:
    """train/validation = SenticNet rows of one platform (stratified seeded split, label = AUTOMATED label); test = the
    Dreaddit test split (only predicted, never used for fitting or selection)."""
    ct = settings(cfg)["cross_training"]
    df, _ = load_senticnet_eval(cfg)
    df = df[df["platform"] == platform].copy()
    rng = np.random.default_rng(ct["seed"])
    df["split"] = "train"
    for _, idx in df.groupby("label_auto").groups.items():
        idx = np.array(list(idx))
        df.loc[rng.choice(idx, int(round(len(idx) * ct["val_fraction"])), replace=False), "split"] = "validation"
    frame = pd.DataFrame({"record_id": df["record_id"], "text_clean": df["text_clean"], "label": df["label_auto"].astype(int),
                          "confidence": 1.0, "community": df["community"], "post_id": df["record_id"], "split": df["split"],
                          "official_split": "senticnet"})
    return {"train": frame[frame["split"] == "train"].reset_index(drop=True),
            "validation": frame[frame["split"] == "validation"].reset_index(drop=True),
            "test": SM.load_dreaddit(cfg)["test"]}


def train_cross(cfg: dict[str, Any], platform: str) -> Path:
    """# >>> USER RUNS THIS CELL. DO NOT EXECUTE. <<<  Fine-tunes the Step 3 architecture on SenticNet (automated labels).
    Resume-safe through SM.train_stress_seed (epoch checkpoints, DONE.json)."""
    ct = settings(cfg)["cross_training"]
    if not ct["enabled"]:
        raise RuntimeError("external_validation.cross_training.enabled is false")
    choice = ct["model_choice"] or cfg["stress_model"]["model_choice"]
    return SM.train_stress_seed(cfg, ct["seed"], choice, data=cross_training_data(cfg, platform),
                                run_dir=cross_run_dir(cfg, platform, ct["seed"]), epochs=ct["epochs"])


def evaluate_cross(cfg: dict[str, Any], platform: str) -> dict[str, Any]:
    """After train_cross: temperature fitted on the SenticNet VALIDATION logits (source domain), threshold fixed at the bundle's,
    then Dreaddit test metrics (post-clustered CIs) next to the Step 3 model's. Saves cross_training_<platform>.json."""
    ev, ct = settings(cfg), settings(cfg)["cross_training"]
    run = cross_run_dir(cfg, platform, ct["seed"])
    if not (run / "DONE.json").exists():
        raise FileNotFoundError(f"{run} has no DONE.json: run train_cross first")
    info = bundle_info(latest_stress_bundle(cfg))
    thr = decision_threshold(cfg, info)
    va = SM.load_predictions(run, "validation")
    T = fit_temperature(va[["logit_0", "logit_1"]].to_numpy(), va["label"].to_numpy())
    te = SM.load_predictions(run, "test", T)
    n, seed, lvl, bins = ev["bootstrap_resamples"], ev["seed"], ev["ci_level"], ev["ece_bins"]
    blk, dr = evaluate_block(te["label"], te["p"], thr, te["post_id"], n, seed, lvl, bins)
    ref = dreaddit_test_reference(cfg, info)
    rblk, rdr = evaluate_block(ref["label"], ref["p"], thr, ref["post_id"], n, seed, lvl, bins)
    res = {"direction": f"SenticNet-{platform} -> Dreaddit test", "temperature_fitted_on": "SenticNet validation", "temperature": T,
           "train_label_note": LABEL_NOTE,
           "validation_in_domain": binary_metrics(va["label"], positive_proba(va[["logit_0", "logit_1"]].to_numpy(), T), thr),
           "dreaddit_test": blk, "step3_model_dreaddit_test": rblk,
           "delta_vs_step3_model": delta_block(blk["point"], dr, rblk["point"], rdr, lvl),
           "reverse_direction": "Dreaddit -> SenticNet is the zero-shot result of 6B.3 (the Step 3 model was trained on Dreaddit)."}
    write_json(out_dir(cfg) / f"cross_training_{platform}.json", res)
    return res


# ================================================================ verdict
def decide_verdict(zs: dict[str, Any] | None, noise: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the PROPOSED rules in external_validation.verdict to the measured numbers. Every observed value is read from
    the result dicts. Returns the verdict plus the evidence table (one row per condition: met / not met / unknown)."""
    v = settings(cfg)["verdict"]
    prim = v["primary_platform"]
    if not zs or prim not in zs.get("platforms", {}):
        return {"verdict": "not available", "reason": "zero-shot SenticNet evaluation (6B.3) has not been run", "rows": []}
    P = zs["platforms"][prim]
    others = [k for k in zs["platforms"] if k != prim]
    S = zs["platforms"][others[0]] if others else None
    auc, (alo, ahi) = P["point"]["roc_auc"], P["ci"]["roc_auc"]
    eceP = P["point"]["ece"]
    dlt = P["delta_vs_dreaddit_test"]["roc_auc"]
    ceil = ((noise or {}).get("per_platform", {}).get(prim) or {}).get("ceiling") if (noise or {}).get("status") == "complete" else None
    skill = skill_lo = skill_hi = None
    if ceil and ceil["roc_auc"]["point"] > 0.5 and ceil["roc_auc"]["ci"][0] > 0.5:
        c, (clo, chi) = ceil["roc_auc"]["point"], ceil["roc_auc"]["ci"]
        skill, skill_lo, skill_hi = (auc - 0.5) / (c - 0.5), (alo - 0.5) / (chi - 0.5), (ahi - 0.5) / (clo - 0.5)

    def row(rid: str, level: str, text: str, observed: str, met: bool | None) -> dict[str, Any]:
        return {"id": rid, "level": level, "condition": text, "observed": observed, "met": met}

    f = lambda x: "n/a" if x is None else f"{x:.3f}"   # noqa: E731
    rows = [
        row("R1", "retrain", f"{prim} ROC-AUC CI upper bound < {v['retrain_if_auc_ci_upper_below']}",
            f"AUC {f(auc)} [{f(alo)}, {f(ahi)}]", ahi < v["retrain_if_auc_ci_upper_below"]),
        row("R2", "retrain", f"ceiling-relative skill CI upper bound < {v['retrain_if_skill_ci_upper_below']}",
            f"skill {f(skill)} [{f(skill_lo)}, {f(skill_hi)}]" if skill is not None else "label-noise ceiling not available",
            None if skill is None else skill_hi < v["retrain_if_skill_ci_upper_below"]),
        row("S1", "ship", f"{prim} ROC-AUC CI lower bound >= {v['ship_min_auc_ci_lower']}", f"lower bound {f(alo)}", alo >= v["ship_min_auc_ci_lower"]),
        row("S2", "ship", f"label-noise ceiling known and skill >= {v['ship_min_skill']}",
            f"skill {f(skill)}" if skill is not None else "label-noise ceiling not available (hand-check pending)",
            None if skill is None else skill >= v["ship_min_skill"]),
        row("S3", "ship", f"{prim} ECE <= {v['ship_max_ece']} (else read probabilities as relative, or recalibrate)",
            f"ECE {f(eceP)}", eceP <= v["ship_max_ece"]),
        row("S4", "ship", f"{others[0] if others else 'other platform'} ROC-AUC point >= {v['ship_min_secondary_auc']}",
            f"AUC {f(S['point']['roc_auc'])}" if S else "not available", None if S is None else S["point"]["roc_auc"] >= v["ship_min_secondary_auc"]),
        row("S5", "ship", "the hand-check behind the ceiling was labelled by a human",
            f"label source: {(noise or {}).get('label_source', 'n/a')}", (noise or {}).get("label_source") == "human"),
        row("C1", "caveat", f"drop in ROC-AUC vs Dreaddit test is not clearly larger than {v['caveat_if_auc_drop_ci_lower_above']}",
            f"delta {f(dlt['delta'])} [{f(dlt['ci'][0])}, {f(dlt['ci'][1])}]", not (-dlt["ci"][1] > v["caveat_if_auc_drop_ci_lower_above"])),
    ]
    retrain = any(r["met"] is True for r in rows if r["level"] == "retrain")
    ship = all(r["met"] is True for r in rows if r["level"] in ("ship", "caveat"))
    verdict = "retrain" if retrain else "ship" if ship else "ship with caveats"
    caveats = [f"{r['id']}: {r['condition']} -> {r['observed']}" for r in rows if r["level"] != "retrain" and r["met"] is not True]
    machine = (noise or {}).get("label_source") == "machine"
    prov = ceil is None or machine
    reason = ("the label-noise ceiling needs the completed hand-check; until then 'ship' cannot be reached" if ceil is None else
              "the hand-check was labelled by a machine, not a human; 'ship' cannot be reached until a human has done it" if machine else None)
    return {"verdict": verdict, "provisional": prov, "provisional_reason": reason,
            "caveats": caveats, "rows": rows, "primary_platform": prim, "rules": v,
            "rules_source": "proposed by the assistant, not from the literature; edit config.yaml before reading results",
            "skill": {"point": skill, "ci": [skill_lo, skill_hi], "definition": "(AUC - 0.5) / (ceiling AUC - 0.5); CI by conservative interval propagation"}}


# ================================================================ report
def _f(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _ci_s(ci: Sequence[float] | None) -> str:
    return "n/a" if not ci else f"[{_f(ci[0])}, {_f(ci[1])}]"


def render_markdown(rep: dict[str, Any]) -> str:
    """The .md is rendered from the JSON dict, so no number is typed by hand."""
    L = ["# External validation (Step 6)", "",
         "> Aggregate, population-level language signals only. No diagnosis. No individual-level analysis.", "",
         f"Generated {rep['created_utc']} for stress bundle `{rep['bundle']}`. {LABEL_NOTE}", ""]
    v = rep["verdict"]
    L += ["## Verdict (proposed rules, see config)", "", f"**{v['verdict'].upper()}**" + (" (PROVISIONAL)" if v.get("provisional") else ""), ""]
    if v.get("provisional_reason"):
        L += [f"- Provisional because {v['provisional_reason']}."]
    L += [f"- Rules: {v.get('rules_source', '')}.", ""]
    if v["rows"]:
        L += ["| id | level | condition | observed | met |", "|---|---|---|---|---|"]
        L += [f"| {r['id']} | {r['level']} | {r['condition']} | {r['observed']} | {'unknown' if r['met'] is None else 'yes' if r['met'] else 'no'} |"
              for r in v["rows"]]
        L += [""]
    if v.get("caveats"):
        L += ["Caveats triggered:", *[f"- {c}" for c in v["caveats"]], ""]
    L += ["### What each verdict means", "",
          "| verdict | evidence required |", "|---|---|",
          "| ship | no retrain condition met AND every ship condition (S1-S5) and caveat check (C1) met, with a completed hand-check |",
          "| ship with caveats | no retrain condition met, but at least one ship condition not met or unknown; the caveats are listed above |",
          "| retrain | any retrain condition (R1, R2) met: the model carries too little rank signal on the primary platform |", ""]
    zs = rep["senticnet_zero_shot"]
    L += ["## 6B. Zero-shot agreement with SenticNet labels (no fine-tuning)", ""]
    if zs.get("status") != "done":
        L += ["NOT RUN: " + str(zs.get("reason", "scores missing")), ""]
    else:
        m = zs["model"]
        L += [f"Model: {m['base']} seed {m['seed']}, T = {_f(m['temperature'])}, threshold {_f(m['threshold'])}. Rows evaluated: "
              f"{zs['frame']['rows_evaluated']} of {zs['frame']['rows_in_file']} (excluded non-English: "
              f"{zs['frame'].get('excluded_non_english_or_unknown', 0)}, duplicates: {zs['frame']['dropped_duplicate_rows']}).", "",
              "| set | n | prevalence of label 1 | F1 (stress) | ROC-AUC | PR-AUC | ECE |", "|---|---|---|---|---|---|---|"]

        def line(name: str, b: dict[str, Any]) -> str:
            p, c = b["point"], b["ci"]
            return (f"| {name} | {b['n']} | {_f(b['prevalence_label_1'])} | {_f(p['f1_stress'])} {_ci_s(c['f1_stress'])} | "
                    f"{_f(p['roc_auc'])} {_ci_s(c['roc_auc'])} | {_f(p['pr_auc'])} {_ci_s(c['pr_auc'])} | {_f(p['ece'])} {_ci_s(c['ece'])} |")
        L += [line("Dreaddit test (reference)", zs["dreaddit_test"])]
        L += [line(f"SenticNet {k}", b) for k, b in zs["platforms"].items()]
        L += [line(k, b) for k, b in zs["files"].items()]
        L += ["", "Difference to Dreaddit test (SenticNet minus Dreaddit; negative = lower on SenticNet), 95% CI:", "",
              "| set | delta F1 | delta ROC-AUC | delta PR-AUC | delta ECE |", "|---|---|---|---|---|"]
        for k, b in {**zs["platforms"], **zs["files"]}.items():
            d = b["delta_vs_dreaddit_test"]
            L += [f"| {k} | " + " | ".join(f"{_f(d[x]['delta'])} {_ci_s(d[x]['ci'])}" for x in ("f1_stress", "roc_auc", "pr_auc", "ece")) + " |"]
        L += ["", "Differences mix domain shift, label noise, prevalence and length; F1 and PR-AUC move with prevalence. ", ""]
    nz = rep["label_noise"]
    L += ["## 6B. Label noise and the ceiling on agreement", ""]
    if (nz or {}).get("label_source") == "machine":
        L += [f"> **{MACHINE_NOTE}**", ""]
    if not nz or nz.get("status") != "complete":
        L += [f"**PENDING.** {(nz or {}).get('reason') or 'the hand-check is not filled in enough'}. Rows filled: "
              f"{(nz or {}).get('n_filled', 0)} of {(nz or {}).get('n_rows_in_sheet', '?')}. "
              "Until it is, no ceiling can be stated and the verdict stays provisional.", ""]
        for k, b in (nz or {}).get("per_platform", {}).items():
            L += [f"- {k}: {b['n']} rows filled; {b.get('pending_reason', '')}"]
    else:
        L += ["| platform | n | human-auto agreement | est. label noise | ceiling ROC-AUC | ceiling accuracy | ceiling F1 |", "|---|---|---|---|---|---|---|"]
        for k, b in nz["per_platform"].items():
            c = b["ceiling"]
            L += [f"| {k} | {b['n']} | {_f(b['agreement']['point'])} {_ci_s(b['agreement']['ci'])} | {_f(b['estimated_label_noise'])} | "
                  f"{_f(c['roc_auc']['point'])} {_ci_s(c['roc_auc']['ci'])} | {_f(c['accuracy']['point'])} {_ci_s(c['accuracy']['ci'])} | "
                  f"{_f(c['f1_stress']['point'])} {_ci_s(c['f1_stress']['ci'])} |"]
        L += ["", nz["implication_template"], ""]
    ts = rep["tensistrength"]
    L += ["## 6A. TensiStrength auxiliary signal", ""]
    if ts.get("status") != "done":
        L += ["NOT RUN: " + "; ".join(map(str, ts.get("reason", ["no reason recorded"]))), ""]
    else:
        L += ["| set | n | Spearman(stress strength, P(stress)) | Spearman(relaxation, P(stress)) | share with any stress term |", "|---|---|---|---|---|"]
        for k, b in ts["datasets"].items():
            s, r = b["spearman_stress_vs_p_stress"], b["spearman_relaxation_vs_p_stress"]
            L += [f"| {k} | {b['n']} | {_f(s['rho'])} {_ci_s(s['ci'])} | {_f(r['rho'])} {_ci_s(r['ci'])} | {_f(b['share_texts_with_any_stress_term'])} |"]
        L += ["", ts["interpretation_guard"], ""]
    run = rep["tensistrength_run"]
    if run.get("status") == "done":
        L += ["Speed on this machine: " + "; ".join(f"{k}: {t['n_texts']} texts in {t['seconds']} s ({t['texts_per_second']}/s) in one Java process"
                                                   for k, t in run["timings"].items()) +
              f"; one Java launch per text: {run['probe']['mean_seconds_per_single_launch']} s each.", ""]
    L += ["### Why not run TensiStrength on every record", "", *[f"- {w}" for w in WHY_NOT_AT_SCALE], ""]
    cts = rep.get("cross_training") or {}
    L += ["## Optional: cross-training", ""]
    if not cts:
        L += ["NOT RUN (config external_validation.cross_training.enabled is false, or no results yet).",
              "Dreaddit -> SenticNet is the zero-shot result above.", ""]
    for k, b in cts.items():
        d = b["dreaddit_test"]
        L += [f"- {b['direction']}: F1 {_f(d['point']['f1_stress'])} {_ci_s(d['ci']['f1_stress'])}, ROC-AUC {_f(d['point']['roc_auc'])} "
              f"{_ci_s(d['ci']['roc_auc'])}, versus the Step 3 model on the same test set (see JSON)."]
    L += ["", "## Limitations", "", *[f"- {x}" for x in LIMITATIONS], ""]
    return "\n".join(L)


def build_report(cfg: dict[str, Any]) -> dict[str, Any]:
    """6C: assemble external_validation.json/.md (paths.reports) from the saved part files and write the manifest.
    Parts that were not run are written as NOT RUN with the reason; nothing is invented."""
    d = out_dir(cfg)
    bundle = latest_stress_bundle(cfg)
    zs = _read_json(d / "senticnet_result.json")
    noise = _read_json(d / "senticnet_noise.json")
    rep = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "bundle": bundle.name,
           "label_note": LABEL_NOTE, "limitations": LIMITATIONS,
           "senticnet_zero_shot": zs or {"status": "not_run", "reason": "6B.3 has not been run"},
           "label_noise": noise or {"status": "pending", "reason": "6B.2 has not been run"},
           "tensistrength_run": _read_json(d / "tensistrength_run.json") or {"status": "not_run"},
           "tensistrength": _read_json(d / "tensistrength_result.json") or {"status": "not_run", "reason": ["6A.2 has not been run"]},
           "cross_training": {p.stem.replace("cross_training_", ""): _read_json(p) for p in sorted(d.glob("cross_training_*.json"))},
           "verdict": decide_verdict(zs, noise, cfg), "config_used": settings(cfg)}
    reports = Path(cfg["_paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    jp, mp = reports / "external_validation.json", reports / "external_validation.md"
    write_json(jp, rep)
    mp.write_text(render_markdown(rep), encoding="utf-8")
    parts = [p for p in (d / "senticnet_result.json", d / "senticnet_noise.json", d / "tensistrength_run.json",
                         d / "tensistrength_result.json", d / "tensistrength_scores.parquet",
                         d / f"senticnet_scores_{bundle.name}.parquet", jp, mp) if p.exists()]
    write_manifest(Path(cfg["_paths"]["manifests"]) / "step6_external_validation_manifest.json", artifacts=parts,
                   seeds=[settings(cfg)["seed"]],
                   dataset_versions={"senticnet_validation_sha256": sha256_file(Path(cfg["_paths"]["processed"]) / "senticnet_validation.parquet"),
                                     "stress_bundle_manifest_sha256": sha256_file(bundle / "manifest.json")},
                   extra={"step": 6, "bundle": bundle.name, "verdict": rep["verdict"]["verdict"],
                          "libraries": library_versions()}, root=Path(cfg["_root"]))
    return rep
