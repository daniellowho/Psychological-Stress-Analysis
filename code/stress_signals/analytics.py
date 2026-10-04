"""Step 10: analytics over the enriched Parquet (temporal, composition-adjusted, emotions, stressors, topics). NOTHING here trains a classifier.
The only fitted model is the UNSUPERVISED topic model (BERTopic), and `fit_topic_model` is code the user runs; it is never executed by the assistant.

Output: aggregate Parquet tables in `analytics.gold_dir` (default data/gold). Every table passes `write_gold`, which
  * refuses columns that could identify a record or hold text (record_id, text, author, ...), and
  * refuses any published count below privacy.k_min (cells with n < k_min are blanked and flagged `suppressed=True` before writing).
Suppression rule: the "cell" is the group whose size controls precision, i.e. the denominator (n_total, n_stress_scored, ...); topic and
stressor-share rows are also blanked when the numerator is the cell (topic_series). Rows stay in the table with `suppressed=True`, values blank.
Complementary suppression is NOT applied: the tables are not additive totals meant to be differenced, but nobody should difference them to
recover a blanked cell; state this when you share them.

Why pandas/pyarrow and not Spark for the aggregates: the inputs are narrow (a dozen columns, column-pruned), the enriched files are read with pyarrow
because Spark's own reader needs Hadoop's winutils on Windows (see spark_jobs), and Spark adds nothing for a group-by this size. Spark is used where it helps:
batch topic assignment (`assign_topics_spark`, mapInPandas over dates, like Step 9) and the MLlib LDA alternative (`fit_lda_mllib`).

Scored records only: rows whose `input_status` is not ok / ok_truncated have no probabilities and are excluded from every count.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from .utils import get_logger, sha256_file, write_json

LOG = get_logger("stress_signals.analytics")
SCORED = ("ok", "ok_truncated")
Z95 = 1.959963984540054
FREQ = {"day": "1D", "week": "7D", "month": "MS"}
FORBIDDEN_COLUMNS = re.compile(r"(?i)^(record_id|id|text|text_clean|texts?_raw|body|selftext|title|author\w*|user\w*|username|post_id|doc\w*|representative\w*)$")
DOMAIN_SHIFT_NOTE = (
    "Emotion, stressor and stress models were trained on other text: GoEmotions (Reddit comments), SAD (short SMS-like messages) and Dreaddit "
    "(segments from 10 subreddits). Applied to submissions in other communities, an association (emotion lift, stressor x emotion) can reflect how a "
    "model behaves on that text rather than the text itself. `community_agreement` (share of communities whose lift is on the same side of 1 as the "
    "overall lift) is a rough guide only: agreement does not rule out a shared shift. Treat associations as hypotheses, not findings.")


# ================================================================ small statistics
def wilson(k: Any, n: Any, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for k successes in n trials (vectorised). NaN where n == 0."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def period_start(dates: pd.Series, gran: str) -> pd.Series:
    """Start of the day / ISO week (Monday) / month containing each date (UTC dates)."""
    d = pd.to_datetime(dates).dt.normalize()
    if gran == "day":
        return d
    if gran == "week":
        return d - pd.to_timedelta(d.dt.weekday, unit="D")
    if gran == "month":
        return d.dt.to_period("M").dt.to_timestamp()
    raise ValueError(f"granularity must be day | week | month, got {gran!r}")


def period_grid(lo: pd.Timestamp, hi: pd.Timestamp, gran: str) -> pd.DatetimeIndex:
    return pd.date_range(lo, hi, freq=FREQ[gran])


def suppress(df: pd.DataFrame, n_col: str, k_min: int, blank: Iterable[str], extra_mask: pd.Series | None = None) -> pd.DataFrame:
    """Blank `blank` columns (and n_col) where df[n_col] < k_min (or extra_mask) and set suppressed=True. Integer columns become nullable Int64."""
    df = df.copy()
    mask = df[n_col] < k_min
    if extra_mask is not None:
        mask = mask | extra_mask
    df["suppressed"] = mask.to_numpy()
    for c in {n_col, *blank}:
        if c not in df.columns:
            continue
        if pd.api.types.is_integer_dtype(df[c]) or c == n_col:
            df[c] = df[c].astype("Int64").mask(mask)
        else:
            df[c] = df[c].mask(mask)
    return df


def check_aggregate_only(df: pd.DataFrame, k_min: int, count_cols: Iterable[str] = ()) -> None:
    """Raise if the table could hold identifiers/text or a published count is below k_min."""
    bad = [c for c in df.columns if FORBIDDEN_COLUMNS.match(str(c))]
    if bad:
        raise ValueError(f"aggregate table has forbidden columns {bad}")
    for c in df.columns:
        if df[c].dtype == object and len(df) and df[c].dropna().astype(str).str.len().max() > 200:
            raise ValueError(f"column {c!r} holds strings longer than 200 characters: looks like text")
    for c in count_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if ((v < k_min) & v.notna()).any():
            raise ValueError(f"column {c!r} publishes a count below k_min={k_min}")


def gold_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_root"]) / cfg["analytics"]["gold_dir"]


def write_gold(cfg: dict[str, Any], name: str, df: pd.DataFrame, count_cols: Iterable[str] = ()) -> Path:
    check_aggregate_only(df, int(cfg["privacy"]["k_min"]), count_cols)
    p = gold_dir(cfg) / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), tmp, compression="zstd")
    tmp.replace(p)
    return p


# ================================================================ loading
def _partitioning() -> Any:
    return pds.partitioning(pa.schema([("source", pa.string()), ("date", pa.date32())]), flavor="hive")


def load_enriched(root: str | Path, columns: Iterable[str] | None = None, prefixes: Iterable[str] = ()) -> pd.DataFrame:
    """Enriched rows with only the wanted columns (names in `columns` + every column starting with one of `prefixes`). Scored records only."""
    ds = pds.dataset(str(root), format="parquet", partitioning=_partitioning(), ignore_prefixes=["_", "."])
    names = ds.schema.names
    want = [c for c in names if (columns is None or c in set(columns)) or any(c.startswith(p) for p in prefixes)]
    if "input_status" not in want:
        want.append("input_status")
    df = ds.to_table(columns=want).to_pandas()
    df = df[df["input_status"].isin(SCORED)].drop(columns="input_status").reset_index(drop=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def enriched_dir(cfg: dict[str, Any], corpus: str) -> Path:
    from . import spark_jobs as SJ
    return SJ.enriched_root(cfg, corpus)


# ================================================================ 1. temporal aggregates
def stress_cells(df: pd.DataFrame, gran: str) -> pd.DataFrame:
    """UNSUPPRESSED counts per period x source x community, plus rollups (community=ALL per source; source=ALL, community=ALL). Internal use only."""
    d = pd.DataFrame({"period": period_start(df["date"], gran), "source": df["source"].astype(str), "community": df["community"].astype(str),
                      "n_stress": df["stress_flag"].astype(int), "prob_sum": df["stress_prob"].astype(float)})
    keys = ["period", "source", "community"]
    base = d.groupby(keys, sort=True).agg(n_total=("n_stress", "size"), n_stress=("n_stress", "sum"), prob_sum=("prob_sum", "sum")).reset_index()
    by_src = base.groupby(["period", "source"], as_index=False)[["n_total", "n_stress", "prob_sum"]].sum().assign(community="ALL")
    allr = base.groupby("period", as_index=False)[["n_total", "n_stress", "prob_sum"]].sum().assign(source="ALL", community="ALL")
    return pd.concat([base, by_src[base.columns], allr[base.columns]], ignore_index=True).sort_values(keys, kind="stable").reset_index(drop=True)


def stress_rate_table(cells: pd.DataFrame, k_min: int) -> pd.DataFrame:
    """n_total, n_stress, mean calibrated probability, stress rate with Wilson 95% CI; cells with n_total < k_min suppressed."""
    t = cells.copy()
    t["mean_stress_prob"] = t["prob_sum"] / t["n_total"]
    t["stress_rate"] = t["n_stress"] / t["n_total"]
    t["ci_low"], t["ci_high"] = wilson(t["n_stress"], t["n_total"])
    t = t.drop(columns="prob_sum")
    return suppress(t, "n_total", k_min, ["n_stress", "mean_stress_prob", "stress_rate", "ci_low", "ci_high"])


# ================================================================ 2. composition-adjusted series
def reference_weights(df: pd.DataFrame, ref: dict[str, Any], k_min: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """Weights = share of records per (source, community) stratum inside the reference window, as a table with `suppressed` for strata with
    n < k_min (those get weight 0 and the rest are renormalised in `adjusted_series`). Window: [start, end] if given, else the first `first_n_weeks`."""
    dates = pd.to_datetime(df["date"])
    start = pd.Timestamp(ref["start"]) if ref.get("start") else dates.min()
    end = pd.Timestamp(ref["end"]) if ref.get("end") else start + pd.Timedelta(days=7 * int(ref.get("first_n_weeks", 8))) - pd.Timedelta(days=1)
    m = (dates >= start) & (dates <= end)
    w = df[m].groupby(["source", "community"]).size().rename("n_ref").reset_index()
    w["source"], w["community"] = w["source"].astype(str), w["community"].astype(str)
    w["usable"] = w["n_ref"] >= k_min
    w["weight_all"] = np.where(w["usable"], w["n_ref"], 0) / max(int(w.loc[w["usable"], "n_ref"].sum()), 1)
    tot_src = w[w["usable"]].groupby("source")["n_ref"].transform("sum")
    w["weight_in_source"] = np.where(w["usable"], w["n_ref"] / tot_src.reindex(w.index), 0.0)
    return w, {"start": str(start.date()), "end": str(end.date())}


def adjusted_series(cells: pd.DataFrame, weights: pd.DataFrame, k_min: int, min_coverage: float = 0.8) -> pd.DataFrame:
    """Fixed-weight (direct standardisation) stress rate: sum_s w_s * rate_s over strata s = (source, community) with n >= k_min in that period,
    renormalised over the strata present; `coverage` = share of reference weight those strata carry; adjusted value is NaN below `min_coverage`.
    CI: normal approximation of the weighted sum, se = sqrt(sum (w_s/coverage)^2 p_s (1-p_s) / n_s) (approximate, strata independent).
    Raw series = the plain pooled rate of the same scope, shown next to it. Scopes: ALL sources, and each source."""
    base = cells[(cells["community"] != "ALL")].copy()
    out = []
    for scope in ["ALL", *sorted(base["source"].unique())]:
        wcol = "weight_all" if scope == "ALL" else "weight_in_source"
        cs = base if scope == "ALL" else base[base["source"] == scope]
        raw = cs.groupby("period").agg(n_total=("n_total", "sum"), n_stress=("n_stress", "sum"))
        ok = cs[cs["n_total"] >= k_min].merge(weights[["source", "community", wcol]], on=["source", "community"], how="left").fillna({wcol: 0.0})
        ok = ok[ok[wcol] > 0].copy()
        ok["p"] = ok["n_stress"] / ok["n_total"]
        ok["wp"], ok["w"] = ok[wcol] * ok["p"], ok[wcol]
        g = ok.groupby("period")
        cov = g["w"].sum()
        est = g["wp"].sum() / cov
        ok["v"] = (ok["w"] ** 2) * ok["p"] * (1 - ok["p"]) / ok["n_total"]
        se = np.sqrt(g["v"].sum()) / cov
        t = raw.join(pd.DataFrame({"coverage": cov, "adjusted_rate": est, "se": se, "n_strata": g.size()}), how="left").reset_index()
        t["coverage"] = t["coverage"].fillna(0.0)
        bad = t["coverage"] < min_coverage
        t.loc[bad, ["adjusted_rate", "se"]] = np.nan
        t["adj_ci_low"], t["adj_ci_high"] = np.clip(t["adjusted_rate"] - Z95 * t["se"], 0, 1), np.clip(t["adjusted_rate"] + Z95 * t["se"], 0, 1)
        t["raw_rate"] = t["n_stress"] / t["n_total"]
        t["raw_ci_low"], t["raw_ci_high"] = wilson(t["n_stress"], t["n_total"])
        t["scope"] = scope
        out.append(t.drop(columns="se"))
    r = pd.concat(out, ignore_index=True)
    r["n_strata"] = r["n_strata"].fillna(0).astype(int)
    r = suppress(r, "n_total", k_min, ["n_stress", "raw_rate", "raw_ci_low", "raw_ci_high", "adjusted_rate", "adj_ci_low", "adj_ci_high", "coverage"])
    return r[["period", "scope", "n_total", "n_stress", "raw_rate", "raw_ci_low", "raw_ci_high", "adjusted_rate", "adj_ci_low", "adj_ci_high",
              "coverage", "n_strata", "suppressed"]].sort_values(["scope", "period"]).reset_index(drop=True)


# ================================================================ 3. emotions
def emotion_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    return {"label": [c for c in df.columns if c.startswith("emotion_p__")], "ekman": [c for c in df.columns if c.startswith("emotion_group__")]}


def _present(df: pd.DataFrame, cols: list[str], thr: float) -> np.ndarray:
    return (df[cols].to_numpy(dtype=np.float32) >= thr).astype(np.float32)


def _wprev(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float32)
    return (w @ X).astype(float) / float(w.sum())


def _lift_block(Xf: np.ndarray, wf: np.ndarray, Xu: np.ndarray, wu: np.ndarray, B: int, rng: np.random.Generator, alpha: float) -> dict[str, np.ndarray]:
    """Weighted prevalence in each group, lift = flagged / unflagged, and a Poisson-bootstrap percentile CI for the lift (records resampled within group)."""
    pf, pu = _wprev(Xf, wf), _wprev(Xu, wu)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = pf / pu
        boots = np.empty((B, Xf.shape[1]))
        for b in range(B):
            a = (rng.poisson(1.0, len(wf)) * wf).astype(np.float32)
            c = (rng.poisson(1.0, len(wu)) * wu).astype(np.float32)
            boots[b] = ((a @ Xf) / a.sum()) / ((c @ Xu) / c.sum())
    boots = np.where(np.isfinite(boots), boots, np.nan)
    lo, hi = (np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0) if B else (np.full_like(lift, np.nan),) * 2)
    return {"prev_flagged": pf, "prev_unflagged": pu, "lift": lift, "lift_ci_low": lo, "lift_ci_high": hi}


def emotion_lift(df: pd.DataFrame, cfg: dict[str, Any], per_community: bool = True) -> pd.DataFrame:
    """Prevalence of each emotion among stress-flagged vs not-flagged records, lift = prevalence(flagged) / prevalence(not flagged), bootstrap CI.
    Only second-stage-scored records have emotions. The unflagged group is a random sample, so every record is weighted by `sample_weight`
    (1 for flagged, 1/sample_rate for sampled unflagged). A row is suppressed unless BOTH groups have >= k_min scored records (unweighted)."""
    a, k_min = cfg["analytics"], int(cfg["privacy"]["k_min"])
    thr, bs = float(a["emotions"]["threshold"]), a["emotions"]["bootstrap"]
    e = df[df["second_stage_scored"].astype(bool) & df["sample_weight"].notna()]
    rng = np.random.default_rng(int(bs["seed"]))
    levels = emotion_columns(e)
    scopes = [("ALL", e)] + ([(str(c), g) for c, g in e.groupby("community", sort=True)] if per_community else [])
    rows = []
    for scope, g in scopes:
        f, u = g[g["stress_flag"].astype(bool)], g[~g["stress_flag"].astype(bool)]
        for level, cols in levels.items():
            if not cols:
                continue
            row0 = {"scope": scope, "level": level, "n_flagged": len(f), "n_unflagged_sampled": len(u), "n_unflagged_weighted": float(u["sample_weight"].sum())}
            if len(f) < k_min or len(u) < k_min:
                rows += [{**row0, "emotion": c.split("__", 1)[1], "suppressed": True} for c in cols]
                continue
            r = _lift_block(_present(f, cols, thr), f["sample_weight"].to_numpy(float), _present(u, cols, thr), u["sample_weight"].to_numpy(float),
                            int(bs["n"]), rng, float(bs["alpha"]))
            rows += [{**row0, "emotion": c.split("__", 1)[1], "suppressed": False, **{k: float(v[i]) for k, v in r.items()}} for i, c in enumerate(cols)]
    if not rows:
        return pd.DataFrame(columns=["scope", "level", "emotion", "n_flagged", "n_unflagged_sampled", "n_unflagged_weighted", "suppressed", "community_agreement"])
    t = pd.DataFrame(rows)
    t["community_agreement"] = np.nan
    allr = t[(t["scope"] == "ALL") & ~t["suppressed"]]
    comm = t[(t["scope"] != "ALL") & ~t["suppressed"] & t["lift"].notna()]
    for i, r in allr.iterrows():
        c = comm[(comm["level"] == r["level"]) & (comm["emotion"] == r["emotion"])]
        if len(c) and pd.notna(r["lift"]):
            t.loc[i, "community_agreement"] = float(((c["lift"] > 1) == (r["lift"] > 1)).mean())
    for c in ("n_flagged", "n_unflagged_sampled"):
        t[c] = t[c].astype("Int64").mask(t["suppressed"])
    t["n_unflagged_weighted"] = t["n_unflagged_weighted"].mask(t["suppressed"])
    return t


def stressor_emotion_cooccurrence(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Among stress-flagged, second-stage-scored records: for each stressor flag, P(emotion present | stressor) with Wilson CI, P(emotion | any flagged
    record) and lift = ratio. Rows with fewer than k_min records carrying the stressor are suppressed. Flagged records all have weight 1."""
    k_min, thr = int(cfg["privacy"]["k_min"]), float(cfg["analytics"]["emotions"]["threshold"])
    f = df[df["second_stage_scored"].astype(bool) & df["stress_flag"].astype(bool)]
    scols = [c for c in f.columns if c.startswith("stressor_flag__")]
    rows = []
    for level, cols in emotion_columns(f).items():
        X = _present(f, cols, thr) if len(f) else np.zeros((0, len(cols)), np.float32)
        base = X.mean(axis=0) if len(f) else np.full(len(cols), np.nan)
        for sc in scols:
            m = f[sc].to_numpy(bool)
            n = int(m.sum())
            ks = X[m].sum(axis=0) if n else np.zeros(len(cols))
            lo, hi = wilson(ks, n)
            for i, c in enumerate(cols):
                rows.append({"stressor": sc.split("__", 1)[1], "level": level, "emotion": c.split("__", 1)[1], "n_stressor": n, "prev_given_stressor": ks[i] / n if n else np.nan,
                             "ci_low": lo[i], "ci_high": hi[i], "prev_all_flagged": base[i], "lift": (ks[i] / n) / base[i] if n and base[i] > 0 else np.nan})
    t = pd.DataFrame(rows)
    return suppress(t, "n_stressor", k_min, ["prev_given_stressor", "ci_low", "ci_high", "prev_all_flagged", "lift"])


# ================================================================ 4. stressors over time
def stressor_share(df: pd.DataFrame, gran: str, k_min: int) -> pd.DataFrame:
    """Per period x community (+ ALL): among stress-flagged, second-stage-scored records, the share carrying each stressor flag (multi-label, may sum
    to more than 1) and the share whose top stressor it is (sums to 1), with Wilson CIs. Suppressed when the denominator n_stress_scored < k_min."""
    f = df[df["second_stage_scored"].astype(bool) & df["stress_flag"].astype(bool)].copy()
    f["period"], f["community"] = period_start(f["date"], gran), f["community"].astype(str)
    scols = [c for c in f.columns if c.startswith("stressor_flag__")]
    keys = ["period", "community"]

    def roll(x: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([x, x.assign(community="ALL")], ignore_index=True)

    fl = roll(f[keys + scols])
    den = fl.groupby(keys).size().rename("n_stress_scored")
    flag = fl.groupby(keys)[scols].sum().stack().rename("n_flag").reset_index().rename(columns={"level_2": "stressor"})
    flag["stressor"] = flag["stressor"].str.split("__", n=1).str[1]
    top = roll(f[keys + ["stressor_top"]].assign(stressor_top=f["stressor_top"].fillna("none_unclear"))).groupby(keys + ["stressor_top"]).size().rename("n_top").reset_index()
    top = top.rename(columns={"stressor_top": "stressor"})
    t = flag.merge(top, on=keys + ["stressor"], how="outer").merge(den.reset_index(), on=keys, how="left").fillna({"n_flag": 0, "n_top": 0})
    t["share_flag"], t["share_top"] = t["n_flag"] / t["n_stress_scored"], t["n_top"] / t["n_stress_scored"]
    t["flag_ci_low"], t["flag_ci_high"] = wilson(t["n_flag"], t["n_stress_scored"])
    t["top_ci_low"], t["top_ci_high"] = wilson(t["n_top"], t["n_stress_scored"])
    t = t.astype({"n_flag": "int64", "n_top": "int64"})
    return suppress(t, "n_stress_scored", k_min, ["n_flag", "n_top", "share_flag", "share_top", "flag_ci_low", "flag_ci_high", "top_ci_low", "top_ci_high"]
                    ).sort_values(["community", "stressor", "period"]).reset_index(drop=True)


# ================================================================ 5. topics: sample, fit (USER RUNS), assign at scale
def topic_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["analytics"]["topics"]


def _u01(ids: pd.Series, seed: int) -> np.ndarray:
    """Deterministic uniform(0,1) per record id (independent of batching/order)."""
    h = pd.util.hash_pandas_object(ids.astype(str) + f"|{seed}", index=False).to_numpy(dtype=np.uint64)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _restricted_files(cfg: dict[str, Any], corpus: str) -> list[Path]:
    from . import spark_jobs as SJ
    root = SJ.restricted_root(cfg, corpus)
    files = sorted(root.glob("source=*/date=*/part-00000.parquet")) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"no restricted text under {root}: set spark.keep_text_for_topics: true and re-run the Step 9 job (the text stays local, never in gold/Postgres/dashboard)")
    return files


def _partition_key(p: Path) -> tuple[str, str]:
    return p.parent.parent.name.split("=", 1)[1], p.parent.name.split("=", 1)[1]


def sample_flagged_text(cfg: dict[str, Any], corpus: str, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Uniform seeded sample of `n` stress-flagged, scored records with their cleaned text (columns record_id, community, date, text_clean).
    LOCAL USE ONLY: the frame holds verbatim text; it is for embedding and is never written anywhere by this module."""
    ts = topic_settings(cfg)
    n, seed = int(n or ts["sample_size"]), int(seed if seed is not None else ts["sample_seed"])
    meta = load_enriched(enriched_dir(cfg, corpus), ["record_id", "community", "date", "stress_flag"])
    meta = meta[meta["stress_flag"].astype(bool)].reset_index(drop=True)
    if len(meta) > n:
        meta = meta.iloc[np.argsort(_u01(meta["record_id"], seed), kind="stable")[:n]]
    keep, parts = set(meta["record_id"]), []
    for f in _restricted_files(cfg, corpus):
        t = pq.read_table(f, columns=["record_id", "text_clean"]).to_pandas()
        parts.append(t[t["record_id"].isin(keep)])
    txt = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["record_id", "text_clean"])
    return meta.merge(txt, on="record_id", how="inner").drop(columns="stress_flag")


_EMBEDDERS: dict[str, Any] = {}


def get_embedder(name: str, device: str | None = None) -> Any:
    """Per-process cached SentenceTransformer (lazy import; never loaded at import time)."""
    if name not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer
        _EMBEDDERS[name] = SentenceTransformer(name, device=device)
    return _EMBEDDERS[name]


def embed(texts: list[str], name: str, batch_size: int = 64, device: str | None = None, show_progress: bool = False) -> np.ndarray:
    """L2-normalised float32 embeddings (cosine = dot product)."""
    return get_embedder(name, device).encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show_progress).astype(np.float32)


def topic_model_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_root"]) / topic_settings(cfg)["model_dir"]


def latest_topic_model(cfg: dict[str, Any]) -> Path | None:
    root = topic_model_root(cfg)
    vs = sorted(p for p in root.glob("v????????") if (p / "centroids.npz").is_file()) if root.exists() else []
    return vs[-1] if vs else None


# >>> USER RUNS THIS FUNCTION. DO NOT EXECUTE. <<<   (fits an unsupervised topic model: BERTopic = embeddings + UMAP + HDBSCAN + c-TF-IDF)
def fit_topic_model(cfg: dict[str, Any], sample: pd.DataFrame, version: str | None = None) -> Path:
    """Fit BERTopic on `sample` (stress-flagged records only), reduce outliers, and save what is needed to label and to assign topics at scale:
      v<date>/centroids.npz        topic ids + L2-normalised mean embedding per topic (embedding-space centroids; numbers only)
      v<date>/topics_keywords.parquet   topic_id, rank, term, ctfidf_score, size, label  (KEYWORDS ONLY)
      v<date>/topic_model.json     settings used, sample size, versions
    The BERTopic object itself is NOT saved and no representative documents are exported: both can contain verbatim text.
    API NOTE (UNVERIFIED against your installed version; check `pip show bertopic umap-learn hdbscan`): `reduce_outliers(..., strategy="embeddings",
    embeddings=..., threshold=...)` and `update_topics(docs, topics=..., top_n_words=..., vectorizer_model=..., ctfidf_model=...)`."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from .utils import library_versions, select_device

    ts = topic_settings(cfg)
    docs = sample["text_clean"].tolist()
    LOG.info("embedding %d documents with %s", len(docs), ts["embedding_model"])
    emb = embed(docs, ts["embedding_model"], int(ts["embed_batch_size"]), select_device(cfg), show_progress=True)
    vec = CountVectorizer(stop_words=ts["vectorizer"]["stop_words"], ngram_range=tuple(ts["vectorizer"]["ngram_range"]), min_df=int(ts["vectorizer"]["min_df"]))
    ctf = ClassTfidfTransformer(reduce_frequent_words=bool(ts["reduce_frequent_words"]))
    tm = BERTopic(umap_model=UMAP(**ts["umap"]), hdbscan_model=HDBSCAN(prediction_data=True, **ts["hdbscan"]), vectorizer_model=vec, ctfidf_model=ctf,
                  top_n_words=int(ts["top_n_words"]), nr_topics=ts["nr_topics"], calculate_probabilities=False, verbose=True)
    topics, _ = tm.fit_transform(docs, emb)
    o = ts["outliers"]
    topics = tm.reduce_outliers(docs, topics, strategy=o["strategy"], embeddings=emb, threshold=float(o["threshold"]))
    tm.update_topics(docs, topics=topics, top_n_words=int(ts["top_n_words"]), vectorizer_model=vec, ctfidf_model=ctf)

    topics = np.asarray(topics)
    ids = np.array(sorted(t for t in set(topics.tolist()) if t != -1))
    cent = np.stack([emb[topics == t].mean(axis=0) for t in ids]).astype(np.float32)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sizes = {int(t): int((topics == t).sum()) for t in ids}
    kw = []
    for t in ids:
        terms = tm.get_topic(int(t)) or []
        label = f"{int(t)}_" + "_".join(w for w, _ in terms[:3])
        kw += [{"topic_id": int(t), "rank": r + 1, "term": w, "ctfidf_score": float(s), "size": sizes[int(t)], "label": label} for r, (w, s) in enumerate(terms)]
    out = topic_model_root(cfg) / (version or f"v{time.strftime('%Y%m%d')}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "centroids.npz", topic_ids=ids, centroids=cent, embedding_model=np.array(ts["embedding_model"]))
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(kw), preserve_index=False), out / "topics_keywords.parquet")
    write_json(out / "topic_model.json", {"settings": ts, "n_sample": len(docs), "n_topics": len(ids), "outliers_after_reduction": int((topics == -1).sum()),
                                          "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "versions": library_versions(("bertopic", "umap-learn", "hdbscan", "sentence-transformers", "scikit-learn", "numpy"))})
    LOG.info("topic model saved to %s: %d topics, sizes %s", out, len(ids), dict(list(sizes.items())[:10]))
    return out


def load_centroids(model_dir: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    z = np.load(Path(model_dir) / "centroids.npz", allow_pickle=False)
    return z["topic_ids"], z["centroids"], str(z["embedding_model"])


def assign_texts(texts: list[str], ids: np.ndarray, cent: np.ndarray, embedding_model: str, *, batch_size: int = 2048, embed_batch: int = 64,
                 min_similarity: float = 0.0, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid topic per text (cosine). Returns (topic_id, similarity); similarity < min_similarity -> topic -1.
    TRADE-OFF vs BERTopic.transform: no UMAP / HDBSCAN approximate_predict, so it is fast and memory-bounded, but boundaries are flat (centroid) rather than
    density-based. `centroid_agreement` (below) measures the difference on the fitting sample; check it before trusting the assignment."""
    topic, sim = np.empty(len(texts), dtype=np.int64), np.empty(len(texts), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        e = embed(texts[i:i + batch_size], embedding_model, embed_batch, device)
        s = e @ cent.T
        j = s.argmax(axis=1)
        topic[i:i + batch_size], sim[i:i + batch_size] = ids[j], s[np.arange(len(j)), j]
    topic[sim < min_similarity] = -1
    return topic, sim


def assign_partition(payload: dict[str, Any], source: str, date: str) -> dict[str, Any]:
    """Assign topics to the stress-flagged records of one source/date and write (record_id, topic_id, topic_sim). Idempotent: an existing output is kept."""
    out = Path(payload["assignments_root"]) / f"source={source}" / f"date={date}" / "part-00000.parquet"
    if out.is_file():
        return {"source": source, "date": date, "status": "skipped", "rows": 0}
    rf = Path(payload["restricted_root"]) / f"source={source}" / f"date={date}" / "part-00000.parquet"
    ef = Path(payload["enriched_root"]) / f"source={source}" / f"date={date}" / "part-00000.parquet"
    if not (rf.is_file() and ef.is_file()):
        return {"source": source, "date": date, "status": "no_input", "rows": 0}
    flag = pq.read_table(ef, columns=["record_id", "stress_flag", "input_status"]).to_pandas()
    flag = flag[flag["stress_flag"].fillna(False).astype(bool) & flag["input_status"].isin(SCORED)][["record_id"]]
    txt = pq.read_table(rf, columns=["record_id", "text_clean"]).to_pandas().merge(flag, on="record_id")
    ids, cent, emb_name = load_centroids(payload["model_dir"])
    if len(txt):
        topic, sim = assign_texts(txt["text_clean"].tolist(), ids, cent, emb_name, batch_size=payload["batch_size"], min_similarity=payload["min_similarity"], device=payload.get("device"))
    else:
        topic, sim = np.empty(0, np.int64), np.empty(0, np.float32)
    res = pd.DataFrame({"record_id": txt["record_id"].to_numpy(), "topic_id": topic, "topic_sim": sim})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(res, preserve_index=False), tmp, compression="zstd")
    tmp.replace(out)
    return {"source": source, "date": date, "status": "done", "rows": int(len(res))}


def topic_payload(cfg: dict[str, Any], corpus: str, model_dir: str | Path | None = None) -> dict[str, Any]:
    from . import spark_jobs as SJ
    from .utils import select_device
    md = Path(model_dir) if model_dir else latest_topic_model(cfg)
    if md is None:
        raise FileNotFoundError("no fitted topic model: run fit_topic_model first")
    ts = topic_settings(cfg)
    return {"model_dir": str(md.resolve()), "model_version": md.name, "restricted_root": str(SJ.restricted_root(cfg, corpus).resolve()),
            "enriched_root": str(SJ.enriched_root(cfg, corpus).resolve()), "assignments_root": str((Path(cfg["_root"]) / ts["assignments_dir"] / corpus / md.name).resolve()),
            "batch_size": int(ts["assign"]["batch_size"]), "min_similarity": float(ts["assign"]["min_similarity"]), "device": select_device(cfg)}


def assign_topics(cfg: dict[str, Any], corpus: str, model_dir: str | Path | None = None, max_partitions: int | None = None) -> list[dict[str, Any]]:
    """Assign topics at scale in batches, one source/date file at a time (restartable: finished partitions are skipped). Plain Python, one process."""
    p = topic_payload(cfg, corpus, model_dir)
    parts = [_partition_key(f) for f in _restricted_files(cfg, corpus)]
    res = []
    for s, d in parts[:max_partitions]:
        res.append(assign_partition(p, s, d))
    return res


# one task = one (source, date); the same code as above, scheduled by Spark (pandas UDF via mapInPandas). Needs torch + sentence-transformers in the
# Spark Python worker: on Windows Step 9 found a JVM-started worker crashed on `import torch`; if it does here, use `assign_topics` (same result).
def assign_topics_spark(cfg: dict[str, Any], corpus: str, model_dir: str | Path | None = None, workers: int = 1) -> pd.DataFrame:
    from . import spark_jobs as SJ
    p = topic_payload(cfg, corpus, model_dir)
    pf = Path(cfg["_paths"]["interim"]) / f"topic_payload_{corpus}.json"
    write_json(pf, p)
    pairs = pd.DataFrame([_partition_key(f) for f in _restricted_files(cfg, corpus)], columns=["source", "date"])
    path = str(pf.resolve())

    def work(batches):
        for pdf in batches:
            pl = json.loads(Path(path).read_text(encoding="utf-8"))
            yield pd.DataFrame([assign_partition(pl, s, d) for s, d in zip(pdf["source"], pdf["date"])], columns=["source", "date", "status", "rows"])

    spark = SJ.build_spark(cfg, workers=workers)
    try:
        sdf = spark.createDataFrame(pairs).repartition(max(1, workers))
        return sdf.mapInPandas(work, schema="source string, date string, status string, rows long").toPandas()
    finally:
        spark.stop()


def centroid_agreement(cfg: dict[str, Any], sample: pd.DataFrame, model_dir: str | Path, fitted_topics: np.ndarray) -> float:
    """Share of the fitting sample on which the nearest-centroid assignment equals the BERTopic topic (after outlier reduction). Run it after fitting."""
    ids, cent, emb_name = load_centroids(model_dir)
    topic, _ = assign_texts(sample["text_clean"].tolist(), ids, cent, emb_name, batch_size=topic_settings(cfg)["assign"]["batch_size"])
    return float((topic == np.asarray(fitted_topics)).mean())


# ---------------------------------------------------------------- MLlib LDA alternative (distributed, no embeddings)
def fit_lda_mllib(spark: Any, texts: pd.DataFrame, cfg: dict[str, Any]) -> tuple[Any, pd.DataFrame]:
    """Spark MLlib LDA alternative. Returns (pipeline model, keywords frame). UNVERIFIED API details: check against your pyspark 3.5 (`pyspark.ml.clustering.LDA`).
    # >>> USER RUNS THIS FUNCTION. DO NOT EXECUTE. <<<   (fits an unsupervised topic model)
    Trade-offs vs BERTopic:  + fully distributed (fit AND assign scale with Spark), no GPU, no embedding model, no UMAP/HDBSCAN memory limit, `transform` assigns
    every document natively;  - bag-of-words: no semantics, synonyms split topics; - k must be chosen (no density-based discovery); - every document is forced into a
    topic (no outlier class), and short or generic texts give diffuse topics; - keywords are the highest topic-term WEIGHTS, not c-TF-IDF (generic words rank high
    unless you extend the stop-word list); - results vary with seed and optimizer (online vs em)."""
    from pyspark.ml import Pipeline
    from pyspark.ml.clustering import LDA
    from pyspark.ml.feature import CountVectorizer, RegexTokenizer, StopWordsRemover

    ld = topic_settings(cfg)["lda"]
    sdf = spark.createDataFrame(texts[["record_id", "text_clean"]])
    pipe = Pipeline(stages=[RegexTokenizer(inputCol="text_clean", outputCol="tok", pattern="[^a-zA-Z']+", minTokenLength=3),
                            StopWordsRemover(inputCol="tok", outputCol="words"),
                            CountVectorizer(inputCol="words", outputCol="features", vocabSize=int(ld["vocab_size"]), minDF=float(ld["min_df"])),
                            LDA(k=int(ld["k"]), maxIter=int(ld["max_iter"]), optimizer="online", seed=int(ld["seed"]), featuresCol="features")])
    model = pipe.fit(sdf)
    vocab = model.stages[2].vocabulary
    top = model.stages[3].describeTopics(int(topic_settings(cfg)["top_n_words"])).toPandas()
    kw = [{"topic_id": int(r.topic), "rank": j + 1, "term": vocab[ti], "ctfidf_score": float(w), "size": None, "label": f"{int(r.topic)}_" + "_".join(vocab[x] for x in r.termIndices[:3])}
          for r in top.itertuples() for j, (ti, w) in enumerate(zip(r.termIndices, r.termWeights))]
    return model, pd.DataFrame(kw)


def assign_lda(model: Any, spark: Any, texts: pd.DataFrame) -> pd.DataFrame:
    """(record_id, topic_id) = argmax topicDistribution for each text, computed in Spark."""
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    out = model.transform(spark.createDataFrame(texts[["record_id", "text_clean"]])).select("record_id", vector_to_array("topicDistribution").alias("p"))
    return out.select("record_id", F.expr("array_position(p, array_max(p)) - 1").alias("topic_id")).toPandas()


# ================================================================ 6. topic series, emerging vs persistent
def topic_counts(cfg: dict[str, Any], corpus: str, model_version: str | None = None) -> pd.DataFrame:
    """Record-level assignments joined to community/date -> INTERNAL counts (date, community, topic_id, n). Not publishable as is (not suppressed)."""
    root = Path(cfg["_root"]) / topic_settings(cfg)["assignments_dir"] / corpus
    md = model_version or sorted(p.name for p in root.glob("v????????"))[-1]
    a = pds.dataset(str(root / md), format="parquet", partitioning=_partitioning(), ignore_prefixes=["_", "."]).to_table(columns=["record_id", "topic_id"]).to_pandas()
    m = load_enriched(enriched_dir(cfg, corpus), ["record_id", "community", "date"])
    j = a.merge(m, on="record_id", how="inner")
    return j.groupby(["date", "community", "topic_id"]).size().rename("n").reset_index()


def _scope_counts(tc: pd.DataFrame, gran: str) -> pd.DataFrame:
    t = tc.assign(period=period_start(tc["date"], gran) if "date" in tc.columns else tc["period"])
    t = t.groupby(["period", "community", "topic_id"], as_index=False)["n"].sum()
    return pd.concat([t, t.groupby(["period", "topic_id"], as_index=False)["n"].sum().assign(community="ALL")[t.columns]], ignore_index=True)


def topic_share_series(tc: pd.DataFrame, gran: str, k_min: int) -> pd.DataFrame:
    """Topic share per period x scope among stress-flagged assigned records (denominator n_total includes topic -1). A row is blanked if either the
    scope-period denominator or the topic count is < k_min. Topic -1 (unassigned) is not listed."""
    t = _scope_counts(tc, gran)
    t["n_total"] = t.groupby(["period", "community"])["n"].transform("sum")
    t = t[t["topic_id"] != -1].rename(columns={"n": "n_topic"})
    t["share"] = t["n_topic"] / t["n_total"]
    t["ci_low"], t["ci_high"] = wilson(t["n_topic"], t["n_total"])
    t = suppress(t, "n_topic", k_min, ["share", "ci_low", "ci_high"], extra_mask=t["n_total"] < k_min)
    t["n_total"] = t["n_total"].astype("Int64").mask(t["suppressed"])
    return t.sort_values(["community", "topic_id", "period"]).reset_index(drop=True)[["period", "community", "topic_id", "n_topic", "n_total", "share", "ci_low", "ci_high", "suppressed"]]


def _matrices(g: pd.DataFrame, gran: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """topics x periods count matrix (zeros for missing), the per-period total (all topics incl. -1), topic ids; periods form a gap-free grid."""
    grid = period_grid(g["period"].min(), g["period"].max(), gran)
    piv = g.pivot_table(index="topic_id", columns="period", values="n", aggfunc="sum", fill_value=0).reindex(columns=grid, fill_value=0)
    tot = piv.sum(axis=0).to_numpy(float)
    keep = piv.index != -1
    return grid, piv[keep].to_numpy(float), tot, piv.index[keep].to_numpy()


def emerging_topics(tc: pd.DataFrame, gran: str, ecfg: dict[str, Any], k_min: int, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """EMERGING topic (evaluated per scope = community or ALL, as of the last period <= `as_of`):
      recent window W = the last R periods; baseline B = the B periods immediately before W (R = recent_periods, B = baseline_periods; no overlap, no future data).
      c_W, N_W = topic count and all-topic total in W; c_B, N_B likewise in B (pooled counts, not averages of period shares).
      share_W = c_W / N_W;  share_B = max(c_B, k_min) / N_B   (the baseline count is FLOORED at k_min: it avoids dividing by ~0 for brand-new topics, makes the
      ratio conservative, and means no ratio can reveal a baseline count below k_min).
      growth_ratio = share_W / share_B.   A period p of W is `hot` when its own share_p / share_B >= growth_ratio_threshold.
      EMERGING iff  c_W >= max(min_count, k_min)  AND  growth_ratio >= threshold  AND  the longest run of consecutive hot periods in W >= persistence_k
                    AND (if require_ci_above_baseline) the Wilson 95% lower bound of share_W > share_B.
    Not evaluable (row omitted): N_W < k_min or N_B < k_min or fewer than R + B periods of data."""
    R, Bn, g, pk, mc = int(ecfg["recent_periods"]), int(ecfg["baseline_periods"]), float(ecfg["growth_ratio"]), int(ecfg["persistence_k"]), max(int(ecfg["min_count"]), k_min)
    out = []
    for scope, s in _scope_counts(tc, gran).groupby("community"):
        grid, M, tot, ids = _matrices(s, gran)
        end = len(grid) if as_of is None else int(grid.searchsorted(pd.Timestamp(as_of), side="right"))
        if end < R + Bn:
            continue
        w, b = slice(end - R, end), slice(end - R - Bn, end - R)
        N_W, N_B = tot[w].sum(), tot[b].sum()
        if N_W < k_min or N_B < k_min:
            continue
        cW, cB = M[:, w].sum(axis=1), M[:, b].sum(axis=1)
        sW, sB = cW / N_W, np.maximum(cB, k_min) / N_B
        with np.errstate(divide="ignore", invalid="ignore"):
            hot = (M[:, w] / np.where(tot[w] > 0, tot[w], np.nan)) / sB[:, None] >= g
        hot = np.nan_to_num(hot, nan=0).astype(bool)
        run = np.zeros(len(ids), int)
        cur = np.zeros(len(ids), int)
        for j in range(R):
            cur = np.where(hot[:, j], cur + 1, 0)
            run = np.maximum(run, cur)
        lo, _ = wilson(cW, N_W)
        ratio = sW / sB
        ok = (cW >= mc) & (ratio >= g) & (run >= pk) & ((lo > sB) if ecfg.get("require_ci_above_baseline", True) else True)
        out.append(pd.DataFrame({"community": scope, "topic_id": ids, "as_of_period": grid[end - 1], "n_recent": cW.astype(int), "n_recent_total": int(N_W),
                                 "share_recent": sW, "n_baseline": cB.astype(int), "n_baseline_total": int(N_B), "share_baseline_floored": sB,
                                 "growth_ratio": ratio, "max_consecutive_hot": run, "ci_low_recent": lo, "is_emerging": ok}))
    if not out:
        return pd.DataFrame(columns=["community", "topic_id", "as_of_period", "n_recent", "n_recent_total", "share_recent", "n_baseline", "n_baseline_total",
                                     "share_baseline_floored", "growth_ratio", "max_consecutive_hot", "ci_low_recent", "is_emerging", "suppressed"])
    t = pd.concat(out, ignore_index=True)
    t["baseline_below_k"] = t["n_baseline"] < k_min
    t["n_baseline"] = t["n_baseline"].astype("Int64").mask(t["baseline_below_k"])
    return suppress(t, "n_recent", k_min, ["share_recent", "share_baseline_floored", "growth_ratio", "max_consecutive_hot", "ci_low_recent", "n_baseline", "is_emerging"],
                    ).assign(is_emerging=lambda x: x["is_emerging"].fillna(False).astype(bool))


def persistent_topics(tc: pd.DataFrame, gran: str, pcfg: dict[str, Any], k_min: int) -> pd.DataFrame:
    """PERSISTENT topic (per scope): over the VALID periods (scope total N_p >= k_min), the topic's share_p = c_p / N_p is >= share_threshold in at least
    min_fraction of them, and there are at least min_periods valid periods. fraction_above = periods above / valid periods. Uses the whole span."""
    thr, frac, minp = float(pcfg["share_threshold"]), float(pcfg["min_fraction"]), int(pcfg["min_periods"])
    out = []
    for scope, s in _scope_counts(tc, gran).groupby("community"):
        grid, M, tot, ids = _matrices(s, gran)
        valid = tot >= k_min
        nv = int(valid.sum())
        if nv == 0:
            continue
        share = M[:, valid] / tot[valid]
        above = (share >= thr).sum(axis=1)
        out.append(pd.DataFrame({"community": scope, "topic_id": ids, "n_valid_periods": nv, "n_periods_above": above, "fraction_above": above / nv,
                                 "mean_share": share.mean(axis=1), "n_topic_valid_periods": M[:, valid].sum(axis=1).astype(int), "is_persistent": (nv >= minp) & (above / nv >= frac)}))
    if not out:
        return pd.DataFrame()
    t = pd.concat(out, ignore_index=True)
    return suppress(t, "n_topic_valid_periods", k_min, ["n_periods_above", "fraction_above", "mean_share", "is_persistent"]).assign(
        is_persistent=lambda x: x["is_persistent"].fillna(False).astype(bool))


def topic_status(emerging: pd.DataFrame, persistent: pd.DataFrame) -> pd.DataFrame:
    """One row per scope x topic with status: both | emerging | persistent | neither (suppressed rows stay `neither`)."""
    m = persistent[["community", "topic_id", "is_persistent"]].merge(emerging[["community", "topic_id", "is_emerging"]], on=["community", "topic_id"], how="outer")
    m = m.fillna({"is_persistent": False, "is_emerging": False}).astype({"is_persistent": bool, "is_emerging": bool})
    m["status"] = np.select([m["is_persistent"] & m["is_emerging"], m["is_emerging"], m["is_persistent"]], ["both", "emerging", "persistent"], "neither")
    return m


# ================================================================ 7. orchestration: tables -> data/gold
def _file_record(p: Path) -> dict[str, Any]:
    return {"file": p.name, "sha256": sha256_file(p), "rows": pq.ParquetFile(p).metadata.num_rows}


def run_core_analytics(cfg: dict[str, Any], corpus: str, per_community_emotions: bool = True) -> dict[str, Any]:
    """Items 1-4: temporal aggregates, composition-adjusted series, emotions, stressor x emotion, stressor shares -> data/gold/*.parquet + manifest."""
    a, k = cfg["analytics"], int(cfg["privacy"]["k_min"])
    root = enriched_dir(cfg, corpus)
    df = load_enriched(root, ["date", "source", "community", "stress_prob", "stress_flag", "sample_weight", "second_stage_scored", "stressor_top"],
                       prefixes=("emotion_p__", "emotion_group__", "stressor_flag__"))
    written: dict[str, Path] = {}
    n_supp: dict[str, int] = {}

    def put(name: str, t: pd.DataFrame, counts: Iterable[str]) -> None:
        written[name] = write_gold(cfg, name, t, counts)
        n_supp[name] = int(t["suppressed"].sum()) if "suppressed" in t else 0

    w, ref = reference_weights(df, a["composition"]["reference"], k)
    for gran in a["granularities"]:
        cells = stress_cells(df, gran)
        put(f"stress_rate_{gran}", stress_rate_table(cells, k), ["n_total"])
        put(f"stress_rate_adjusted_{gran}", adjusted_series(cells, w, k, float(a["composition"]["min_coverage"])), ["n_total"])
    wt = suppress(w.rename(columns={"n_ref": "n_total"}), "n_total", k, ["weight_all", "weight_in_source"]).drop(columns="usable")
    put("composition_weights", wt, ["n_total"])
    put("emotion_prevalence_lift", emotion_lift(df, cfg, per_community_emotions), ["n_flagged", "n_unflagged_sampled"])
    put("stressor_emotion_cooccurrence", stressor_emotion_cooccurrence(df, cfg), ["n_stressor"])
    put("stressor_share", stressor_share(df, a["stressors"]["granularity"], k), ["n_stress_scored"])
    man = {"corpus": corpus, "step": 10, "k_min": k, "rows_scored": int(len(df)), "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
           "reference_window": ref, "analytics_settings": a, "suppressed_rows": n_supp, "domain_shift_note": DOMAIN_SHIFT_NOTE,
           "files": {n: _file_record(p) for n, p in written.items()}, "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    write_json(gold_dir(cfg) / "analytics_core_manifest.json", man)
    return man


def run_topic_analytics(cfg: dict[str, Any], corpus: str, model_dir: str | Path | None = None) -> dict[str, Any]:
    """Items 5-6 after `assign_topics`: keywords (no text), topic share series, emerging and persistent topics, status -> data/gold/*.parquet."""
    ts, k = topic_settings(cfg), int(cfg["privacy"]["k_min"])
    md = Path(model_dir) if model_dir else latest_topic_model(cfg)
    gran = ts["granularity"]
    tc = topic_counts(cfg, corpus, md.name)
    kw = pd.read_parquet(md / "topics_keywords.parquet")
    em, pe = emerging_topics(tc, gran, ts["emerging"], k), persistent_topics(tc, gran, ts["persistent"], k)
    tables = {"topic_keywords": (kw, ["size"]), f"topic_share_{gran}": (topic_share_series(tc, gran, k), ["n_topic", "n_total"]),
              "topics_emerging": (em, ["n_recent"]), "topics_persistent": (pe, ["n_topic_valid_periods"]), "topics_status": (topic_status(em, pe), [])}
    written = {n: write_gold(cfg, n, t, c) for n, (t, c) in tables.items()}
    man = {"corpus": corpus, "step": 10, "k_min": k, "topic_model": md.name, "topic_settings": ts, "files": {n: _file_record(p) for n, p in written.items()},
           "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    write_json(gold_dir(cfg) / "analytics_topics_manifest.json", man)
    return man
