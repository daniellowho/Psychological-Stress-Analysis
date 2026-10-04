"""Step 7: unified INFERENCE over the saved stress, emotion and stressor bundles. Nothing here trains or updates weights.

    pipe = StressSignalPipeline.from_artifacts("trained_models")        # latest version of each component, hashes verified
    df = pipe.predict(["some text", ...], batch_size=8)                  # one row per input text, same order

Output (one row per input; see StressSignalPipeline.output_columns for the exact list and order):
    input_status                      ok | ok_truncated | none | non_string | empty | empty_after_cleaning | no_text | non_english
                                      (anything other than ok/ok_truncated: every score is NaN, never a crash)
    stress_prob_calibrated, stress_flag
    emotion_p__<28 labels>, emotion_group__<7 Ekman groups>
    stressor_{p|cos|count}__<category>, stressor_flag__<category>, stressor_top, stressor_none_unclear   (if a stressor bundle exists)
    tension_stress_strength, tension_relaxation                                                          (only if inference.tension.enabled)
    stress_version, emotion_version, stressor_version, n_chunks_stress, n_chunks_emotion

The columns are SCORES OF TEXTS. They are not statements about the people who wrote them; the system feeds aggregate,
population-level signals only (no diagnosis, no individual-level use).

Design notes
    * Same cleaning and chunking as training: preprocess.clean_text, then preprocess.chunk_text with the window, stride and
      max-chunk values stored in each bundle's config JSON; chunk probabilities are calibrated (temperature for stress,
      per-label Platt for emotion) and combined with preprocess.aggregate_chunk_scores.
    * Spark-safe: `get_pipeline` keeps one pipeline per Python process in a module-level cache, so every worker loads the
      models once. A pipeline holds only frozen settings (no global mutable config, no notebook state).
    * Efficiency: texts are length-sorted for batching, torch.inference_mode, fp32 unless inference.use_fp16 (CUDA only; off by
      default because Pascal GPUs gain little), CUDA out-of-memory halves the batch and finally falls back to the CPU.
    * Determinism: eval mode, no sampling, stable sorting. Floating-point results can differ by ~1e-6 if the SAME text lands in
      a batch with different padding (different batch size or neighbours); the tests allow 1e-4 for that case and require exact
      equality for an identical call.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import __version__ as PACKAGE_VERSION
from .metrics import apply_label_calibration, positive_proba, softmax
from .preprocess import PLACEHOLDER_RE, aggregate_chunk_scores, chunk_text, clean_text
from .utils import get_logger, library_versions, select_device, sha256_file, write_json

LOG = get_logger("stress_signals.inference")

COMPONENTS = ("stress", "emotion", "stressor")
REQUIRED_COMPONENTS = ("stress", "emotion")          # the stressor bundle is optional until Step 5 has produced one
VERSION_RE = re.compile(r"^v\d{8}$")                 # .bak-* directories (moved-aside bundles) never match
OK_STATUSES = ("ok", "ok_truncated")
SCOPE_NOTE = ("Scores describe stress-related LANGUAGE in a text. They are not a diagnosis or an assessment of any person and "
              "are meant for aggregate, population-level analysis only.")


class BundleIntegrityError(RuntimeError):
    """A bundle file is missing or its SHA-256 differs from the manifest written when the bundle was saved."""


# ================================================================ settings
@dataclass(frozen=True)
class InferenceSettings:
    """Frozen copy of the `inference` config section (+ defaults), read once when a pipeline is built."""
    device: str = "auto"
    batch_size: int = 8
    use_fp16: bool = False
    max_chars: int = 20000
    min_alpha_chars: int = 3
    english_only: bool = True
    adaptive_batch: bool = True
    tension_enabled: bool = False
    gate_stressors: bool = True
    cpu_int8: bool = False

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None, **overrides: Any) -> "InferenceSettings":
        inf = dict((cfg or {}).get("inference", {}) or {})
        base = cls(
            device=inf.get("device", "auto"), batch_size=int(inf.get("batch_size", 8)),
            use_fp16=bool(inf.get("use_fp16", inf.get("half_precision", False))),
            max_chars=int(inf.get("max_chars", 20000)), min_alpha_chars=int(inf.get("min_alpha_chars", 3)),
            english_only=bool(inf.get("english_only", True)), adaptive_batch=bool(inf.get("adaptive_batch", True)),
            tension_enabled=bool((inf.get("tension") or {}).get("enabled", False)),
            gate_stressors=bool(inf.get("gate_stressors_on_stress_flag", True)), cpu_int8=bool(inf.get("cpu_int8", False)))
        return cls(**{**base.__dict__, **overrides})


# ================================================================ bundle discovery and integrity
def list_versions(artifacts_dir: str | Path, component: str) -> list[str]:
    """Version directory names (v<YYYYMMDD>) of a component that hold a manifest, oldest first."""
    root = Path(artifacts_dir) / component
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and VERSION_RE.match(p.name) and (p / "manifest.json").is_file())


def resolve_version(artifacts_dir: str | Path, component: str, pin: str | None = "latest") -> str | None:
    """`pin` = "latest" (default), None (same), or an exact name like "v20261004". Returns None if the component has no bundle."""
    found = list_versions(artifacts_dir, component)
    if pin in (None, "latest"):
        return found[-1] if found else None
    if pin not in found:
        raise FileNotFoundError(f"{component}: pinned version {pin!r} not found under {Path(artifacts_dir) / component} "
                                f"(available: {found})")
    return pin


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Re-hash every file listed in the bundle's manifest.json and compare. Raises BundleIntegrityError on any difference.
    Returns {manifest_sha256, n_files}. Files not listed in the manifest are ignored (manifest.json itself is hashed by the caller)."""
    d = Path(bundle_dir)
    mpath = d / "manifest.json"
    if not mpath.is_file():
        raise BundleIntegrityError(f"{d}: manifest.json missing")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    entries: dict[str, str] = {}
    for rel, meta in (manifest.get("artifacts") or {}).items():
        if meta.get("type") == "file":
            entries[rel] = meta["sha256"]
        elif meta.get("type") == "dir":
            entries.update({f"{rel}/{r}": h for r, h in meta["files"].items()})
        else:
            raise BundleIntegrityError(f"{d}: manifest lists {rel!r} with type {meta.get('type')!r}")
    if not entries:
        raise BundleIntegrityError(f"{d}: manifest lists no files")
    bad = []
    for rel, want in entries.items():
        p = d / rel
        if not p.is_file():
            bad.append(f"{rel}: missing")
        elif sha256_file(p) != want:
            bad.append(f"{rel}: sha256 differs from the manifest")
    if bad:
        raise BundleIntegrityError(f"{d}: integrity check failed: " + "; ".join(bad))
    return {"manifest_sha256": sha256_file(mpath), "n_files": len(entries)}


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


# ================================================================ model runner
class _Runner:
    """One loaded transformer + tokenizer on one device. Length-sorted batches, inference_mode, OOM fallback."""

    def __init__(self, model: Any, tokenizer: Any, device: str, half: bool, adaptive_batch: bool):
        import torch
        self.tok = tokenizer
        self.device = device
        self.half = bool(half and device == "cuda")
        self.adaptive = adaptive_batch
        self.model = model.to(device).eval()
        if self.half:
            self.model = self.model.half()
        self._torch = torch

    @classmethod
    def load(cls, bundle: Path, device: str, half: bool, adaptive_batch: bool) -> "_Runner":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(bundle), use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(bundle))
        return cls(model, tok, device, half, adaptive_batch)

    def _to_cpu(self) -> None:
        self.model = self.model.float().to("cpu")
        self.device, self.half = "cpu", False
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def logits(self, texts: Sequence[str], max_length: int, batch_size: int) -> np.ndarray:
        """Logits (n, num_labels) in input order (float32)."""
        torch = self._torch
        n = len(texts)
        out = np.zeros((n, self.model.config.num_labels), dtype=np.float32)
        if n == 0:
            return out
        order = np.argsort([len(t) for t in texts], kind="stable")
        bs, i = max(1, int(batch_size)), 0
        with torch.inference_mode():
            while i < n:
                idx = order[i:i + bs]
                try:
                    enc = self.tok([texts[j] for j in idx], truncation=True, max_length=max_length, padding=True,
                                   return_tensors="pt").to(self.device)
                    out[idx] = self.model(**enc).logits.float().cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    if not self.adaptive or self.device != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    if bs > 1:
                        bs = max(1, bs // 2)
                        LOG.warning("CUDA out of memory; batch size reduced to %d", bs)
                    else:
                        LOG.warning("CUDA out of memory at batch size 1; continuing on the CPU")
                        self._to_cpu()
                    continue
                i += len(idx)
        return out


def _chunk_all(tok: Any, texts: Sequence[str], max_length: int, stride: int, max_chunks: int | None
               ) -> tuple[list[str], np.ndarray, list[list[Any]], list[int]]:
    """Chunk every text (preprocess.chunk_text, same rule as training). Returns (flat chunk texts, chunks per text, chunk lists,
    positions of texts whose chunking failed or produced nothing; they get NaN, never an exception)."""
    budget = max_length - tok.num_special_tokens_to_add(pair=False)
    stride = min(int(stride), budget // 2)                      # identical to stress_model/emotion_model inference
    flat: list[str] = []
    counts: list[int] = []
    lists: list[list[Any]] = []
    failed: list[int] = []
    for i, t in enumerate(texts):
        try:
            cs = chunk_text(t, tok, max_length, stride, max_chunks)
        except Exception as exc:  # noqa: BLE001 - a pathological text must not kill the batch
            LOG.warning("chunking failed for one text (%s); its scores are NaN", type(exc).__name__)
            cs = []
        if not cs:
            failed.append(i)
        flat += [c.text for c in cs]
        counts.append(len(cs))
        lists.append(cs)
    return flat, np.asarray(counts, dtype=int), lists, failed


def _aggregate(P: np.ndarray, counts: np.ndarray, lists: Sequence[Sequence[Any]], method: str) -> np.ndarray:
    """Per-text aggregate of per-chunk (calibrated) probabilities P (n_chunks, k). Texts with 0 chunks -> NaN rows."""
    out = np.full((len(counts), P.shape[1]), np.nan)
    pos = 0
    for i, c in enumerate(counts):
        if c:
            w = [ch.n_tokens for ch in lists[i]] if method == "weighted_mean" else None
            out[i] = aggregate_chunk_scores(P[pos:pos + c], method, w)
        pos += int(c)
    return out


# ================================================================ components
class _StressComponent:
    """Binary stress classifier: calibrated P(stress) per chunk (temperature scaling), mean over chunks, flag at the bundle's threshold."""

    def __init__(self, bundle: Path, st: InferenceSettings, device: str):
        self.cfg = _read_json(bundle / "stress_config.json")
        self.runner = _Runner.load(bundle, device, st.use_fp16, st.adaptive_batch)
        cal = self.cfg["calibration"]
        if cal.get("method") != "temperature_scaling":
            raise ValueError(f"stress calibration {cal.get('method')!r} not supported by this loader")
        self.T = float(cal["temperature"])
        self.threshold = float(self.cfg["threshold"]["value"])
        self.positive = int(self.cfg.get("positive_label", 1))
        self.max_length = int(self.cfg["max_length"])
        self.ch = self.cfg["chunking"]

    def score(self, texts: Sequence[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        flat, counts, lists, _ = _chunk_all(self.runner.tok, texts, self.max_length, self.ch["stride"], self.ch["max_chunks_per_doc"])
        pc = positive_proba(self.runner.logits(flat, self.max_length, batch_size), self.T, self.positive)[:, None]
        return _aggregate(pc, counts, lists, self.ch["aggregation"])[:, 0], counts


class _EmotionComponent:
    """28-label multi-label emotion model: per-label Platt calibration per chunk, mean over chunks, Ekman group = max of members."""

    def __init__(self, bundle: Path, st: InferenceSettings, device: str):
        from .emotion_model import ekman_members, ekman_probs
        self.cfg = _read_json(bundle / "emotion_config.json")
        self.runner = _Runner.load(bundle, device, st.use_fp16, st.adaptive_batch)
        self.names = [self.cfg["label_map"][str(i)] for i in range(len(self.cfg["label_map"]))]
        cal = self.cfg["calibration"]
        if cal.get("method") != "platt":
            raise ValueError(f"emotion calibration {cal.get('method')!r} not supported by this loader")
        self.params = [cal["params"][n] for n in self.names]
        mapping = _read_json(bundle / self.cfg["ekman"]["mapping_file"])
        order = list(self.cfg["ekman"]["groups"])             # column order recorded at save time (the JSON file is key-sorted)
        if set(order) != set(mapping["groups"]):
            raise ValueError("ekman_mapping.json groups differ from emotion_config.json")
        groups = {g: mapping["groups"][g] for g in order}
        self.groups = order
        self.members = ekman_members(groups, self.names)
        self.combine = self.cfg["ekman"]["combine"]
        self._ekman_probs = ekman_probs
        self.max_length = int(self.cfg["max_length"])
        self.ch = self.cfg["chunking"]

    def score(self, texts: Sequence[str], batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat, counts, lists, _ = _chunk_all(self.runner.tok, texts, self.max_length, self.ch["stride"], self.ch["max_chunks_per_doc"])
        pc = apply_label_calibration(self.runner.logits(flat, self.max_length, batch_size), self.params)
        P = _aggregate(pc, counts, lists, self.ch["aggregation"])
        ok = ~np.isnan(P).any(1)
        G = np.full((len(texts), len(self.groups)), np.nan)
        if ok.any():
            G[ok] = self._ekman_probs(P[ok], self.members, self.combine)
        return P, G, counts


class _ProtoScorer:
    """Zero-shot cosine to the saved category prototypes (stands in for stressor_model.ZeroShotScorer, which would re-embed descriptions)."""

    def __init__(self, prototypes: np.ndarray):
        self.prototypes = np.asarray(prototypes, dtype=float)

    def window_scores(self, windows: Sequence[str], E: np.ndarray | None = None) -> np.ndarray:
        if E is None:
            raise ValueError("zero-shot scoring needs window embeddings")
        return np.asarray(E, dtype=float) @ self.prototypes.T


class _StressorComponent:
    """Stressor-category tagger rebuilt from a stressor bundle (stressor_config.json): sentence windows -> window scores ->
    post score (max/mean/...) -> per-category thresholds tuned on the gold-dev set. Methods: keyword | zero_shot | embed_lr | finetune."""

    PREFIX = {"probability": "p", "cosine": "cos", "count": "count"}

    def __init__(self, bundle: Path, st: InferenceSettings, device: str, embedder: Any | None = None):
        from . import stressor_model as SX
        self.SX = SX
        self.cfg = _read_json(bundle / "stressor_config.json")
        self.tax = SX.load_taxonomy(path=bundle / self.cfg["taxonomy_file"])
        self.cats = SX.category_ids(self.tax)
        self.kinds = {c["id"]: c["score_kind"] for c in self.cfg["categories"]}
        self.lower_conf = [c["id"] for c in self.cfg["categories"] if c.get("lower_confidence")]
        self.thr = np.array([self.cfg["thresholds"][c] for c in self.cats], dtype=float)
        self.agg = self.cfg["aggregation"]
        self.wcfg = self.cfg["inference_unit"]["windows"]
        self.method = self.cfg["method"]
        self.batch_size = st.batch_size
        protos = bundle / "prototypes.npy"
        proto = _ProtoScorer(np.load(protos)) if protos.is_file() else None
        self.embedder = embedder
        if self.method != "keyword" and (self.method in ("zero_shot", "embed_lr") or proto is not None) and embedder is None:
            spec = self.cfg.get("embedding_model")
            if not spec:
                raise ValueError("stressor bundle needs an embedding model but stressor_config.json records none")
            self.embedder = SX.Embedder(SX.EmbeddingSpec(**spec), device, st.batch_size)
        if self.method == "keyword":
            self.scorer = SX.KeywordScorer(self.tax)
        elif self.method == "zero_shot":
            self.scorer = proto
        elif self.method in ("embed_lr", "finetune"):
            T = float(self.cfg["calibration"]["temperature"])
            if self.method == "embed_lr":
                z = np.load(bundle / "lr.npz", allow_pickle=False)
                fn = SX.lr_logits_fn({"coef": z["coef"], "intercept": z["intercept"]})
            else:
                self.runner = _Runner.load(bundle, device, st.use_fp16, st.adaptive_batch)
                ml = int(self.cfg.get("finetune_max_length") or 128)    # bundles saved before Step 7 lack the key: UNVERIFIED default

                def fn(windows: Sequence[str], E: Any = None, _r=self.runner, _ml=ml) -> np.ndarray:
                    return _r.logits(list(windows), _ml, self.batch_size)
            self.scorer = SX.SupervisedScorer(self.tax, self.method, fn, T, proto, self.agg)
        else:
            raise ValueError(f"unknown stressor method {self.method!r}")

    def columns(self) -> list[str]:
        return ([f"stressor_{self.PREFIX[self.kinds[c]]}__{c}" for c in self.cats] + [f"stressor_flag__{c}" for c in self.cats]
                + ["stressor_top", "stressor_none_unclear"])

    def score(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        n, L = len(texts), len(self.cats)
        S = np.full((n, L), np.nan)
        windows: list[str] = []
        owner: list[int] = []
        for i, t in enumerate(texts):
            w = self.SX.sentence_windows(t, self.wcfg["sentences_per_window"], self.wcfg["min_words"], self.wcfg["max_windows"])
            windows += w
            owner += [i] * len(w)
        if windows:
            E = self.embedder.encode(windows) if getattr(self.scorer, "needs_embeddings", self.method == "zero_shot") else None
            W = self.scorer.window_scores(windows, E)
            own = np.asarray(owner)
            have = np.unique(own)
            S[have] = self.SX.aggregate_windows(W, own, n, self.agg)[have]
        ok = ~np.isnan(S).any(1)
        flags = np.zeros((n, L), dtype=bool)
        flags[ok] = S[ok] >= self.thr
        margin = np.where(flags, S - self.thr, -np.inf)
        top = np.where(flags.any(1), np.array(self.cats, dtype=object)[margin.argmax(1)], "none_unclear").astype(object)
        top[~ok] = None
        return {"S": S, "flags": flags, "ok": ok, "top": top, "none": ~flags.any(1)}


def unit_hash(keys: Sequence[str], seed: int) -> np.ndarray:
    """Deterministic U[0, 1) per key: sha256(seed:key) -> first 8 bytes. Same key + seed -> same value on every machine and run."""
    return np.array([int.from_bytes(hashlib.sha256(f"{seed}:{k}".encode("utf-8")).digest()[:8], "little") / 2.0 ** 64 for k in keys], dtype=float)


# ================================================================ the pipeline
class StressSignalPipeline:
    """Stress + emotion (+ stressor) inference with the training-time cleaning and chunking. Build with `from_artifacts`."""

    def __init__(self, stress: _StressComponent, emotion: _EmotionComponent, stressor: _StressorComponent | None,
                 settings: InferenceSettings, device: str, cfg: dict[str, Any] | None, info: dict[str, Any]):
        self.stress, self.emotion, self.stressor = stress, emotion, stressor
        self.settings, self.device, self.cfg, self.bundle_info = settings, device, cfg, info
        self.last_run: dict[str, Any] = {}
        self._lid: Any = None
        self._lid_lock = threading.Lock()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path, device: str = "auto", cfg: dict[str, Any] | None = None,
                       versions: dict[str, str] | None = None, verify: bool = True, embedder: Any | None = None,
                       **setting_overrides: Any) -> "StressSignalPipeline":
        """Load the latest (or pinned, `versions={"stress": "v20261004", ...}`) bundle of each component.

        `artifacts_dir` is the folder holding stress/, emotion/ and stressor/ (config paths.models = trained_models/). Every file in each
        loaded bundle is re-hashed against its manifest (`verify=False` skips this). `cfg` supplies the `inference`, `privacy` and
        `preprocess` sections; None loads config.yaml (pass it explicitly on Spark workers). `device`: auto | cuda | cpu (cuda without a
        usable GPU falls back to cpu). Missing stress or emotion bundle -> FileNotFoundError; a missing stressor bundle only drops its columns."""
        if cfg is None:
            from .config import load_config
            cfg = load_config()
        st = InferenceSettings.from_cfg(cfg, device=device if device != "auto" else (cfg.get("inference", {}) or {}).get("device", "auto"),
                                        **setting_overrides)
        dev = select_device({"inference": {"device": st.device}})
        root = Path(artifacts_dir)
        pins = {**((cfg.get("inference", {}) or {}).get("versions") or {}), **(versions or {})}
        chosen: dict[str, str | None] = {c: resolve_version(root, c, pins.get(c, "latest")) for c in COMPONENTS}
        for c in REQUIRED_COMPONENTS:
            if chosen[c] is None:
                raise FileNotFoundError(f"no {c} bundle (v<YYYYMMDD>/ with manifest.json) under {root / c}")
        if chosen["stressor"] is None:
            LOG.warning("no stressor bundle under %s: stressor_* columns are omitted (Step 5 has not produced one yet)", root)
        info: dict[str, Any] = {"artifacts_dir": root.resolve().as_posix(), "components": {}}
        for c, v in chosen.items():
            if v is None:
                continue
            d = root / c / v
            check = verify_bundle(d) if verify else {"manifest_sha256": sha256_file(d / "manifest.json"), "n_files": None}
            info["components"][c] = {"version": v, "path": f"{c}/{v}", "manifest_sha256": check["manifest_sha256"],
                                     "files_verified": check["n_files"]}
        stress = _StressComponent(root / "stress" / chosen["stress"], st, dev)
        emotion = _EmotionComponent(root / "emotion" / chosen["emotion"], st, dev)
        stressor = _StressorComponent(root / "stressor" / chosen["stressor"], st, dev, embedder) if chosen["stressor"] else None
        if stressor is not None:
            info["components"]["stressor"]["method"] = stressor.method
            info["components"]["stressor"]["lower_confidence_categories"] = stressor.lower_conf
        info["bundle_id"] = hashlib.sha256("|".join(f"{c}:{m['version']}:{m['manifest_sha256']}" for c, m in sorted(info["components"].items()))
                                           .encode()).hexdigest()[:16]
        info["device"], info["fp16"] = dev, bool(st.use_fp16 and dev == "cuda")
        pipe = cls(stress, emotion, stressor, st, dev, cfg, info)
        if st.cpu_int8:
            pipe.quantize_int8()
        return pipe

    def quantize_int8(self) -> "StressSignalPipeline":
        """OPTIONAL, CPU only: dynamic int8 quantisation of the Linear layers (torch.ao.quantization.quantize_dynamic). It changes the
        scores slightly: you MUST measure the difference on your own data (spark_jobs.int8_accuracy_check) before using it; the
        thresholds and calibration were fitted on fp32 outputs. Not applied on CUDA."""
        if self.device != "cpu":
            raise RuntimeError("int8 quantisation is for the CPU path only (device is %s)" % self.device)
        import torch
        runners = [self.stress.runner, self.emotion.runner] + ([self.stressor.runner] if self.stressor is not None and hasattr(self.stressor, "runner") else [])
        for r in runners:
            r.model = torch.ao.quantization.quantize_dynamic(r.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.bundle_info["int8"] = True
        return self

    # ------------------------------------------------------------ schema
    @property
    def has_stressor(self) -> bool:
        return self.stressor is not None

    def output_columns(self) -> list[str]:
        """Exact output columns, in order (Spark needs the schema up front)."""
        cols = ["input_status", "stress_prob_calibrated", "stress_flag"]
        cols += [f"emotion_p__{n}" for n in self.emotion.names] + [f"emotion_group__{g}" for g in self.emotion.groups]
        if self.stressor:
            cols += self.stressor.columns()
        if self.settings.tension_enabled:
            cols += ["tension_stress_strength", "tension_relaxation"]
        return cols + ["stress_version", "emotion_version", "stressor_version", "n_chunks_stress", "n_chunks_emotion"]

    # ------------------------------------------------------------ input handling
    def _language_identifier(self) -> Any:
        with self._lid_lock:
            if self._lid is None:
                from .preprocess import LanguageIdentifier
                self._lid = LanguageIdentifier(self.cfg)
            return self._lid

    def _prepare(self, texts: Sequence[Any]) -> tuple[list[str], list[str]]:
        """Clean every input; returns (clean text or "", status). Statuses other than ok/ok_truncated mean 'do not score'."""
        st = self.settings
        clean: list[str] = []
        status: list[str] = []

        def put(c: str, s: str) -> None:
            clean.append(c)
            status.append(s)

        for t in texts:
            if t is None or (isinstance(t, float) and math.isnan(t)) or t is pd.NA:
                put("", "none")
            elif not isinstance(t, str):
                put("", "non_string")
            elif not t.strip():
                put("", "empty")
            else:
                c = clean_text(t[: st.max_chars], self.cfg)
                if not c:
                    put("", "empty_after_cleaning")
                elif sum(ch.isalpha() for ch in PLACEHOLDER_RE.sub(" ", c)) < st.min_alpha_chars:
                    put("", "no_text")                                          # emoji-only, digits-only, punctuation-only, only <URL>/<USER> placeholders
                else:
                    put(c, "ok_truncated" if len(t) > st.max_chars else "ok")
        if st.english_only:
            idx = [i for i, s in enumerate(status) if s in OK_STATUSES]
            if idx:
                codes, _ = self._language_identifier().detect([clean[i] for i in idx])
                for i, code in zip(idx, codes):
                    if code != "en":
                        status[i] = "non_english"
        return clean, status

    # ------------------------------------------------------------ predict
    def predict(self, texts: Sequence[Any], batch_size: int | None = None, _second_stage: Any = None) -> pd.DataFrame:
        """Score `texts` (any length; None/empty/emoji-only/non-English allowed). One row per input, same order, RangeIndex.
        Unscorable inputs get NaN scores and a reason in `input_status`; no exception is raised for bad TEXT."""
        t0 = time.time()
        texts = list(texts)
        n = len(texts)
        bs = int(batch_size or self.settings.batch_size)
        t_prep = time.time()
        clean, status = self._prepare(texts)
        t_prep = time.time() - t_prep
        ok = np.array([s in OK_STATUSES for s in status], dtype=bool)
        vi = np.flatnonzero(ok)
        vt = [clean[i] for i in vi]
        em = self.emotion
        cols: dict[str, Any] = {"input_status": np.array(status, dtype=object)}
        p_stress, n_ch_s = np.full(n, np.nan), np.zeros(n, dtype=int)
        P = np.full((n, len(em.names)), np.nan)
        G = np.full((n, len(em.groups)), np.nan)
        n_ch_e = np.zeros(n, dtype=int)
        sel = np.zeros(len(vi), dtype=bool)                  # which valid rows get the emotion + stressor models (stage 2)
        t_stress = t_second = 0.0
        if len(vi):
            import torch
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            ps, cs = self.stress.score(vt, bs)
            t_stress = time.time() - t1
            p_stress[vi], n_ch_s[vi] = ps, cs
            sel = np.ones(len(vi), dtype=bool) if _second_stage is None else np.asarray(_second_stage(vi, ps), dtype=bool)
            t1 = time.time()
            if sel.any():
                vts = [vt[k] for k in np.flatnonzero(sel)]
                pe, ge, ce = em.score(vts, bs)
                rows = vi[sel]
                P[rows], G[rows], n_ch_e[rows] = pe, ge, ce
            t_second = time.time() - t1
        sel_full = np.zeros(n, dtype=bool)
        sel_full[vi[sel]] = True
        # a text whose chunking failed has NaN scores: record why
        failed = ok & (np.isnan(p_stress) | (sel_full & np.isnan(P).any(1)))
        for i in np.flatnonzero(failed):
            status[i] = "error_chunking"
        cols["input_status"] = np.array(status, dtype=object)
        cols["stress_prob_calibrated"] = p_stress
        cols["stress_flag"] = pd.array([pd.NA if np.isnan(p) else bool(p >= self.stress.threshold) for p in p_stress], dtype="boolean")
        for j, name in enumerate(em.names):
            cols[f"emotion_p__{name}"] = P[:, j]
        for j, g in enumerate(em.groups):
            cols[f"emotion_group__{g}"] = G[:, j]
        sx_ver = None
        if self.stressor:
            r = {"S": np.full((n, len(self.stressor.cats)), np.nan), "flags": np.zeros((n, len(self.stressor.cats)), dtype=bool),
                 "ok": np.zeros(n, dtype=bool), "top": np.full(n, None, dtype=object), "none": np.zeros(n, dtype=bool)}
            if sel.any():
                t1 = time.time()
                sub = self.stressor.score([vt[k] for k in np.flatnonzero(sel)])
                t_second += time.time() - t1
                for k in ("S", "flags", "ok", "top", "none"):
                    r[k][vi[sel]] = sub[k]
            if self.settings.gate_stressors:
                # zero-shot thresholds are loose (neutral text gets tagged), so only texts flagged as stressed may carry stressor tags;
                # raw scores are kept. Un-flagged texts read as none_unclear.
                off = r["ok"] & ~np.asarray(cols["stress_flag"].fillna(False), dtype=bool)
                r["flags"][off] = False
                r["top"][off] = "none_unclear"
                r["none"][off] = True
            for j, c in enumerate(self.stressor.cats):
                cols[f"stressor_{_StressorComponent.PREFIX[self.stressor.kinds[c]]}__{c}"] = r["S"][:, j]
            for j, c in enumerate(self.stressor.cats):
                cols[f"stressor_flag__{c}"] = pd.array([bool(r["flags"][i, j]) if r["ok"][i] else pd.NA for i in range(n)], dtype="boolean")
            cols["stressor_top"] = r["top"]
            cols["stressor_none_unclear"] = pd.array([bool(r["none"][i]) if r["ok"][i] else pd.NA for i in range(n)], dtype="boolean")
            sx_ver = self.bundle_info["components"]["stressor"]["version"]
        if self.settings.tension_enabled:
            ts, tr_ = np.full(n, np.nan), np.full(n, np.nan)
            if len(vi):
                from .external_validation import tensistrength_batch
                sc, _ = tensistrength_batch(self.cfg, vt)
                ts[vi], tr_[vi] = sc["stress_strength"].to_numpy(), sc["relaxation"].to_numpy()
            cols["tension_stress_strength"], cols["tension_relaxation"] = ts, tr_
        comp = self.bundle_info["components"]
        cols["stress_version"] = np.full(n, comp["stress"]["version"], dtype=object)
        cols["emotion_version"] = np.full(n, comp["emotion"]["version"], dtype=object)
        cols["stressor_version"] = np.full(n, sx_ver, dtype=object)
        cols["n_chunks_stress"], cols["n_chunks_emotion"] = n_ch_s, n_ch_e
        df = pd.DataFrame(cols, index=pd.RangeIndex(n))[self.output_columns()]
        run: dict[str, Any] = {"n_texts": n, "n_scored": int(len(vi)), "n_second_stage": int(sel.sum()), "seconds": round(time.time() - t0, 3),
                               "seconds_prepare": round(t_prep, 3), "seconds_stress": round(t_stress, 3), "seconds_second_stage": round(t_second, 3), "device": self.device,
                               "batch_size": bs, "status_counts": pd.Series(status).value_counts().to_dict()}
        if self.device == "cuda":
            import torch
            run["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        self.last_run = run
        df.attrs["bundle_id"] = self.bundle_info["bundle_id"]
        return df


    def predict_two_stage(self, texts: Sequence[Any], record_ids: Sequence[str], sample_rate: float = 0.15, seed: int = 42,
                          batch_size: int | None = None) -> pd.DataFrame:
        """Compute-saving two-stage inference. Stage 1: the stress model scores EVERY text. Stage 2 (emotion + stressor models) runs only on
        stress-flagged texts plus a seeded random sample of the unflagged ones. The sample is decided per record_id (hash of seed + id), so
        it does not depend on batching, partitioning or run order. Adds two columns:
          second_stage_scored  True where the emotion/stressor models ran
          sample_weight        1.0 for flagged texts, 1/sample_rate for sampled unflagged texts, NaN for unsampled (and unscorable) texts
        Step 10 reweights non-stress emotion estimates with sample_weight; flagged-only estimates need no weights."""
        rate = float(sample_rate)
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        ids = list(record_ids)
        if len(ids) != len(list(texts)):
            raise ValueError("record_ids and texts differ in length")
        u = unit_hash(ids, seed)
        thr = self.stress.threshold
        flagged_at: dict[str, np.ndarray] = {}

        def second(vi: np.ndarray, ps: np.ndarray) -> np.ndarray:
            flagged = ps >= thr
            flagged_at["flag"], flagged_at["vi"] = flagged, vi
            return flagged | (u[vi] < rate)

        df = self.predict(texts, batch_size, _second_stage=second)
        n = len(df)
        scored = np.zeros(n, dtype=bool)
        weight = np.full(n, np.nan)
        if "vi" in flagged_at:
            vi, fl = flagged_at["vi"], flagged_at["flag"]
            samp = u[vi] < rate
            scored[vi] = fl | samp
            weight[vi[fl]] = 1.0
            weight[vi[~fl & samp]] = 1.0 / rate
        df["second_stage_scored"] = scored
        df["sample_weight"] = weight
        return df


# ================================================================ process-level cache (Spark workers)
_CACHE: dict[str, StressSignalPipeline] = {}
_CACHE_LOCK = threading.Lock()


def get_pipeline(artifacts_dir: str | Path, device: str = "auto", cfg: dict[str, Any] | None = None,
                 versions: dict[str, str] | None = None, **setting_overrides: Any) -> StressSignalPipeline:
    """One pipeline per Python process: the first call loads the models, later calls with the same arguments reuse them.
    Use this inside Spark `mapInPandas` / pandas UDFs so each worker loads the models once (never at import time, never on the driver)."""
    key_cfg = json.dumps({k: (cfg or {}).get(k) for k in ("inference", "privacy", "preprocess")}, sort_keys=True, default=str)
    key = json.dumps([str(Path(artifacts_dir).resolve()), device, key_cfg, versions or {}, setting_overrides], sort_keys=True, default=str)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = StressSignalPipeline.from_artifacts(artifacts_dir, device, cfg, versions, **setting_overrides)
        return _CACHE[key]


def clear_pipeline_cache() -> None:
    """Drop cached pipelines (tests, or to free GPU memory)."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ================================================================ bundle manifest
def write_bundle_manifest(artifacts_dir: str | Path, out_path: str | Path, versions: dict[str, str] | None = None,
                          cfg: dict[str, Any] | None = None, verify: bool = True) -> dict[str, Any]:
    """Write the manifest listing the EXACT component versions that are used together (and their manifest hashes).
    Reads bundle files only (no model is loaded, nothing is run). `from_bundle_manifest` can reload exactly this set."""
    root = Path(artifacts_dir)
    pins = {**((cfg or {}).get("inference", {}).get("versions") or {}), **(versions or {})}
    comps: dict[str, Any] = {}
    for c in COMPONENTS:
        v = resolve_version(root, c, pins.get(c, "latest"))
        if v is None:
            continue
        d = root / c / v
        check = verify_bundle(d) if verify else {"manifest_sha256": sha256_file(d / "manifest.json"), "n_files": None}
        bm = _read_json(d / "manifest.json")
        comps[c] = {"version": v, "path": f"{c}/{v}", "manifest_sha256": check["manifest_sha256"], "files_verified": check["n_files"],
                    "created_utc": bm.get("created_utc"), "seeds": bm.get("seeds"), "dataset_versions": bm.get("dataset_versions"),
                    "weights_sha256": next((m["sha256"] for k, m in (bm.get("artifacts") or {}).items()
                                           if k in ("model.safetensors", "lr.npz") and m.get("type") == "file"), None)}
    missing = [c for c in REQUIRED_COMPONENTS if c not in comps]
    if missing:
        raise FileNotFoundError(f"cannot write a bundle manifest: missing required component(s) {missing}")
    out = {
        "bundle_id": hashlib.sha256("|".join(f"{c}:{m['version']}:{m['manifest_sha256']}" for c, m in sorted(comps.items())).encode()).hexdigest()[:16],
        "components": comps,
        "missing_optional_components": [c for c in COMPONENTS if c not in comps],
        "package_version": PACKAGE_VERSION, "library_versions": library_versions(), "scope": SCOPE_NOTE,
        "created_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
    }
    write_json(out_path, out)
    return out


def versions_from_bundle_manifest(path: str | Path) -> dict[str, str]:
    """{component: version} pinned by a bundle manifest."""
    return {c: m["version"] for c, m in _read_json(Path(path))["components"].items()}


def check_bundle_manifest(artifacts_dir: str | Path, manifest_path: str | Path) -> None:
    """Raise BundleIntegrityError unless every component of the manifest exists with the recorded manifest hash."""
    for c, m in _read_json(Path(manifest_path))["components"].items():
        mp = Path(artifacts_dir) / m["path"] / "manifest.json"
        if not mp.is_file() or sha256_file(mp) != m["manifest_sha256"]:
            raise BundleIntegrityError(f"{c} {m['version']}: bundle manifest.json differs from bundle_manifest.json (bundle changed)")
