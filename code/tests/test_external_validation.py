"""Step 6 tests. NOTHING here trains a model, loads one, or runs the TensiStrength jar: only parsing, statistics, the
license/Java gate, the verdict rules and report rendering on small synthetic inputs."""

import copy

import numpy as np
import pandas as pd
import pytest

from stress_signals.config import load_config
from stress_signals import external_validation as X


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------- TensiStrength parsing + gate
def test_parse_line():
    assert X.parse_tensistrength_line("4 -1") == (4, -1)
    assert X.parse_tensistrength_line("  1\t-4 extra") == (1, -4)
    assert X.parse_tensistrength_line("hello") is None


def test_parse_output_ok_and_scale():
    df = X.parse_tensistrength_output("1 -4\n4 -1\n1 -1\n\n", 3)
    assert df["stress_strength"].tolist() == [4, 1, 1] and df["relaxation"].tolist() == [1, 4, 1]


def test_parse_output_refuses_misalignment_and_bad_scale():
    with pytest.raises(ValueError, match="alignment"):
        X.parse_tensistrength_output("1 -1\n1 -1\n", 3)
    with pytest.raises(ValueError, match="cannot parse"):
        X.parse_tensistrength_output("1 -1\nabc\n", 2)
    with pytest.raises(ValueError, match="observed scale"):
        X.parse_tensistrength_output("1 3\n", 1)            # stress must be negative on the observed scale


def test_gate_blocks_until_academic_use_confirmed(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["external_validation"]["tensistrength"]["academic_use_confirmed"] = False     # the shipped config has the user's confirmation
    g = X.tensistrength_gate(cfg)
    assert g["usable"] is False and any("academic" in r for r in g["reasons"])
    with pytest.raises(RuntimeError, match="not usable"):
        X.tensistrength_batch(cfg, ["i am stressed"])
    c2 = copy.deepcopy(cfg)
    c2["external_validation"]["tensistrength"]["academic_use_confirmed"] = True
    assert not any("academic" in r for r in X.tensistrength_gate(c2)["reasons"])


# ---------------------------------------------------------------- statistics
def test_spearman_ci_known_values():
    x = np.arange(200.0)
    r = X.spearman_ci(x, x, n=50)
    assert r["rho"] == pytest.approx(1.0) and r["ci"][0] > 0.99
    r = X.spearman_ci(x, -x, n=50)
    assert r["rho"] == pytest.approx(-1.0)
    assert np.isnan(X.spearman_ci(x, np.ones(200), n=10)["rho"])       # constant side -> undefined, not 0
    g = np.repeat(np.arange(20), 10)
    assert X.spearman_ci(x, x, groups=g, n=30)["resampling"] == "whole groups"


def test_wilson():
    lo, hi = X.wilson(12, 12)
    assert 0.7 < lo < 1.0 and hi == pytest.approx(1.0)
    assert np.isnan(X.wilson(0, 0)[0])
    lo, hi = X.wilson(5, 10)
    assert lo < 0.5 < hi


def test_noise_ceiling_formulas():
    c = X.noise_ceiling(1.0, 0.0, 0.5)                                   # no noise: a perfect classifier agrees perfectly
    assert c["roc_auc"] == 1.0 and c["accuracy"] == 1.0 and c["f1_stress"] == pytest.approx(1.0)
    c = X.noise_ceiling(0.5, 0.5, 0.5)                                   # labels independent of truth: nothing beats chance
    assert c["roc_auc"] == pytest.approx(0.5) and c["accuracy"] == pytest.approx(0.5)
    c = X.noise_ceiling(0.9, 0.2, 0.7)
    assert c["roc_auc"] == pytest.approx(0.85)
    assert c["accuracy"] == pytest.approx(0.7 * 0.9 + 0.3 * 0.8)
    assert 0 < c["f1_stress"] < 1


def test_noise_ceiling_matches_simulation():
    rng = np.random.default_rng(0)
    n = 200_000
    truth = rng.random(n) < 0.6
    auto = np.where(truth, rng.random(n) < 0.9, rng.random(n) < 0.2).astype(int)   # P(auto=1|true)=.9 / .2
    ppv1, q0 = truth[auto == 1].mean(), truth[auto == 0].mean()
    from sklearn.metrics import roc_auc_score
    assert X.noise_ceiling(ppv1, q0, auto.mean())["roc_auc"] == pytest.approx(roc_auc_score(auto, truth.astype(int)), abs=1e-3)


def test_evaluate_block_and_delta():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.2, 600), 0.01, 0.99)
    blk, dr = X.evaluate_block(y, p, 0.5, None, n_boot=60, seed=3)
    lo, hi = blk["ci"]["roc_auc"]
    assert lo <= blk["point"]["roc_auc"] <= hi and 0 < blk["point"]["ece"] < 0.5
    d = X.delta_block(blk["point"], dr, blk["point"], dr)
    assert d["roc_auc"]["delta"] == 0 and d["roc_auc"]["ci"] == [0.0, 0.0]


# ---------------------------------------------------------------- real processed data (read only)
def test_senticnet_eval_frame_is_clean(cfg):
    p = X.Path(cfg["_paths"]["processed"]) / "senticnet_validation.parquet"
    if not p.exists():
        pytest.skip("senticnet_validation.parquet not present")
    df, info = X.load_senticnet_eval(cfg)
    assert info["rows_evaluated"] == len(df) <= info["rows_in_file"]
    assert set(df["platform"]) <= {"reddit", "twitter"} and set(df["label_auto"]) <= {0, 1}
    assert not df.duplicated(["platform", "text_clean"]).any()
    assert "author" not in {c.lower() for c in df.columns}


# ---------------------------------------------------------------- verdict + report
def _zs(auc, ci, ece_v, delta=-0.05, delta_ci=(-0.08, -0.02), twitter_auc=0.8):
    def blk(a, c):
        return {"n": 1000, "prevalence_label_1": 0.5, "point": {"roc_auc": a, "ece": ece_v, "f1_stress": .7, "pr_auc": .7},
                "ci": {"roc_auc": list(c), "ece": [ece_v - .01, ece_v + .01], "f1_stress": [.6, .8], "pr_auc": [.6, .8]},
                "delta_vs_dreaddit_test": {k: {"delta": delta, "ci": list(delta_ci)} for k in ("roc_auc", "f1_stress", "pr_auc", "ece")}}
    return {"status": "done", "platforms": {"reddit": blk(auc, ci), "twitter": blk(twitter_auc, (twitter_auc - .02, twitter_auc + .02))}}


def _noise(ceiling_auc=0.92, ci=(0.85, 0.97)):
    return {"status": "complete", "label_source": "human", "per_platform": {"reddit": {"ceiling": {"roc_auc": {"point": ceiling_auc, "ci": list(ci)}}}}}


def test_verdict_retrain_when_signal_is_absent(cfg):
    v = X.decide_verdict(_zs(0.60, (0.57, 0.64), 0.1), _noise(), cfg)
    assert v["verdict"] == "retrain"


def test_verdict_ship_needs_ceiling_and_all_conditions(cfg):
    good = _zs(0.88, (0.85, 0.91), 0.05)
    assert X.decide_verdict(good, _noise(0.92, (0.90, 0.94)), cfg)["verdict"] == "ship"
    pend = X.decide_verdict(good, {"status": "pending"}, cfg)
    assert pend["verdict"] == "ship with caveats" and pend["provisional"] is True
    assert any("S2" in c for c in pend["caveats"])


def test_verdict_caveats_for_ece_and_large_drop(cfg):
    v = X.decide_verdict(_zs(0.85, (0.82, 0.88), 0.2, delta=-0.2, delta_ci=(-0.25, -0.15)), _noise(0.92, (0.90, 0.94)), cfg)
    assert v["verdict"] == "ship with caveats"
    assert any("S3" in c for c in v["caveats"]) and any("C1" in c for c in v["caveats"])


def test_verdict_without_zero_shot(cfg):
    assert X.decide_verdict(None, None, cfg)["verdict"] == "not available"


def test_report_renders_not_run_sections(cfg):
    rep = {"created_utc": "t", "bundle": "v0", "senticnet_zero_shot": {"status": "not_run", "reason": "x"},
           "label_noise": {"status": "pending", "reason": "no human_label filled in yet (n_filled = 0)", "n_filled": 0,
                           "n_rows_in_sheet": 48, "per_platform": {}},
           "tensistrength_run": {"status": "not_run"}, "tensistrength": {"status": "not_run", "reason": ["academic use"]},
           "cross_training": {}, "verdict": X.decide_verdict(None, None, cfg)}
    md = X.render_markdown(rep)
    assert "PENDING" in md and "NOT RUN" in md and "Why not run TensiStrength on every record" in md
    assert "AUTOMATED" in md


def test_report_renders_full_results(cfg):
    zs = _zs(0.88, (0.85, 0.91), 0.05)
    zs.update({"model": {"base": "roberta-base", "seed": 2024, "temperature": 1.37, "threshold": 0.5},
               "frame": {"rows_evaluated": 10, "rows_in_file": 12, "dropped_duplicate_rows": 0}, "files": {},
               "dreaddit_test": zs["platforms"]["reddit"]})
    rep = {"created_utc": "t", "bundle": "v0", "senticnet_zero_shot": zs, "label_noise": {"status": "pending", "per_platform": {}},
           "tensistrength_run": {"status": "not_run"}, "tensistrength": {"status": "not_run", "reason": ["x"]}, "cross_training": {},
           "verdict": X.decide_verdict(zs, _noise(), cfg)}
    md = X.render_markdown(rep)
    assert "SenticNet reddit" in md and "0.880" in md and "| R1 |" in md


def test_machine_labelled_hand_check_caps_verdict(cfg):
    good = _zs(0.88, (0.85, 0.91), 0.05)
    n = _noise(0.92, (0.90, 0.94))
    n["label_source"] = "machine"
    v = X.decide_verdict(good, n, cfg)
    assert v["verdict"] == "ship with caveats" and v["provisional"] is True and "machine" in v["provisional_reason"]
    assert any("S5" in c for c in v["caveats"])


def test_handcheck_noise_reads_machine_sheet(cfg, tmp_path, monkeypatch):
    c2 = copy.deepcopy(cfg)
    c2["_paths"] = {**cfg["_paths"], "interim": tmp_path}
    c2["external_validation"]["out_dir"] = str(tmp_path / "out")
    c2["external_validation"]["handcheck"]["label_source"] = "machine"
    rows = []
    for f in ("Reddit_Title", "Reddit_Combi", "Twitter_Full", "Twitter_Non-Advert"):
        for a in (0, 1):
            for i in range(6):
                rows.append({"file": f, "row_index": len(rows), "auto_label": a, "machine_label": a if i < 5 else 1 - a, "machine_score": .5})
    pd.DataFrame(rows).to_csv(tmp_path / "senticnet_handcheck_machine.csv", index=False)
    r = X.handcheck_noise(c2, prevalence={"reddit": 0.6, "twitter": 0.5})
    assert r["label_source"] == "machine" and "NOT by a person" in r["banner"] and r["status"] == "complete"
    assert r["per_platform"]["reddit"]["agreement"]["point"] == pytest.approx(20 / 24)
    assert 0.5 < r["per_platform"]["reddit"]["ceiling"]["roc_auc"]["point"] < 1
