"""Machine-labelling tests. No NLI model is loaded: only the label logic, the model-label check and the provenance
plumbing (a machine sheet must be readable by the gold-set code and every report must say MACHINE-LABELLED)."""

import copy

import numpy as np
import pandas as pd
import pytest

from stress_signals import machine_labeler as ML
from stress_signals import stressor_model as ST
from stress_signals.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_entailment_indices_read_from_model_labels():
    assert ML.entailment_indices({0: "contradiction", 1: "neutral", 2: "entailment"}) == (0, 2)
    assert ML.entailment_indices({0: "ENTAILMENT", 1: "neutral", 2: "Contradiction"}) == (2, 0)
    with pytest.raises(ValueError):
        ML.entailment_indices({0: "LABEL_0", 1: "LABEL_1"})


def test_stressor_labels_rules():
    cats = ["workplace_pressure", "financial_concerns"]
    scores = {"workplace_pressure": np.array([0.9, 0.2, 0.2, 0.7]), "financial_concerns": np.array([0.8, 0.1, 0.3, 0.1])}
    generic = np.array([0.1, 0.9, 0.2, 0.1])
    out = ML.stressor_labels_from_scores(scores, generic, 0.5, cats)
    assert out.loc[0, ["workplace_pressure", "financial_concerns"]].tolist() == [1, 1]          # multi-label
    assert out.loc[1, ["other_unclear", "none_unclear"]].tolist() == [1, 0]                      # cause named, fits none
    assert out.loc[2, ["other_unclear", "none_unclear"]].tolist() == [0, 1]                      # nothing named
    assert out.loc[3, ["workplace_pressure", "other_unclear", "none_unclear"]].tolist() == [1, 0, 0]
    assert (out.sum(axis=1) >= 1).all()
    assert not ((out["none_unclear"] == 1) & (out.drop(columns="none_unclear").sum(axis=1) > 0)).any()


def test_hypotheses_cover_every_taxonomy_category(cfg):
    tax = ST.load_taxonomy(cfg)
    need = [c for c in ST.category_ids(tax) if c != "other_unclear"]
    assert not [c for c in need if c not in cfg["machine_labeler"]["stressor_hypotheses"]]


def test_gold_provenance_switches_with_label_source(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["_paths"] = {**cfg["_paths"], "gold": tmp_path}
    c["stressor_model"]["gold"]["label_source"] = "machine"
    f = ST.gold_files(c)
    assert f["sheet1"].name.endswith("_machine.csv") and f["human_sheet"].name.endswith("_annotator1.csv")
    assert "MACHINE-LABELLED" in ST.annotation_note(c) and "NOT by a person" in ST.annotation_limitation(c)
    c["stressor_model"]["gold"]["label_source"] = "human"
    assert ST.gold_files(c)["sheet1"].name.endswith("_annotator1.csv") and ST.annotation_note(c) == ST.ANNOTATION_NOTE
    c["stressor_model"]["gold"]["label_source"] = "robot"
    with pytest.raises(ValueError):
        ST.label_source(c)


def test_machine_gold_sheet_is_readable_by_gold_code(cfg, tmp_path):
    tax = ST.load_taxonomy(cfg)
    cols = ST.gold_columns(tax)
    df = pd.DataFrame({"gold_id": ["g1", "g2"], **{c: [0, 0] for c in cols}, "notes": "machine label"})
    df.loc[0, "workplace_pressure"], df.loc[1, "none_unclear"] = 1, 1
    p = tmp_path / "m.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    lab, probs = ST.read_sheet(p, cols)
    assert probs == [] and lab["annotated"].all()
