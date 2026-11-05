"""Pruebas rápidas sin GPU ni descargas de modelos. Uso: pytest -q"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mediqa_corr.baselines import parse_llm_output  # noqa: E402
from mediqa_corr.data import load_data, make_case_groups, parse_sentences, split_train_dev_test  # noqa: E402
from mediqa_corr.detector import choose_threshold, decide_flag, pick_error_sentence  # noqa: E402
from mediqa_corr.evaluation import (composite_score, eval_correction, eval_flag_accuracy,  # noqa: E402
                                    eval_sentence_accuracy, evaluate_all)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "MEDIQA-CORR-2024-MS-TrainingData.csv")


def test_parse_sentences_continuacion():
    s = "0 Primera.\n1 Laboratorio:\n37 C\n2 Final."
    assert parse_sentences(s) == [(0, "Primera."), (1, "Laboratorio:\n37 C"), (2, "Final.")]


def test_metricas_ab():
    assert eval_flag_accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    assert eval_sentence_accuracy([6, -1, 2], [6, -1, 3]) == 2 / 3


def test_rouge_y_vacios():
    r = eval_correction(["the cat sat"], ["the cat sat down"], bertscore=False)
    assert abs(r["rouge1"] - 6 / 7) < 1e-6
    r = eval_correction(["", "the cat sat"], ["a dog", "the cat sat"], bertscore=False)
    assert r["per_text"][0] == 0 and r["per_text"][1] > 0.99


def test_composite():
    assert composite_score("NA", "NA") == 1
    assert composite_score("NA", "algo") == 0
    assert composite_score("algo", "NA") == 0
    assert composite_score("x", "y", 0.4) == 0.4


def test_decision():
    assert decide_flag([0.1, 0.7, 0.3], 0.55) == 1
    assert pick_error_sentence([0.1, 0.7, 0.3], 1) == 1
    assert pick_error_sentence([0.1, 0.2], 0) == -1
    assert choose_threshold([[0.9, 0.1], [0.2, 0.3]], [1, 0]) <= 0.9


def test_parse_llm():
    assert parse_llm_output("CORRECT") == (0, -1, "NA")
    assert parse_llm_output("6 The pathogen was Streptococcus.") == (1, 6, "The pathogen was Streptococcus.")


def test_evaluate_all():
    gold = pd.DataFrame({"Text ID": ["a", "b"], "Error Flag": [1, 0], "Error Sentence ID": [2, -1],
                         "Corrected Sentence": ["the dog ran", "NA"]})
    pred = pd.DataFrame({"Text ID": ["a", "b"], "Error Flag": [1, 0], "Error Sentence ID": [2, -1],
                         "Corrected Sentence": ["the dog ran", "NA"]})
    r = evaluate_all(pred, gold, bertscore=False)
    assert r["flag_acc"] == 1 and r["sentence_acc"] == 1 and abs(r["composite"] - 1) < 1e-6


@pytest.mark.skipif(not os.path.exists(DATA), reason="falta el CSV en data/")
def test_dataset_y_split_sin_fuga():
    df = load_data(DATA)
    assert len(df) == 2189 and df["Error Flag"].sum() == 1219
    # la segmentación oficial coincide con la oración errónea en todas las filas con error
    for _, r in df[df["Error Flag"] == 1].iterrows():
        assert " ".join(dict(parse_sentences(r["Sentences"]))[r["Error Sentence ID"]].split()) == \
               " ".join(r["Error Sentence"].split())
    df["group"] = make_case_groups(df)
    train, dev, test = split_train_dev_test(df)
    assert (len(train), len(dev), len(test)) == (1563, 314, 312)
    for a, b in [(train, dev), (train, test), (dev, test)]:
        assert set(a["group"]).isdisjoint(set(b["group"]))
