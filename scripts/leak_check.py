"""Mide la fuga entre casos gemelos con un "buscador" sin conocimiento médico.

Para cada texto de evaluación busca el texto de entrenamiento más parecido (TF-IDF), compara oración
por oración y copia la respuesta del gemelo. Se corre con dos divisiones 80/20:
  - al azar por Text ID (la del notebook original)
  - por grupo de caso (la corregida)

Uso:
    python scripts/leak_check.py --data data/MEDIQA-CORR-2024-MS-TrainingData.csv
Resultado esperado: por Text ID A ≈ 91,8 %, B ≈ 84,2 %; por grupo A ≈ 55,7 %, B ≈ 34,9 %.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mediqa_corr.data import clean, load_data, make_case_groups, parse_sentences  # noqa: E402
from mediqa_corr.evaluation import eval_flag_accuracy, eval_sentence_accuracy, rouge1_f  # noqa: E402

_tok = lambda x: set(re.findall(r"\w+", x.lower()))  # noqa: E731


def lookup_system(train, ev, sim_threshold=0.7):
    """Predice A, B y C copiando la respuesta del texto de train más parecido."""
    vec = TfidfVectorizer().fit(train.Text)
    S = cosine_similarity(vec.transform(ev.Text), vec.transform(train.Text))
    tr_sents = [[clean(t) for _, t in parse_sentences(s)] for s in train.Sentences]
    out = []
    for k, (_, v) in enumerate(ev.iterrows()):
        sv = [clean(t) for _, t in parse_sentences(v.Sentences)]
        j = int(S[k].argmax())
        t = train.iloc[j]
        flag, sid, corr = 1, len(sv) - 1, sv[-1]          # sin gemelo: error en la última oración
        if S[k, j] > sim_threshold:
            st = tr_sents[j]
            if t["Error Flag"] == 1:                         # el gemelo es la versión con error
                e = clean(t["Error Sentence"])
                if e in sv:
                    flag, sid, corr = 1, sv.index(e), clean(t["Corrected Sentence"])
                else:
                    flag, sid, corr = 0, -1, "NA"
            else:                                            # el gemelo es la versión correcta
                diff = [i for i, s in enumerate(sv) if s not in set(st)]
                if diff:
                    sid = diff[-1]
                    cand = [s for s in st if s not in set(sv)] or st
                    corr = max(cand, key=lambda c: len(_tok(c) & _tok(sv[sid])) / max(1, len(_tok(c) | _tok(sv[sid]))))
                    flag = 1
                else:
                    flag, sid, corr = 0, -1, "NA"
        out.append((flag, sid, corr))
    p = pd.DataFrame(out, columns=["f", "s", "c"])
    both = (ev["Error Flag"].values == 1) & (p.f.values == 1)
    refs = ev["Corrected Sentence"].values[both]
    return {
        "A": round(eval_flag_accuracy(ev["Error Flag"], p.f), 3),
        "B": round(eval_sentence_accuracy(ev["Error Sentence ID"], p.s), 3),
        "ROUGE1": round(float(np.mean([rouge1_f(c, clean(r)) for c, r in zip(p.c[both], refs)])), 3),
        "correcciones_exactas": round(float(np.mean([c == clean(r) for c, r in zip(p.c[both], refs)])), 3),
        "textos_eval": len(ev),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df = load_data(a.data)
    df["group"] = make_case_groups(df)

    tr, va = train_test_split(df, test_size=0.2, stratify=df["Error Flag"], random_state=42)
    print("Split por Text ID (original):", lookup_system(tr.reset_index(drop=True), va.reset_index(drop=True)))

    i_tr, i_ev = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                      .split(df, df["Error Flag"], groups=df["group"]))
    print("Split por grupo (corregido): ", lookup_system(df.iloc[i_tr].reset_index(drop=True),
                                                         df.iloc[i_ev].reset_index(drop=True)))


if __name__ == "__main__":
    main()
