"""Caja 4 · Evaluación: exactitud de A y B, ROUGE-1 y BERTScore para C, y Composite Score.

Configurado como el script oficial de MEDIQA-CORR 2024 (mediqa-corr-2024-eval-script-2-all-metrics.py):
paquete `rouge` para ROUGE-1 F, BERTScore con deberta-xlarge-mnli y rescale_with_baseline=True,
recorte a [0, 1]. BLEURT-20 no se calcula (depende de TensorFlow).
"""
import gc

import numpy as np
import torch
from rouge import Rouge

from .config import BERTSCORE_MODEL
from .data import clean


def eval_flag_accuracy(y_true, y_pred):
    """Exactitud de la subtarea A."""
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def eval_sentence_accuracy(true_ids, pred_ids):
    """Exactitud de la subtarea B (un texto sin error se representa con -1)."""
    return float(np.mean(np.array(true_ids) == np.array(pred_ids)))


def aggregate_score(rouge1, bertscore=None, bleurt=None):
    """Promedio de las métricas de texto disponibles (el oficial usa las tres)."""
    valores = [v for v in (rouge1, bertscore, bleurt) if v is not None]
    return float(np.mean(valores))


_BERT_SCORER = None


def get_bert_scorer(model_type=BERTSCORE_MODEL):
    """BERTScorer cargado una sola vez.

    En transformers 5 el tokenizador de DeBERTa reporta model_max_length ~1e30 y bert_score lo pasa
    como max_length, lo que produce OverflowError. Se acota al largo real del modelo (512).
    """
    global _BERT_SCORER
    if _BERT_SCORER is None or _BERT_SCORER.model_type != model_type:
        from bert_score import BERTScorer
        _BERT_SCORER = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=True,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
        if _BERT_SCORER._tokenizer.model_max_length > 100_000:
            _BERT_SCORER._tokenizer.model_max_length = 512
    return _BERT_SCORER


def release_bert_scorer():
    """Libera la memoria de BERTScore (~3 GB) antes de entrenar; se vuelve a cargar cuando haga falta."""
    global _BERT_SCORER
    _BERT_SCORER = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_ROUGE = Rouge()


def rouge1_f(pred, ref):
    """ROUGE-1 F del paquete `rouge` (el del script oficial). Una predicción vacía puntúa 0 en vez de fallar."""
    try:
        return _ROUGE.get_scores(pred, ref)[0]["rouge-1"]["f"]
    except ValueError:            # "Hypothesis is empty" (p. ej. solo signos de puntuación)
        return 0.0


def _no_vacia(s):
    """bert_score falla con cadenas vacías en transformers 5; se reemplazan por '.' (puntaje ~0)."""
    s = clean(s)
    return s if s else "."


def eval_correction(preds, refs, bertscore=True, bert_model=BERTSCORE_MODEL):
    """ROUGE-1 F (y BERTScore F1) entre correcciones generadas y de referencia.

    Devuelve las medias, el AggregateScore (sin BLEURT) y 'per_text': el puntaje de cada par.
    """
    preds = [_no_vacia(p) for p in preds]
    refs = [_no_vacia(r) for r in refs]
    rouge = [rouge1_f(p, r) for p, r in zip(preds, refs)]
    out = {"rouge1": float(np.mean(rouge))}
    por_texto = list(rouge)
    if bertscore:
        _, _, f1 = get_bert_scorer(bert_model).score(preds, refs)
        f1 = np.clip(f1.numpy(), 0, 1)
        out["bertscore"] = float(f1.mean())
        por_texto = [(r + b) / 2 for r, b in zip(rouge, f1.tolist())]
    out["aggregate"] = aggregate_score(out["rouge1"], out.get("bertscore"))
    out["per_text"] = por_texto
    return out


def _es_na(x):
    return x is None or str(x).strip().upper() in ("", "NA", "NAN")


def composite_score(pred_correction, ref_correction, text_score=None):
    """1 si ambas son NA, 0 si solo una lo es, y el puntaje de texto si ninguna lo es."""
    if _es_na(pred_correction) and _es_na(ref_correction):
        return 1.0
    if _es_na(pred_correction) or _es_na(ref_correction):
        return 0.0
    return float(text_score)


def evaluate_all(pred_df, gold_df, bertscore=True, bert_model=BERTSCORE_MODEL):
    """Métricas de A, B y C, más el Composite Score promedio sobre todos los textos."""
    m = gold_df.merge(pred_df, on="Text ID", suffixes=("", "_pred"))
    res = {"flag_acc": eval_flag_accuracy(m["Error Flag"], m["Error Flag_pred"]),
           "sentence_acc": eval_sentence_accuracy(m["Error Sentence ID"], m["Error Sentence ID_pred"])}
    ambos = m[(m["Error Flag"] == 1) & (m["Error Flag_pred"] == 1)]
    por_texto = {}
    if len(ambos):
        c = eval_correction([clean(x) for x in ambos["Corrected Sentence_pred"]],
                            [clean(x) for x in ambos["Corrected Sentence"]],
                            bertscore=bertscore, bert_model=bert_model)
        por_texto = dict(zip(ambos["Text ID"], c.pop("per_text")))
        res.update(c)
    res["pares_de_correccion"] = int(len(ambos))
    res["composite"] = float(np.mean([composite_score(r["Corrected Sentence_pred"], r["Corrected Sentence"],
                                                      por_texto.get(r["Text ID"]))
                                      for _, r in m.iterrows()]))
    return res
