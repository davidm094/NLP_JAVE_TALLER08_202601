"""Pipeline completo: detector + corrector sobre cada texto, archivo de envío y orquestación del experimento."""
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .baselines import baseline_always_error, baseline_copy_sentence, baseline_last_sentence, baseline_never_error
from .config import BERTSCORE_MODEL, SEED, TrainConfig
from .corrector import build_correction_pairs, build_correction_source, generate_correction, train_corrector
from .data import (build_sentence_table, clean, describe_dataset, load_data, make_case_groups, parse_sentences,
                   sample_groups, split_train_dev_test)
from .detector import choose_threshold, decide_flag, pick_error_sentence, predict_sentence_scores, train_error_classifier
from .evaluation import eval_correction, eval_flag_accuracy, eval_sentence_accuracy, evaluate_all, release_bert_scorer


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_pipeline(df, detector, det_tokenizer, corrector, cor_tokenizer, threshold, num_beams=4):
    """Predice A, B y C para cada texto de df. Devuelve un DataFrame con una fila por texto.

    Si el flag es 0, la salida es NA y no se llama al corrector.
    """
    filas = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="pipeline"):
        oraciones = parse_sentences(r["Sentences"])
        scores = predict_sentence_scores(oraciones, clean(r["Text"]), detector, det_tokenizer)
        flag = decide_flag(scores, threshold)
        sid = pick_error_sentence(scores, flag)
        if flag == 1:
            fuente = build_correction_source(dict(oraciones)[sid], r["Text"])
            correccion = generate_correction(fuente, corrector, cor_tokenizer, num_beams=num_beams,
                                             fallback=dict(oraciones)[sid])
        else:
            correccion = "NA"
        filas.append({"Text ID": r["Text ID"], "Error Flag": flag, "Error Sentence ID": sid,
                      "Corrected Sentence": correccion, "max_score": max(scores)})
    return pd.DataFrame(filas)


def write_predictions(pred_df, path):
    """Guarda las predicciones en CSV y en el formato de envío oficial (.txt):
    una línea por texto con `text_id flag sent_id corrección`."""
    pred_df.drop(columns=["max_score"], errors="ignore").to_csv(path, index=False)
    with open(re.sub(r"\.csv$", "", path) + ".txt", "w", encoding="utf-8") as f:
        for _, r in pred_df.iterrows():
            correccion = "NA" if r["Error Flag"] == 0 else (clean(r["Corrected Sentence"]) or "NA")
            f.write(f'{r["Text ID"]} {r["Error Flag"]} {r["Error Sentence ID"]} {correccion}\n')


def _scores_por_texto(df, detector, tok):
    return [predict_sentence_scores(parse_sentences(r["Sentences"]), clean(r["Text"]), detector, tok)
            for _, r in tqdm(df.iterrows(), total=len(df), desc="detector")]


def run_experiment(data_path, out_dir="results", cfg=None, quick=False, bertscore=True,
                   bert_model=BERTSCORE_MODEL, quick_groups=110):
    """Corre el experimento completo y guarda predicciones y métricas en out_dir.

    Orden: datos -> pisos -> detector (train) -> umbral (dev) -> corrector (train) -> evaluación (test).
    Test no se usa para ninguna decisión.
    """
    cfg = cfg or TrainConfig()
    if quick:
        cfg.det_epochs, cfg.cor_epochs = 1, 1
    set_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    res = {"config": cfg.to_dict(), "quick": quick}

    # 1. Datos
    df = load_data(data_path)
    df["group"] = make_case_groups(df)
    res["dataset"] = describe_dataset(df)
    if quick:
        df = sample_groups(df, quick_groups, seed=cfg.seed)
    train_df, dev_df, test_df = split_train_dev_test(df, seed=cfg.seed)
    for a, b in [(train_df, dev_df), (train_df, test_df), (dev_df, test_df)]:
        assert set(a["group"]).isdisjoint(set(b["group"])), "hay casos compartidos entre conjuntos"
    train_sent, dev_sent = build_sentence_table(train_df), build_sentence_table(dev_df)
    res["splits"] = {n: {"textos": len(d), "pct_error": round(100 * float(d["Error Flag"].mean()), 1)}
                     for n, d in [("train", train_df), ("dev", dev_df), ("test", test_df)]}
    print(res["splits"])

    # 2. Pisos en test
    res["pisos"] = {"siempre_error": baseline_always_error(test_df), "nunca_error": baseline_never_error(test_df),
                    "ultima_oracion": baseline_last_sentence(test_df),
                    "copiar_oracion": baseline_copy_sentence(test_df, bertscore=bertscore)}
    release_bert_scorer()
    print("Pisos:", res["pisos"])

    # 3. Detector y umbral
    det_trainer, det_tok = train_error_classifier(
        train_sent, dev_sent, model_name=cfg.detector_name, output_dir=os.path.join(out_dir, "_detector"),
        epochs=cfg.det_epochs, lr=cfg.det_lr, batch_size=cfg.det_batch_size,
        eval_batch_size=cfg.det_eval_batch_size, seed=cfg.seed)
    detector = det_trainer.model
    res["detector_log"] = [h for h in det_trainer.state.log_history if "eval_loss" in h]
    threshold = choose_threshold(_scores_por_texto(dev_df, detector, det_tok), dev_df["Error Flag"])
    res["umbral"] = threshold

    test_scores = _scores_por_texto(test_df, detector, det_tok)
    flags = [decide_flag(s, threshold) for s in test_scores]
    ids = [pick_error_sentence(s, f) for s, f in zip(test_scores, flags)]
    con_error = (test_df["Error Flag"] == 1).values
    res["detector_test"] = {
        "flag_acc": eval_flag_accuracy(test_df["Error Flag"], flags),
        "sentence_acc": eval_sentence_accuracy(test_df["Error Sentence ID"], ids),
        "localizacion_argmax_textos_con_error": float(np.mean(
            [int(np.argmax(s)) == t for s, t, c in zip(test_scores, test_df["Error Sentence ID"], con_error) if c])),
    }

    # 4. Corrector
    cor_trainer, cor_tok = train_corrector(
        build_correction_pairs(train_df), build_correction_pairs(dev_df), model_name=cfg.corrector_name,
        output_dir=os.path.join(out_dir, "_corrector"), epochs=cfg.cor_epochs, lr=cfg.cor_lr,
        batch_size=cfg.cor_batch_size, eval_batch_size=cfg.cor_eval_batch_size, seed=cfg.seed)
    corrector = cor_trainer.model
    res["corrector_log"] = [h for h in cor_trainer.state.log_history if "eval_loss" in h]

    ver = test_df[test_df["Error Flag"] == 1]
    preds = [generate_correction(build_correction_source(s, t), corrector, cor_tok, num_beams=cfg.num_beams, fallback=s)
             for s, t in zip(ver["Error Sentence"], ver["Text"])]
    c = eval_correction(preds, [clean(s) for s in ver["Corrected Sentence"]], bertscore=bertscore, bert_model=bert_model)
    c.pop("per_text")
    res["corrector_oracion_real"] = c

    # 5. Pipeline completo en test
    pred_test = run_pipeline(test_df, detector, det_tok, corrector, cor_tok, threshold, num_beams=cfg.num_beams)
    res["pipeline_test"] = evaluate_all(pred_test, test_df, bertscore=bertscore, bert_model=bert_model)
    write_predictions(pred_test, os.path.join(out_dir, "predicciones_test.csv"))
    res["minutos"] = round((time.time() - t0) / 60, 1)

    with open(os.path.join(out_dir, "metricas_test.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print("Resultado en test:", res["pipeline_test"])
    return res
