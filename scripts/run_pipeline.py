"""Corre el experimento completo de MEDIQA-CORR (MS) y guarda predicciones y métricas.

Uso:
    python scripts/run_pipeline.py --data data/MEDIQA-CORR-2024-MS-TrainingData.csv --out results
    python scripts/run_pipeline.py --data ... --quick          # ~200 textos y 1 época, para probar
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mediqa_corr.config import BERTSCORE_MODEL, TrainConfig  # noqa: E402
from mediqa_corr.pipeline import run_experiment  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CSV MEDIQA-CORR-2024-MS-TrainingData.csv")
    p.add_argument("--out", default="results", help="carpeta de salida")
    p.add_argument("--quick", action="store_true", help="~200 textos y 1 época por modelo")
    p.add_argument("--no-bertscore", action="store_true", help="omite BERTScore (no descarga deberta)")
    p.add_argument("--bertscore-model", default=BERTSCORE_MODEL)
    p.add_argument("--detector", default=TrainConfig.detector_name)
    p.add_argument("--corrector", default=TrainConfig.corrector_name)
    p.add_argument("--det-epochs", type=int, default=TrainConfig.det_epochs)
    p.add_argument("--cor-epochs", type=int, default=TrainConfig.cor_epochs)
    p.add_argument("--det-batch", type=int, default=TrainConfig.det_batch_size)
    p.add_argument("--cor-batch", type=int, default=TrainConfig.cor_batch_size)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    a = p.parse_args()

    cfg = TrainConfig(detector_name=a.detector, corrector_name=a.corrector, det_epochs=a.det_epochs,
                      cor_epochs=a.cor_epochs, det_batch_size=a.det_batch, cor_batch_size=a.cor_batch, seed=a.seed)
    run_experiment(a.data, out_dir=a.out, cfg=cfg, quick=a.quick, bertscore=not a.no_bertscore,
                   bert_model=a.bertscore_model)


if __name__ == "__main__":
    main()
