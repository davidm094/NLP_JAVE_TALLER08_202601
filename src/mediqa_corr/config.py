"""Configuración del pipeline: modelos, largos máximos, tamaños de los conjuntos e hiperparámetros.

Los valores por defecto son los de la corrida reportada (RTX 3090, test de 312 notas).
"""
from dataclasses import dataclass, asdict

SEED = 42

DETECTOR_NAME = "emilyalsentzer/Bio_ClinicalBERT"
CORRECTOR_NAME = "google/flan-t5-base"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"   # el mismo del script oficial de MEDIQA-CORR

DUP_THRESHOLD = 0.7          # coseno TF-IDF a partir del cual dos textos son versiones del mismo caso
DEV_SIZE = 0.15
TEST_SIZE = 0.15

MAX_LEN_DETECTOR = 384       # tokens de (oración + nota)
MAX_LEN_SOURCE = 384         # tokens de entrada del corrector
MAX_LEN_TARGET = 64          # tokens de la oración corregida


@dataclass
class TrainConfig:
    """Hiperparámetros de entrenamiento de los dos modelos."""
    detector_name: str = DETECTOR_NAME
    corrector_name: str = CORRECTOR_NAME
    det_epochs: int = 3
    det_lr: float = 2e-5
    det_batch_size: int = 16
    det_eval_batch_size: int = 64
    cor_epochs: int = 6
    cor_lr: float = 1e-4
    cor_batch_size: int = 8
    cor_eval_batch_size: int = 16
    num_beams: int = 4
    seed: int = SEED

    def to_dict(self):
        return asdict(self)
