"""Pipeline para MEDIQA-CORR 2024 (dataset MS): detección y corrección de errores médicos.

Módulos, uno por caja del pipeline:
    data        Caja 1 · carga, segmentación oficial, casos gemelos, división train/dev/test
    detector    Caja 2 · Bio_ClinicalBERT, subtareas A y B
    corrector   Caja 3 · Flan-T5, subtarea C
    evaluation  Caja 4 · métricas como el script oficial
    baselines   pisos y prompt P#1 de MEDEC
    pipeline    inferencia de punta a punta y orquestación del experimento
"""
__version__ = "1.0.0"
