# Taller 8 · MEDIQA-CORR 2024: implementación del pipeline

**Grupo:** David Moreno y Dierick Brochero · Procesamiento de Lenguaje Natural, Pontificia Universidad Javeriana

Implementación del pipeline diseñado en el Taller 7 para detectar y corregir errores médicos en notas clínicas (dataset MS de MEDIQA-CORR 2024). Resuelve las tres subtareas:

- **A**: ¿la nota tiene un error? (0 / 1)
- **B**: ¿en qué oración? (ID de la oración, o −1)
- **C**: ¿cómo queda la oración corregida? (texto, o NA)

## Pipeline

```
CSV ─► Datos ─► Detector (Bio_ClinicalBERT) ─► ¿max P ≥ umbral? ─► sí: Corrector (Flan-T5) ─► C
       (oraciones,      P(error) por oración          │                                          │
        gemelos,                                      └─ no: A=0, B=−1, C=NA ───────────────────┤
        train/dev/test)                                                                          ▼
                                                                          Evaluación + archivo de envío
```

| Caja | Módulo | Modelo / herramienta | Funciones principales |
|---|---|---|---|
| 1. Datos | [`src/mediqa_corr/data.py`](src/mediqa_corr/data.py) | pandas, TF-IDF, scikit-learn | `load_data`, `parse_sentences`, `make_case_groups`, `split_train_dev_test`, `build_sentence_table` |
| 2. Detector (A, B) | [`src/mediqa_corr/detector.py`](src/mediqa_corr/detector.py) | `emilyalsentzer/Bio_ClinicalBERT` | `train_error_classifier`, `predict_sentence_scores`, `choose_threshold`, `decide_flag`, `pick_error_sentence` |
| 3. Corrector (C) | [`src/mediqa_corr/corrector.py`](src/mediqa_corr/corrector.py) | `google/flan-t5-base` | `build_correction_pairs`, `train_corrector`, `generate_correction` |
| 4. Evaluación | [`src/mediqa_corr/evaluation.py`](src/mediqa_corr/evaluation.py) | `rouge`, `bert_score` (`deberta-xlarge-mnli`) | `eval_flag_accuracy`, `eval_sentence_accuracy`, `eval_correction`, `composite_score`, `evaluate_all` |
| Pisos | [`src/mediqa_corr/baselines.py`](src/mediqa_corr/baselines.py) | — | `baseline_always_error`, `baseline_never_error`, `baseline_last_sentence`, `baseline_copy_sentence` |
| Orquestación | [`src/mediqa_corr/pipeline.py`](src/mediqa_corr/pipeline.py) | — | `run_pipeline`, `write_predictions`, `run_experiment` |

Decisiones clave:

- **Cada oración se clasifica junto con la nota completa** (`[CLS] oración [SEP] nota [SEP]`, 384 tokens): el error solo se ve en contexto.
- **Pesos de clase** (≈ 0,52 / 11,3): solo el 4,4 % de las oraciones son erróneas.
- **División por caso, no por texto**: el MS tiene casi todos los casos en dos versiones (correcta y con error). Se agrupan por similitud TF-IDF (> 0,7) y se parte con `StratifiedGroupKFold`.
- **Umbral elegido en dev, resultados en test** (test no se usa para ninguna decisión).
- **Métricas como el script oficial**: paquete `rouge`, BERTScore reescalado y recortado a [0, 1]. Sin BLEURT.

## Cómo correrlo

```bash
pip install -r requirements.txt          # en GPU, instala antes torch con CUDA
# copia el CSV en data/ (ver data/README.md)

pytest -q                                 # pruebas rápidas, sin GPU
python scripts/run_pipeline.py --data data/MEDIQA-CORR-2024-MS-TrainingData.csv --quick   # ~200 textos, 1 época
python scripts/run_pipeline.py --data data/MEDIQA-CORR-2024-MS-TrainingData.csv --out results  # corrida completa
python scripts/leak_check.py  --data data/MEDIQA-CORR-2024-MS-TrainingData.csv             # experimento de fuga
```

O abre [`notebooks/taller8_mediqa_corr.ipynb`](notebooks/taller8_mediqa_corr.ipynb) en Colab con GPU: llama las mismas funciones paso a paso.

La corrida completa en una RTX 3090 tarda unos 14 minutos. `run_pipeline.py` guarda `predicciones_test.csv`, `predicciones_test.txt` (formato de envío oficial) y `metricas_test.json`.

## Resultados (test, 312 notas)

Corrida completa en una RTX 3090: [`notebooks/ejecucion_rtx3090.ipynb`](notebooks/ejecucion_rtx3090.ipynb), métricas en [`results/metricas_test.json`](results/metricas_test.json).

| Subtarea | Modelo | Piso (sin modelo) |
|---|---|---|
| A · exactitud | 54,2 % | 55,8 % (siempre "error") |
| B · exactitud | **53,5 %** | 44,2 % (nunca "error") |
| C con la oración real · ROUGE-1 / BERTScore | 0,567 / 0,639 | 0,600 / 0,673 (copiar la oración) |
| C en cadena · ROUGE-1 / BERTScore (149 pares) | 0,562 / 0,638 | — |
| Composite (sin BLEURT) | 0,351 | 0,442 (nunca "error") |

**Lectura:** el detector localiza la oración errónea en 147 de 149 notas con error que marca (98,7 %), pero también marca error en 118 de 138 notas correctas. Aprendió *dónde* está la respuesta del caso, no *si* es correcta: le falta conocimiento médico. El corrector cambia la entidad correcta, pero por respuestas frecuentes del entrenamiento (*S. pneumoniae*, metronidazol, ceftriaxona). Es coherente con MEDEC, donde Phi-3 (7B) apenas supera el piso de A, y con WangLab, que necesitó recuperar la pregunta de MedQA y GPT-4.

### Fuga de datos medida

[`scripts/leak_check.py`](scripts/leak_check.py) corre un "buscador" sin modelo que copia la respuesta del texto más parecido de train:

| División | A | B | Correcciones exactas |
|---|---|---|---|
| Al azar por Text ID | 91,8 % | 84,2 % | 25,8 % |
| Por grupo de caso (la usada) | 55,7 % | 34,9 % | 0 % |

Con una división al azar cualquier métrica queda inflada; por eso se parte por grupo.

## Problemas resueltos respecto a la versión inicial

1. Fuga entre casos gemelos → división por grupo.
2. `OverflowError` de BERTScore con DeBERTa en transformers 5 → largo máximo acotado a 512.
3. Correcciones vacías tumbaban `bert_score` y `rouge` → puntaje 0 y respaldo con la oración original.
4. Umbral elegido y reportado en el mismo conjunto → train / dev / test.
5. Métrica de C distinta a la oficial → `rouge` + BERTScore reescalado.
6. Bio_ClinicalBERT tokenizado en minúsculas siendo cased → `do_lower_case=False`.
7. Piso de B mal elegido → `baseline_never_error`.
8. Sin archivo de envío → `.txt` con `text_id flag sent_id corrección`.

## Estructura

```
src/mediqa_corr/   código, un módulo por caja
scripts/           run_pipeline.py (experimento completo), leak_check.py (fuga)
notebooks/         taller8_mediqa_corr.ipynb (paso a paso), ejecucion_rtx3090.ipynb (corrida con salidas)
results/           predicciones y métricas de la corrida completa
tests/             pruebas sin GPU (pytest)
data/              el CSV va aquí (no se versiona)
```

## Referencias

- Ben Abacha et al. (2024). *MEDEC: A Benchmark for Medical Error Detection and Correction in Clinical Notes.* [arXiv:2412.19260](https://arxiv.org/abs/2412.19260)
- Toma et al. (2024). *WangLab at MEDIQA-CORR 2024: Optimized LLM-based Programs for Medical Error Detection and Correction.* [ClinicalNLP 2024](https://aclanthology.org/2024.clinicalnlp-1.59/)
- [Script oficial de evaluación MEDIQA-CORR 2024](https://github.com/abachaa/MEDIQA-CORR-2024/tree/main/evaluation)
