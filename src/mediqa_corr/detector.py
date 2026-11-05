"""Caja 2 · Detector (subtareas A y B): Bio_ClinicalBERT afinado para clasificar oraciones.

Cada oración se clasifica junto con la nota completa. A nivel de texto: flag = 1 si la oración
más sospechosa supera el umbral, y B es esa oración.
"""
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from .config import DETECTOR_NAME, MAX_LEN_DETECTOR, SEED
from .data import clean
from .evaluation import eval_flag_accuracy

THRESHOLD_GRID = list(np.arange(0.05, 0.96, 0.05)) + [0.97, 0.98, 0.99]


def load_detector_tokenizer(model_name=DETECTOR_NAME):
    """Tokenizador respetando mayúsculas.

    Bio_ClinicalBERT no trae tokenizer_config.json y Hugging Face asume do_lower_case=True,
    pero el modelo se preentrenó cased (--do_lower_case=False en el repositorio de los autores).
    """
    return AutoTokenizer.from_pretrained(model_name, do_lower_case=False)


def tokenize_sentence_with_context(batch, tokenizer, max_length=MAX_LEN_DETECTOR):
    """Tokeniza el par (oración, nota): [CLS] oración [SEP] nota [SEP]. Trunca solo la nota."""
    return tokenizer(batch["sentence"], batch["context"], truncation="only_second", max_length=max_length)


def compute_class_weights(labels):
    """Peso de cada clase = n / (2 * conteo de la clase). La clase minoritaria (error) pesa más."""
    labels = np.array(labels)
    pesos = [len(labels) / (2 * max((labels == c).sum(), 1)) for c in (0, 1)]
    return torch.tensor(pesos, dtype=torch.float)


class WeightedTrainer(Trainer):
    """Trainer de Hugging Face con entropía cruzada ponderada por clase."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _sentence_metrics(eval_pred):
    logits, labels = eval_pred
    p, r, f1, _ = precision_recall_fscore_support(labels, logits.argmax(-1), average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}


def train_error_classifier(train_sent_df, dev_sent_df, model_name=DETECTOR_NAME, output_dir="outputs/detector",
                           epochs=3, lr=2e-5, batch_size=16, eval_batch_size=64, seed=SEED):
    """Afina el detector sobre la tabla de oraciones. Devuelve (trainer, tokenizer)."""
    tokenizer = load_detector_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def preparar(sdf):
        ds = Dataset.from_pandas(sdf[["sentence", "context", "is_error"]].rename(columns={"is_error": "label"}),
                                 preserve_index=False)
        return ds.map(lambda b: tokenize_sentence_with_context(b, tokenizer), batched=True,
                      remove_columns=["sentence", "context"])

    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
                             per_device_train_batch_size=batch_size, per_device_eval_batch_size=eval_batch_size,
                             eval_strategy="epoch", save_strategy="no", logging_steps=100,
                             fp16=torch.cuda.is_available(), report_to="none", seed=seed)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=preparar(train_sent_df),
                              eval_dataset=preparar(dev_sent_df), processing_class=tokenizer,
                              data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=_sentence_metrics,
                              class_weights=compute_class_weights(train_sent_df["is_error"]))
    trainer.train()
    return trainer, tokenizer


def predict_sentence_scores(sentences, context, model, tokenizer, max_length=MAX_LEN_DETECTOR):
    """Probabilidad de error de cada oración de un texto. sentences = [(id, oración), ...]."""
    textos = [clean(t) for _, t in sentences]
    enc = tokenizer(textos, [context] * len(textos), truncation="only_second", max_length=max_length,
                    padding=True, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1)[:, 1].tolist()


def decide_flag(scores, threshold):
    """Flag del texto = 1 si la oración más sospechosa supera el umbral."""
    return int(max(scores) >= threshold)


def pick_error_sentence(scores, flag):
    """ID de la oración más sospechosa si hay error, y -1 si no."""
    return int(np.argmax(scores)) if flag == 1 else -1


def choose_threshold(scores_per_text, flags_true, grid=THRESHOLD_GRID):
    """Umbral de la malla que maximiza la exactitud del flag (se usa en dev, nunca en test).

    Con pesos de clase las probabilidades quedan corridas hacia 'error', por eso la malla llega a 0,99.
    """
    mejor_umbral, mejor_acc = 0.5, -1.0
    for u in grid:
        acc = eval_flag_accuracy(flags_true, [decide_flag(s, u) for s in scores_per_text])
        if acc > mejor_acc:
            mejor_umbral, mejor_acc = float(u), acc
    return mejor_umbral
