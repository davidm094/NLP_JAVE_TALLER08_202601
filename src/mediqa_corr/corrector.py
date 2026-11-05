"""Caja 3 · Corrector (subtarea C): Flan-T5 afinado para reescribir la oración errónea."""
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

from .config import CORRECTOR_NAME, MAX_LEN_SOURCE, MAX_LEN_TARGET, SEED
from .data import clean


def build_correction_source(sentence, context):
    """Entrada del corrector: instrucción + oración errónea + nota completa. Igual al entrenar y al predecir."""
    return f"Correct the medical error in this sentence. Sentence: {clean(sentence)} Context: {clean(context)}"


def build_correction_pairs(df):
    """Tabla (text_id, source, target) con los textos que tienen error."""
    con_error = df[df["Error Flag"] == 1]
    return pd.DataFrame({
        "text_id": con_error["Text ID"].values,
        "source": [build_correction_source(s, t) for s, t in zip(con_error["Error Sentence"], con_error["Text"])],
        "target": [clean(c) for c in con_error["Corrected Sentence"]],
    })


def tokenize_for_seq2seq(batch, tokenizer, max_input=MAX_LEN_SOURCE, max_target=MAX_LEN_TARGET):
    """Tokeniza source y target; los ids del target son los 'labels'."""
    enc = tokenizer(batch["source"], max_length=max_input, truncation=True)
    enc["labels"] = tokenizer(text_target=batch["target"], max_length=max_target, truncation=True)["input_ids"]
    return enc


def train_corrector(train_pairs, dev_pairs, model_name=CORRECTOR_NAME, output_dir="outputs/corrector",
                    epochs=6, lr=1e-4, batch_size=8, eval_batch_size=16, seed=SEED):
    """Afina el corrector. Sin fp16: T5 desborda en media precisión. Devuelve (trainer, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preparar(pairs):
        ds = Dataset.from_pandas(pairs[["source", "target"]], preserve_index=False)
        return ds.map(lambda b: tokenize_for_seq2seq(b, tokenizer), batched=True, remove_columns=["source", "target"])

    args = Seq2SeqTrainingArguments(output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
                                    per_device_train_batch_size=batch_size, per_device_eval_batch_size=eval_batch_size,
                                    eval_strategy="epoch", save_strategy="no", logging_steps=50,
                                    report_to="none", seed=seed)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=preparar(train_pairs),
                             eval_dataset=preparar(dev_pairs), processing_class=tokenizer,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100))
    trainer.train()
    return trainer, tokenizer


def generate_correction(source, model, tokenizer, num_beams=4, max_new_tokens=MAX_LEN_TARGET, fallback=None):
    """Genera la oración corregida con beam search.

    fallback: texto a devolver si la generación sale vacía (normalmente, la oración errónea).
    """
    enc = tokenizer(source, return_tensors="pt", truncation=True, max_length=MAX_LEN_SOURCE).to(model.device)
    model.eval()
    with torch.no_grad():
        ids = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens)
    salida = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return salida if salida else clean(fallback or "")
