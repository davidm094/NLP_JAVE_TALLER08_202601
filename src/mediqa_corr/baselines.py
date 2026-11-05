"""Líneas base (pisos) de A, B y C, y el prompt P#1 de MEDEC para comparar con un LLM."""
import re

from .data import clean, parse_sentences
from .evaluation import eval_correction, eval_flag_accuracy, eval_sentence_accuracy


def baseline_always_error(df):
    """Predice siempre flag = 1. Piso de A."""
    return {"flag_acc": eval_flag_accuracy(df["Error Flag"], [1] * len(df))}


def baseline_never_error(df):
    """Predice siempre flag = 0 y oración -1. Piso de B y del Composite."""
    return {"flag_acc": eval_flag_accuracy(df["Error Flag"], [0] * len(df)),
            "sentence_acc": eval_sentence_accuracy(df["Error Sentence ID"], [-1] * len(df)),
            "composite": float((df["Error Flag"] == 0).mean())}


def baseline_last_sentence(df):
    """Predice siempre que hay error y que la errónea es la última oración."""
    n = df["Sentences"].apply(lambda s: len(parse_sentences(s)))
    pred = n - 1
    con_error = df["Error Flag"] == 1
    return {"sentence_acc_todos_los_textos": eval_sentence_accuracy(df["Error Sentence ID"], pred),
            "sentence_acc_solo_textos_con_error": eval_sentence_accuracy(df.loc[con_error, "Error Sentence ID"],
                                                                         pred[con_error])}


def baseline_copy_sentence(df, bertscore=True):
    """Piso de C: devolver la oración errónea sin cambios."""
    con_error = df[df["Error Flag"] == 1]
    preds = [clean(s) for s in con_error["Error Sentence"]]
    refs = [clean(s) for s in con_error["Corrected Sentence"]]
    r = eval_correction(preds, refs, bertscore=bertscore)
    r.pop("per_text")
    return r


P1_INSTRUCCION = (
    "The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. "
    "The text is either correct or contains one error. The text has one sentence per line. Each line starts with the "
    "sentence ID, followed by a pipe character then the sentence to check. Check every sentence of the text. "
    "If the text is correct return the following output: CORRECT. If the text has a medical error related to treatment, "
    "management, cause, or diagnosis, return the sentence id of the sentence containing the error, followed by a space, "
    "and then a corrected version of the sentence. Finding and correcting the error requires medical knowledge and reasoning.")


def build_prompt_p1(sentences):
    """Prompt P#1 de MEDEC: una oración por línea con formato 'ID | oración'."""
    return P1_INSTRUCCION + "\n\n" + "\n".join(f"{i} | {clean(t)}" for i, t in sentences)


def parse_llm_output(output):
    """'CORRECT' -> (0, -1, 'NA'); '6 Corrected sentence' -> (1, 6, 'Corrected sentence')."""
    salida = output.strip()
    if salida.upper().startswith("CORRECT"):
        return 0, -1, "NA"
    m = re.match(r"^(\d+)\s+(.*)$", salida, flags=re.S)
    return (1, int(m.group(1)), m.group(2).strip()) if m else (0, -1, "NA")
