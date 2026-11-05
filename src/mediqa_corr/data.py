"""Caja 1 · Datos: carga del CSV, segmentación oficial, agrupación de casos gemelos y división train/dev/test."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from .config import DEV_SIZE, DUP_THRESHOLD, SEED, TEST_SIZE

COLUMNAS = ["Text ID", "Text", "Sentences", "Error Flag", "Error Sentence ID",
            "Error Sentence", "Corrected Sentence", "Corrected Text"]


def load_data(path):
    """Carga el CSV, descarta la columna sin nombre y deja los tipos listos.

    keep_default_na=False deja "NA" como texto, que es como lo exige la tarea para los textos sin error.
    """
    df = pd.read_csv(path, keep_default_na=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    faltan = [c for c in COLUMNAS if c not in df.columns]
    assert not faltan, f"Faltan columnas: {faltan}"
    df["Error Flag"] = df["Error Flag"].astype(int)
    df["Error Sentence ID"] = df["Error Sentence ID"].astype(int)
    return df


def clean(s):
    """Normaliza espacios y saltos de línea."""
    return " ".join(str(s).split())


def parse_sentences(sentences):
    """'0 Primera oración.\\n1 Segunda...' -> [(0, 'Primera oración.'), (1, 'Segunda...')].

    Una línea abre la oración k solo si empieza con '<k> ' (k = siguiente ID esperado).
    Cualquier otra línea es continuación de la oración anterior: así no se confunden
    líneas de tablas como '37 C' con un ID de oración.
    """
    out = []
    for line in sentences.split("\n"):
        prefijo = f"{len(out)} "
        if line.startswith(prefijo):
            out.append([len(out), line[len(prefijo):]])
        elif out:
            out[-1][1] += "\n" + line
        else:
            out.append([0, line])
    return [(i, t.strip()) for i, t in out]


def make_case_groups(df, threshold=DUP_THRESHOLD):
    """ID de grupo por texto: textos con coseno TF-IDF > threshold quedan en el mismo grupo.

    El dataset MS tiene casi todos los casos en dos versiones (correcta y con error).
    Agruparlos evita que una versión quede en train y la otra en evaluación.
    """
    X = TfidfVectorizer().fit_transform(df["Text"])
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    _, grupos = connected_components(sim > threshold, directed=False)
    return grupos


def describe_dataset(df, verbose=True):
    """Resumen del dataset: balance, largo, posición del error y grupos."""
    n_sent = df["Sentences"].apply(lambda s: len(parse_sentences(s)))
    err = df[df["Error Flag"] == 1]
    rel = err["Error Sentence ID"] / (n_sent[err.index] - 1)
    stats = {
        "textos": len(df),
        "con_error": int(df["Error Flag"].sum()),
        "sin_error": int((df["Error Flag"] == 0).sum()),
        "oraciones": int(n_sent.sum()),
        "oraciones_por_texto_media": round(float(n_sent.mean()), 1),
        "oraciones_por_texto_min_max": (int(n_sent.min()), int(n_sent.max())),
        "pct_oraciones_erroneas": round(100 * len(err) / n_sent.sum(), 2),
        "error_en_primera_pct": round(100 * float((err["Error Sentence ID"] == 0).mean()), 1),
        "error_en_ultima_pct": round(100 * float((err["Error Sentence ID"] == n_sent[err.index] - 1).mean()), 1),
        "error_en_ultimo_tercio_pct": round(100 * float((rel >= 2 / 3).mean()), 1),
        "palabras_por_texto_media": round(float(df["Text"].str.split().str.len().mean()), 1),
        "palabras_correccion_media": round(float(err["Corrected Sentence"].str.split().str.len().mean()), 1),
    }
    if "group" in df.columns:
        tam = df["group"].value_counts()
        stats["grupos"] = int(len(tam))
        stats["textos_con_gemelo_pct"] = round(100 * float((df["group"].map(tam) > 1).mean()), 1)
    if verbose:
        for k, v in stats.items():
            print(f"{k:34s} {v}")
    return stats


def build_sentence_table(df):
    """Una fila por oración: text_id, sent_id, sentence, context, n_sentences, rel_position, is_error."""
    filas = []
    for _, r in df.iterrows():
        oraciones = parse_sentences(r["Sentences"])
        n = len(oraciones)
        contexto = clean(r["Text"])
        for sid, texto in oraciones:
            filas.append({
                "text_id": r["Text ID"], "sent_id": sid, "sentence": clean(texto), "context": contexto,
                "n_sentences": n, "rel_position": sid / (n - 1) if n > 1 else 0.0,
                "is_error": int(r["Error Flag"] == 1 and sid == r["Error Sentence ID"]),
            })
    return pd.DataFrame(filas)


def split_by_group(df, size, seed=SEED):
    """Separa una fracción ~size de los GRUPOS, estratificando por 'Error Flag'. Devuelve (resto, parte)."""
    sgkf = StratifiedGroupKFold(n_splits=round(1 / size), shuffle=True, random_state=seed)
    idx_resto, idx_parte = next(sgkf.split(df, df["Error Flag"], groups=df["group"]))
    return df.iloc[idx_resto].reset_index(drop=True), df.iloc[idx_parte].reset_index(drop=True)


def split_train_dev_test(df, dev_size=DEV_SIZE, test_size=TEST_SIZE, seed=SEED):
    """train / dev / test sin grupos compartidos. df debe tener la columna 'group'."""
    resto, test = split_by_group(df, test_size, seed)
    train, dev = split_by_group(resto, dev_size / (1 - test_size), seed)
    return train, dev, test


def sample_groups(df, n_groups, seed=SEED):
    """Submuestra de grupos completos (para pruebas rápidas sin romper los pares)."""
    g = df["group"].drop_duplicates().sample(n_groups, random_state=seed)
    return df[df["group"].isin(g)].reset_index(drop=True)
