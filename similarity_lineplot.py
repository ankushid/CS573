import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


TRANSCRIPTS_CSV = "transcripts_clean.csv"
DOC_EMB_CSV = "document_embeddings.csv"


def parse_emb(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=float)


def compute_similarity_timeseries():
    meta = pd.read_csv(TRANSCRIPTS_CSV)
    docs = pd.read_csv(DOC_EMB_CSV)

    meta["source_file_clean"] = meta["source_file"].str.lower()
    docs["doc_id_clean"] = docs["doc_id"].str.lower()

    merged = meta.merge(
        docs,
        left_on="source_file_clean",
        right_on="doc_id_clean",
        how="inner"
    )
    merged = merged[merged["embedding"].notna()].copy()
    merged["vec"] = merged["embedding"].apply(parse_emb)

    rows = []

    for period, group in merged.groupby("period"):
        tickers = group["ticker_x"].unique()

        if not {"KO", "PEP"}.issubset(set(tickers)):
            continue

        def avg_vec(ticker):
            subset = group[group["ticker_x"] == ticker]["vec"].values
            return np.vstack(subset).mean(axis=0)

        ko_vec = avg_vec("KO")
        pep_vec = avg_vec("PEP")

        sim = cosine_similarity(ko_vec.reshape(1, -1), pep_vec.reshape(1, -1))[0, 0]

        rows.append({"period": period, "similarity": sim})

    df = pd.DataFrame(rows)

    df = df.sort_values("period")

    return df


def plot_similarity(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["period"], df["similarity"], marker="o", linewidth=2)

    plt.xticks(rotation=45)
    plt.ylabel("Cosine Similarity (KO vs PEP)")
    plt.title("KO–PEP Narrative Similarity Over Time")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = compute_similarity_timeseries()
    print(df)
    plot_similarity(df)