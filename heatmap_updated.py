import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

TRANSCRIPTS_CSV = "transcripts_clean.csv"
DOC_EMB_CSV = "document_embeddings.csv"

TARGET_PERIOD = "2020Q2"


def parse_emb(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=float)


def build_firm_vectors_for_period(period: str):
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

    sub = merged[(merged["period"] == period) & merged["embedding"].notna()].copy()
    if sub.empty:
        raise ValueError(f"No documents found for period {period!r}")

    sub["vec"] = sub["embedding"].apply(parse_emb)

    firm_vectors = {}
    for ticker, group in sub.groupby("ticker_x"):
        mat = np.vstack(group["vec"].values)
        firm_vectors[ticker] = mat.mean(axis=0)

    firms = sorted(firm_vectors.keys())
    vecs = np.vstack([firm_vectors[f] for f in firms])

    return firms, vecs


def plot_firm_similarity_heatmap(period: str):
    firms, vecs = build_firm_vectors_for_period(period)

    sim_mat = cosine_similarity(vecs, vecs)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="viridis")

    ax.set_xticks(range(len(firms)))
    ax.set_yticks(range(len(firms)))
    ax.set_xticklabels(firms)
    ax.set_yticklabels(firms)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(firms)):
        for j in range(len(firms)):
            ax.text(
                j, i, f"{sim_mat[i, j]:.2f}",
                ha="center", va="center", color="white" if sim_mat[i,j] < 0.5 else "black"
            )

    ax.set_title(f"Firm-Level Narrative Similarity — {period}")
    fig.colorbar(im, ax=ax, label="Cosine similarity")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_firm_similarity_heatmap(TARGET_PERIOD)