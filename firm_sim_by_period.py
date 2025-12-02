import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

TRANSCRIPTS_CSV = "transcripts_clean.csv"
DOC_EMB_CSV = "document_embeddings.csv"
OUT_CSV = "ko_pep_similarity_by_period.csv"


def parse_emb(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=float)


def main():
    meta = pd.read_csv(TRANSCRIPTS_CSV)
    docs = pd.read_csv(DOC_EMB_CSV)

    meta["source_file_clean"] = meta["source_file"].str.lower()
    docs["doc_id_clean"] = docs["doc_id"].str.lower()

    merged = meta.merge(
        docs, left_on="source_file_clean", right_on="doc_id_clean", how="inner"
    )

    merged = merged[merged["embedding"].notna()].copy()

    merged["vec"] = merged["embedding"].apply(parse_emb)

    rows = []
    for period, sub in merged.groupby("period"):
        tickers = set(sub["ticker_x"].unique())
        # Only periods where we have both KO and PEP
        if not {"KO", "PEP"}.issubset(tickers):
            continue

        firm_vecs = {}
        for t in ["KO", "PEP"]:
            vecs = np.vstack(sub[sub["ticker_x"] == t]["vec"].values)
            firm_vecs[t] = vecs.mean(axis=0)

        sim = cosine_similarity(
            firm_vecs["KO"].reshape(1, -1),
            firm_vecs["PEP"].reshape(1, -1),
        )[0, 0]

        rows.append(
            {
                "period": period,
                "ticker1": "KO",
                "ticker2": "PEP",
                "cosine_similarity": sim,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("period")
    out_df.to_csv(OUT_CSV, index=False)

    print("Done. Saved firm-level similarity to", OUT_CSV)
    print(out_df)


if __name__ == "__main__":
    main()