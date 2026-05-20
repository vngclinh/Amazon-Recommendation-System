import pandas as pd
import numpy as np
import re, html, os
from pathlib import Path

CACHE_DIR  = r"D:\goodreads\raw_cache"
OUTPUT_DIR = r"D:\goodreads\preprocessed"
LDA_DIR    = r"D:\goodreads\lda_corpus"
for d in [OUTPUT_DIR, LDA_DIR]:
    os.makedirs(d, exist_ok=True)

cache_files = sorted(Path(CACHE_DIR).glob("*.parquet"))

# ── Pass 1: global user stats (chỉ load 3 cols nhẹ) ──────────────────────────
print("Pass 1: computing global user stats...")
chunks = []
for f in cache_files:
    chunks.append(pd.read_parquet(f, columns=["user_id", "review_id", "primary_genre", "date_added"]))

light = pd.concat(chunks, ignore_index=True)
light["date_added"] = pd.to_datetime(light["date_added"], errors="coerce", utc=True, format="mixed")

user_review_count = light.groupby("user_id")["review_id"].count()
user_genre_count  = light.groupby("user_id")["primary_genre"].nunique()
bridge_users      = set(user_genre_count[user_genre_count >= 3].index)

train_mask    = light["date_added"].dt.year < 2017
train_counts  = light[train_mask].groupby("user_id").size()
valid_users   = set(train_counts[train_counts >= 5].index)

print(f"Total users: {user_review_count.shape[0]:,}")
print(f"Bridge users: {len(bridge_users):,}")
print(f"Valid users (cold-start): {len(valid_users):,}")
del light, chunks

# ── Pass 2: process từng genre file ──────────────────────────────────────────
def clean(t):
    t = html.unescape(str(t))
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

lda_parts = []

for f in cache_files:
    genre = f.stem
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{genre}.parquet")):
        print(f"Skip (exists): {genre}")
        continue
    print(f"\nProcessing: {genre}")
    df = pd.read_parquet(f)

    # Phase 1
    for col in ["rating", "n_votes", "n_comments", "publication_year", "num_pages"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_votes"]    = df["n_votes"].fillna(0).astype(int)
    df["n_comments"] = df["n_comments"].fillna(0).astype(int)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce", utc=True, format="mixed")
    df = df[df["rating"] > 0].dropna(subset=["review_text", "date_added"])

    # Phase 2
    df["review_text"]        = df["review_text"].map(clean)
    df["review_token_count"] = df["review_text"].str.split().str.len()
    df = df[df["review_token_count"] >= 10]

    # Phase 3
    cap = df["n_votes"].quantile(0.99) or 1
    df["vote_weight"] = np.log1p(df["n_votes"].clip(upper=cap))
    df["vote_weight"] = df["vote_weight"].replace([np.inf, -np.inf], 0).fillna(0)  # ← thêm dòng này
    df["edge_weight"] = df["vote_weight"] * (df["rating"] / 5.0)

    # Phase 4 — dùng global stats
    df["user_review_count"] = df["user_id"].map(user_review_count).fillna(0).astype(int)
    df["user_genre_count"]  = df["user_id"].map(user_genre_count).fillna(0).astype(int)
    df["is_bridge_user"]    = df["user_id"].isin(bridge_users)

    # Phase 5
    df = df[df["user_id"].isin(valid_users)]
    df = df.sort_values("date_added")
    df["split"] = "train"
    df.loc[df["date_added"].dt.year == 2017, "split"] = "val"
    df.loc[df["date_added"].dt.year >= 2018, "split"] = "test"

    print(f"  rows: {len(df):,} | splits: {df['split'].value_counts().to_dict()}")

    # Save
    out = os.path.join(OUTPUT_DIR, f"{genre}.parquet")
    df.to_parquet(out, index=False)
    print(f"  Saved → {out}")

    # Phase 6: LDA sample
    train_df = df[df["split"] == "train"]
    if len(train_df) > 0:
        lda_parts.append(
            train_df.sample(
                min(len(train_df), 200_000),
                weights=train_df["vote_weight"] + 1e-6,
                random_state=42
            )[["review_id", "user_id", "book_id", "primary_genre",
               "rating", "review_text", "vote_weight", "edge_weight"]]
        )

# ── Save LDA corpus ───────────────────────────────────────────────────────────
lda_df = pd.concat(lda_parts, ignore_index=True)
lda_df.to_parquet(os.path.join(LDA_DIR, "lda_corpus.parquet"), index=False)
print(f"\nLDA corpus: {len(lda_df):,} rows → {LDA_DIR}/lda_corpus.parquet")
print("\nDone.")