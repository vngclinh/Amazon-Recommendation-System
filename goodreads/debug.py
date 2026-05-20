import pandas as pd
import os
from pathlib import Path

CACHE_DIR  = r"D:\goodreads\raw_cache"
HF_CACHE   = r"C:\Users\OS\.cache\huggingface\hub\datasets--vngclinh--goodreads-concats\snapshots\3d66b4222d0fc95e5e1a8d8216c736599e5b01c4\data"
os.makedirs(CACHE_DIR, exist_ok=True)

for genre_dir in sorted(Path(HF_CACHE).iterdir()):
    if not genre_dir.is_dir():
        continue
    # folder name = "primary_genre=children" → extract genre
    genre = genre_dir.name.replace("primary_genre=", "")
    safe  = genre.replace("/", "_").replace(",", "_").replace(" ", "-")
    out   = f"{CACHE_DIR}/{safe}.parquet"

    if os.path.exists(out):
        print(f"Skip: {genre}")
        continue

    parquet_files = list(genre_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet in {genre_dir}, skip")
        continue

    print(f"Reading: {genre} ({len(parquet_files)} files)...")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df["primary_genre"] = genre  # thêm lại column bị mất khi partition
    df.to_parquet(out, index=False)
    print(f"  ✓ {len(df):,} rows → {out}")

print("Done.")