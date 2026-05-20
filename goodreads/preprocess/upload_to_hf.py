"""
Convert goodreads_reviews_dedup.json → partitioned Parquet files
and upload to HuggingFace dataset: vngclinh/goodreads-reviews
"""
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "vngclinh/goodreads-reviews"
SRC = r"D:\goodreads\goodreads_reviews_dedup.json"
OUT_DIR = r"D:\goodreads\parquet_chunks"
CHUNK_SIZE = 500_000

# Schema — keep dates as strings (many records have empty/null dates)
SCHEMA = pa.schema([
    ("user_id",      pa.string()),
    ("book_id",      pa.string()),
    ("review_id",    pa.string()),
    ("rating",       pa.int8()),
    ("review_text",  pa.string()),
    ("date_added",   pa.string()),
    ("date_updated", pa.string()),
    ("read_at",      pa.string()),
    ("started_at",   pa.string()),
    ("n_votes",      pa.int32()),
    ("n_comments",   pa.int32()),
])

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: create HF dataset repo ──────────────────────────────────────────
api = HfApi(token=HF_TOKEN)
print("Creating/verifying HuggingFace dataset repo …")
create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    exist_ok=True,
    private=False,
)
print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")

# ── Step 2: convert JSONL → Parquet chunks & upload ─────────────────────────
def rows_to_table(rows):
    cols = {field.name: [] for field in SCHEMA}
    for r in rows:
        cols["user_id"].append(r.get("user_id") or "")
        cols["book_id"].append(r.get("book_id") or "")
        cols["review_id"].append(r.get("review_id") or "")
        cols["rating"].append(int(r.get("rating") or 0))
        cols["review_text"].append(r.get("review_text") or "")
        cols["date_added"].append(r.get("date_added") or "")
        cols["date_updated"].append(r.get("date_updated") or "")
        cols["read_at"].append(r.get("read_at") or "")
        cols["started_at"].append(r.get("started_at") or "")
        cols["n_votes"].append(int(r.get("n_votes") or 0))
        cols["n_comments"].append(int(r.get("n_comments") or 0))
    return pa.table(cols, schema=SCHEMA)


part = 0
buf = []
total_written = 0

print(f"\nReading {SRC} …")
with open(SRC, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            buf.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  ⚠  bad JSON at line {lineno}, skipping")
            continue

        if len(buf) >= CHUNK_SIZE:
            part += 1
            fname = f"train-{part:05d}-of-XXXXX.parquet"
            local_path = os.path.join(OUT_DIR, fname)
            table = rows_to_table(buf)
            pq.write_table(table, local_path, compression="snappy")
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"  chunk {part:3d}: {len(buf):,} rows → {local_path} ({size_mb:.1f} MB) — uploading …")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"data/{fname}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            print(f"           uploaded ✓")
            total_written += len(buf)
            buf.clear()

# flush last chunk
if buf:
    part += 1
    fname = f"train-{part:05d}-of-XXXXX.parquet"
    local_path = os.path.join(OUT_DIR, fname)
    table = rows_to_table(buf)
    pq.write_table(table, local_path, compression="snappy")
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  chunk {part:3d}: {len(buf):,} rows → {local_path} ({size_mb:.1f} MB) — uploading …")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"data/{fname}",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"           uploaded ✓")
    total_written += len(buf)

# rename files to include correct total — upload a README too
total_parts = part
print(f"\nRenaming files to include correct total ({total_parts} parts) …")
for i in range(1, total_parts + 1):
    old_name = f"train-{i:05d}-of-XXXXX.parquet"
    new_name = f"train-{i:05d}-of-{total_parts:05d}.parquet"
    old_local = os.path.join(OUT_DIR, old_name)
    new_local = os.path.join(OUT_DIR, new_name)
    os.rename(old_local, new_local)
    # re-upload with correct name, delete old
    api.upload_file(
        path_or_fileobj=new_local,
        path_in_repo=f"data/{new_name}",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    api.delete_file(
        path_in_repo=f"data/{old_name}",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

# ── Step 3: upload README / dataset card ────────────────────────────────────
readme = f"""---
license: other
task_categories:
- text-classification
- text-generation
language:
- en
tags:
- goodreads
- reviews
- books
size_categories:
- 1M<n<10M
---

# Goodreads Reviews (deduplicated)

~{total_written:,} book reviews scraped from Goodreads, deduplicated.

## Columns

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Anonymised user hash |
| book_id | string | Goodreads book ID |
| review_id | string | Unique review ID |
| rating | int8 | 1–5 star rating (0 = no rating) |
| review_text | string | Full review text |
| date_added | string | Date added to shelf |
| date_updated | string | Date last updated |
| read_at | string | Date finished reading |
| started_at | string | Date started reading |
| n_votes | int32 | Number of helpful votes |
| n_comments | int32 | Number of comments |

## Files

{total_parts} Parquet files (Snappy-compressed) under `data/`, each with up to {CHUNK_SIZE:,} rows.

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("{REPO_ID}")
print(ds["train"][0])
```
"""

readme_path = os.path.join(OUT_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"\nDone! {total_written:,} records in {total_parts} Parquet files.")
print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")
