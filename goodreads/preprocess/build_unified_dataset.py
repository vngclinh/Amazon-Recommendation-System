"""
build_unified_dataset.py
------------------------
Join goodreads reviews (HuggingFace) + UCSD book metadata + genres
then push unified dataset back to HuggingFace.

Requirements:
    pip install duckdb pandas pyarrow datasets huggingface_hub tqdm

Local files needed (download from UCSD):
    ./goodreads_books.json.gz
    ./goodreads_book_genres_initial.json.gz

Usage:
    HF_TOKEN=hf_xxx python build_unified_dataset.py
"""

import os, gzip, json, time, logging
import pandas as pd
import duckdb
from dotenv import load_dotenv
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO_IN = "vngclinh/goodreads-reviews"
HF_REPO_OUT    = "vngclinh/goodreads-concats"   # đổi thành repo của bạn
BOOKS_FILE   = r"C:\Users\OS\Downloads\goodreads_books.json.gz"
GENRES_FILE  = r"D:\goodreads_book_genres_initial.json.gz"
REVIEWS_DIR  = "./reviews_local"   # download 32 parquet files về đây
OUT_DIR      = "./unified_parquet"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
# ─────────────────────────────────────────────────────────────────────────────


def download_reviews(repo_id: str, local_dir: str, token: str):
    """Download all parquet files from HF to local — avoids HTTP timeout during JOIN."""
    os.makedirs(local_dir, exist_ok=True)
 
    # List all parquet files in the repo
    all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    log.info(f"Found {len(parquet_files)} parquet files to download")
 
    for fname in tqdm(parquet_files, desc="downloading reviews"):
        out_path = os.path.join(local_dir, os.path.basename(fname))
        if os.path.exists(out_path):
            log.info(f"  Skip (exists): {fname}")
            continue
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=fname,
            local_dir=local_dir,
            token=token,
        )
 
    downloaded = [f for f in os.listdir(local_dir) if f.endswith(".parquet")]
    log.info(f"  {len(downloaded)} files ready in {local_dir}/")
 
 
def load_genres(path: str) -> pd.DataFrame:
    """Load genres_initial → one row per book with primary_genre + top3."""
    log.info("Loading genres...")
    rows = []
    with gzip.open(path, "rt") as f:
        for line in tqdm(f, desc="genres"):
            obj = json.loads(line)
            g = obj.get("genres", {})
            if not g:
                continue
            sorted_genres = sorted(g.items(), key=lambda x: x[1], reverse=True)
            rows.append({
                "book_id":       str(obj["book_id"]),
                "primary_genre": sorted_genres[0][0],
                "genre_2":       sorted_genres[1][0] if len(sorted_genres) > 1 else None,
                "genre_3":       sorted_genres[2][0] if len(sorted_genres) > 2 else None,
                "primary_genre_count": sorted_genres[0][1],
            })
    df = pd.DataFrame(rows)
    log.info(f"  Genres loaded: {len(df):,} books")
    log.info(f"  Top genres:\n{df.primary_genre.value_counts().head(8).to_string()}")
    return df
 
 
def load_books(path: str) -> pd.DataFrame:
    """Load books metadata — only fields needed for pipeline."""
    log.info("Loading books metadata (2GB, takes ~3–5 min)...")
    rows = []
    with gzip.open(path, "rt") as f:
        for line in tqdm(f, desc="books"):
            b = json.loads(line)
            desc = b.get("description", "") or ""
            rows.append({
                "book_id":          str(b["book_id"]),
                "title":            b.get("title", ""),
                "description":      desc[:1000],   # cap at 1000 chars
                "author_id":        (b.get("authors") or [{}])[0].get("author_id", ""),
                "publication_year": b.get("publication_year", ""),
                "num_pages":        b.get("num_pages", ""),
                "avg_rating_ucsd":  b.get("average_rating", ""),
            })
    df = pd.DataFrame(rows)
    log.info(f"  Books loaded: {len(df):,}")
    return df
 
 
def build_meta(books_path: str, genres_path: str) -> pd.DataFrame:
    """Merge books + genres, save as local parquet for DuckDB to join."""
    df_genres = load_genres(genres_path)
    df_books  = load_books(books_path)
 
    df_meta = df_books.merge(df_genres, on="book_id", how="inner")
    log.info(f"  After merge books+genres: {len(df_meta):,} books")
 
    out = "book_meta.parquet"
    df_meta.to_parquet(out, index=False)
    log.info(f"  Saved → {out}")
    return df_meta
 
 
def check_join_rate(con, reviews_dir):
    """Quick sanity check: what % of review book_ids have metadata."""
    local_pattern = reviews_dir.replace("\\", "/") + "/**/*.parquet"
    result = con.execute(f"""
        SELECT
            COUNT(DISTINCT r.book_id)                    AS total_book_ids,
            COUNT(DISTINCT CASE WHEN b.book_id IS NOT NULL
                           THEN r.book_id END)           AS matched_book_ids
        FROM (
            SELECT DISTINCT book_id
            FROM read_parquet('{local_pattern}')
            USING SAMPLE 50000 ROWS
        ) r
        LEFT JOIN read_parquet('book_meta.parquet') b
            ON r.book_id = b.book_id
    """).fetchone()
    total, matched = result
    rate = matched / total if total else 0
    log.info(f"  Join rate check: {matched}/{total} = {rate:.1%}")
    if rate < 0.4:
        log.warning("  Join rate < 40% -- check book_id type mismatch!")
    return rate
 
 
def build_unified(con, reviews_dir, out_dir):
    """Run the main join query on LOCAL files -- no HTTP, no timeout."""
    os.makedirs(out_dir, exist_ok=True)
    local_pattern = reviews_dir.replace("\\", "/") + "/**/*.parquet"
    out_dir_fwd   = out_dir.replace("\\", "/")
 
    log.info("Counting rows to join (local, fast)...")
    total = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{local_pattern}') r
        JOIN read_parquet('book_meta.parquet') b ON r.book_id = b.book_id
        WHERE r.rating > 0
          AND r.review_text IS NOT NULL
          AND LENGTH(TRIM(r.review_text)) > 20
    """).fetchone()[0]
    log.info(f"  Rows to write: {total:,}")
 
    log.info("Writing unified parquet (partitioned by primary_genre)...")
    query = f"""
        SELECT
            r.review_id,
            r.user_id,
            r.book_id,
            CAST(r.rating AS INTEGER)       AS rating,
            r.review_text,
            r.n_votes,
            r.n_comments,
            r.date_added,
            b.title,
            b.description,
            b.author_id,
            b.publication_year,
            b.num_pages,
            b.primary_genre,
            b.genre_2,
            b.genre_3,
            b.primary_genre_count
        FROM read_parquet('{local_pattern}') r
        JOIN read_parquet('book_meta.parquet') b
            ON r.book_id = b.book_id
        WHERE r.rating > 0
          AND r.review_text IS NOT NULL
          AND LENGTH(TRIM(r.review_text)) > 20
        ORDER BY r.date_added
    """
    con.execute(f"""
        COPY ({query})
        TO '{out_dir_fwd}'
        (FORMAT PARQUET, PARTITION_BY (primary_genre))
    """)
    log.info(f"  Written to {out_dir}/")
    files = []
    for root, _, fnames in os.walk(out_dir):
        for fn in fnames:
            if fn.endswith(".parquet"):
                files.append(os.path.join(root, fn))
    log.info(f"  {len(files)} parquet files created")
    return files
 
 
def push_to_hf(out_dir: str, repo_id: str, token: str):
    """Push all parquet files to HuggingFace dataset repo."""
    api = HfApi(token=token)
 
    # Create repo if not exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        log.info(f"  Repo ready: {repo_id}")
    except Exception as e:
        log.warning(f"  Repo create warning: {e}")
 
    # Upload all parquet files
    files = []
    for root, _, fnames in os.walk(out_dir):
        for fn in fnames:
            if fn.endswith(".parquet"):
                files.append(os.path.join(root, fn))
 
    log.info(f"  Uploading {len(files)} files to {repo_id}...")
    for fpath in tqdm(files, desc="upload"):
        # Preserve subdirectory structure (genre partitions)
        rel = os.path.relpath(fpath, out_dir)
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"data/{rel}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {rel}"
        )
    log.info(f"  Done → https://huggingface.co/datasets/{repo_id}")
 
 
def create_dataset_card(repo_id: str, token: str, n_rows: int):
    """Push a simple README.md as dataset card."""
    api = HfApi(token=token)
    card = f"""---
language:
- en
tags:
- books
- reviews
- goodreads
- recommendation
- nlp
size_categories:
- 1M<n<10M
---
 
# Goodreads Unified Reviews
 
Joined dataset: **{n_rows:,} reviews** with book metadata and genre labels.
 
## Sources
- Reviews: [vngclinh/goodreads-reviews](https://huggingface.co/datasets/vngclinh/goodreads-reviews)
- Book metadata: [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html)
 
## Schema
 
| Column | Type | Description |
|--------|------|-------------|
| review_id | string | Unique review ID |
| user_id | string | Anonymised user hash |
| book_id | string | Goodreads book ID |
| rating | int | 1–5 star rating |
| review_text | string | Full review text |
| n_votes | int | Helpful votes |
| n_comments | int | Number of comments |
| date_added | string | Date added to shelf |
| title | string | Book title |
| description | string | Book description (capped 1000 chars) |
| author_id | string | Primary author ID |
| publication_year | string | Year published |
| num_pages | string | Page count |
| primary_genre | string | Dominant genre from shelf tags |
| genre_2 | string | Second genre |
| genre_3 | string | Third genre |
| primary_genre_count | int | Shelf count for primary genre |
 
## Genre distribution
Partitioned by `primary_genre` for efficient genre-level queries.
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card"
    )
 
 
# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    token = HF_TOKEN
    if not token:
        raise ValueError("Set HF_TOKEN env var: export HF_TOKEN=hf_xxx (or add it to .env)")
 
    t0 = time.time()
 
    # Step 1: Build book_meta.parquet from local files
    log.info("=== Step 1: Build book metadata ===")
    build_meta(BOOKS_FILE, GENRES_FILE)
 
    # Step 2: Download reviews locally (skip if already downloaded)
    log.info("=== Step 2: Download reviews locally ===")
    download_reviews(HF_REPO_IN, REVIEWS_DIR, token)
 
    # Step 3: Connect DuckDB and sanity-check join rate
    log.info("=== Step 3: Join rate check ===")
    con = duckdb.connect()
    check_join_rate(con, REVIEWS_DIR)
 
    # Step 4: Run full join and write parquet partitioned by genre
    log.info("=== Step 4: Build unified dataset ===")
    build_unified(con, REVIEWS_DIR, OUT_DIR)
 
    # Count final rows
    out_pattern = OUT_DIR.replace("\\", "/") + "/**/*.parquet"
    n_rows = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{out_pattern}')
    """).fetchone()[0]
    log.info(f"  Final unified rows: {n_rows:,}")
 
    # Step 5: Push to HuggingFace
    log.info("=== Step 5: Push to HuggingFace ===")
    push_to_hf(OUT_DIR, HF_REPO_OUT, token)
    create_dataset_card(HF_REPO_OUT, token, n_rows)
 
    log.info(f"=== Done in {(time.time()-t0)/60:.1f} min ===")
 