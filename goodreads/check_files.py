"""Check all 32 parquet files for readability and collect good-file list."""
import duckdb, json, time

REPO  = "vngclinh/goodreads-reviews"
BASE  = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
N     = 32

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

good, bad = [], []
for i in range(1, N + 1):
    url = f"{BASE}data/train-{i:05d}-of-{N:05d}.parquet"
    try:
        t0 = time.time()
        cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{url}')").fetchone()[0]
        print(f"  [{i:02d}] ok  — {cnt:>9,} rows  ({time.time()-t0:.1f}s)")
        good.append(url)
    except Exception as e:
        short = str(e)[:80]
        print(f"  [{i:02d}] BAD — {short}")
        bad.append(url)

print(f"\n{len(good)} good / {len(bad)} bad")
if bad:
    print("Bad files:", bad)

with open(r"D:\goodreads\good_files.json", "w") as f:
    json.dump(good, f, indent=2)
print("Saved good_files.json")
