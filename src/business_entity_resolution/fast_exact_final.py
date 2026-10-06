import pandas as pd
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

ROOT = Path(".")
TEST = ROOT / "dataset" / "test"
OUT = ROOT / "output"

def norm(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

print("Loading Source1...")
s1 = pd.read_csv(TEST / "test_source1.tsv", sep="\t", dtype=str,
                 usecols=["entity_id","business_name","business_address","country"]).fillna("")

s1["name_key"] = s1["business_name"].map(norm)
s1["addr_key"] = s1["business_address"].map(norm)
s1["country_key"] = s1["country"].map(norm)

name_map = defaultdict(list)
addr_map = defaultdict(list)

for r in s1.itertuples(index=False):
    if r.name_key and r.country_key:
        name_map[(r.name_key, r.country_key)].append(r.entity_id)
    if r.addr_key and r.country_key:
        addr_map[(r.addr_key, r.country_key)].append(r.entity_id)

candidates = defaultdict(list)

def process_file(path, source):
    print("Processing", source, "...")
    for chunk in pd.read_csv(path, sep="\t", dtype=str,
                              usecols=["entity_id","business_name","business_address","country"],
                              chunksize=100000):
        chunk = chunk.fillna("")
        chunk["name_key"] = chunk["business_name"].map(norm)
        chunk["addr_key"] = chunk["business_address"].map(norm)
        chunk["country_key"] = chunk["country"].map(norm)

        for r in chunk.itertuples(index=False):
            target = r.entity_id
            hits = set()

            if r.name_key and r.country_key:
                hits.update(name_map.get((r.name_key, r.country_key), []))

            if r.addr_key and r.country_key:
                hits.update(addr_map.get((r.addr_key, r.country_key), []))

            for s1id in hits:
                candidates[s1id].append(f"{source}:{target}")

print("Generating candidates...")
process_file(TEST / "test_source2.tsv", "S2")
process_file(TEST / "test_source3.tsv", "S3")

print("Writing results...")

matching_rows = []
candidate_rows = []

for s1id in s1["entity_id"]:
    vals = list(dict.fromkeys(candidates.get(s1id, [])))

    candidate_rows.append({
        "source1_entity_id": s1id,
        "candidate_entity_ids": ",".join(vals)
    })

    matching_rows.append({
        "source1_entity_id": s1id,
        "matched_entity_ids": ",".join(vals)
    })

pd.DataFrame(matching_rows).to_csv(
    OUT / "matching_results.tsv", sep="\t", index=False
)

pd.DataFrame(candidate_rows).to_csv(
    OUT / "candidate_pairs.tsv", sep="\t", index=False
)

nonempty = sum(bool(x["matched_entity_ids"]) for x in matching_rows)

print("DONE")
print("Source1:", len(s1))
print("Non-empty matches:", nonempty)
print("Empty matches:", len(s1) - nonempty)
