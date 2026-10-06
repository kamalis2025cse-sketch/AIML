import pandas as pd
import numpy as np
import joblib
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"

model = joblib.load(OUTPUT / "matching_model_10000.pkl")

with open(OUTPUT / "matching_threshold_10000.txt") as f:
    threshold = float(f.read().strip())

print("Loading test Source1...")
s1 = pd.read_csv(TEST / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)

print("Source1:", len(s1))
print("Building fast blocking indexes...")

def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    x = x.lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def key_name(x):
    return norm(x)

def key_address(x):
    return norm(x)

# Source1 lookup keys
name_map = {}
address_map = {}
country_name_map = {}

for _, r in s1.iterrows():
    sid = r["entity_id"]
    n = key_name(r["business_name"])
    a = key_address(r["business_address"])
    c = norm(r["country"])

    if n:
        name_map.setdefault(n, []).append(sid)

    if a:
        address_map.setdefault(a, []).append(sid)

    if n and c:
        country_name_map.setdefault((c, n), []).append(sid)

print("Name keys:", len(name_map))
print("Address keys:", len(address_map))

candidate_pairs = set()

def process_source(path, prefix):
    print("Processing", prefix, "...")

    count = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100000
    ):
        for _, r in chunk.iterrows():

            tid = r["entity_id"]
            n = key_name(r["business_name"])
            a = key_address(r["business_address"])
            c = norm(r["country"])

            matched_sids = set()

            if n:
                matched_sids.update(name_map.get(n, []))

            if a:
                matched_sids.update(address_map.get(a, []))

            if n and c:
                matched_sids.update(
                    country_name_map.get((c, n), [])
                )

            for sid in matched_sids:
                candidate_pairs.add((sid, tid, prefix))

        count += len(chunk)
        if count % 500000 == 0:
            print(prefix, "processed:", count)

    print(prefix, "completed:", count)

process_source(TEST / "test_source2.tsv", "S2")
process_source(TEST / "test_source3.tsv", "S3")

print("Fast candidates:", len(candidate_pairs))

print("Creating candidate file...")

candidate_df = pd.DataFrame(
    list(candidate_pairs),
    columns=[
        "source1_entity_id",
        "target_entity_id",
        "target_source"
    ]
)

candidate_df.to_csv(
    OUTPUT / "candidate_pairs.tsv",
    sep="\t",
    index=False
)

print("Candidate file saved.")

print("Generating final matches...")

s1_lookup = s1.set_index("entity_id")

# Load only records that became candidates.
needed_s2 = set(
    x[1] for x in candidate_pairs if x[2] == "S2"
)

needed_s3 = set(
    x[1] for x in candidate_pairs if x[2] == "S3"
)

target_records = {}

for path, needed in [
    (TEST / "test_source2.tsv", needed_s2),
    (TEST / "test_source3.tsv", needed_s3)
]:
    if not needed:
        continue

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100000
    ):
        for _, r in chunk.iterrows():
            if r["entity_id"] in needed:
                target_records[r["entity_id"]] = r

print("Target records loaded:", len(target_records))

matches = {sid: [] for sid in s1["entity_id"]}

for i, (sid, tid, source) in enumerate(candidate_pairs):

    a = s1_lookup.loc[sid]
    b = target_records.get(tid)

    if b is None:
        continue

    n1 = norm(a["business_name"])
    n2 = norm(b["business_name"])
    a1 = norm(a["business_address"])
    a2 = norm(b["business_address"])
    c1 = norm(a["country"])
    c2 = norm(b["country"])

    # Strong exact blocking candidates receive a high-confidence feature vector.
    features = np.array([[
        float(n1 == n2 and n1 != ""),
        float(n1 == n2 and n1 != ""),
        float(a1 == a2 and a1 != ""),
        float(a1 == a2 and a1 != ""),
        int(n1 == n2 and n1 != ""),
        int(a1 == a2 and a1 != ""),
        int(c1 == c2 and c1 != "")
    ]], dtype=np.float32)

    probability = model.predict_proba(features)[0, 1]

    if probability >= threshold:
        matches[sid].append(tid)

print("Writing matching_results.tsv...")

rows = []

for sid in s1["entity_id"]:
    ids = sorted(set(matches[sid]))

    rows.append({
        "source1_entity_id": sid,
        "matched_entity_ids": ",".join(ids)
    })

pd.DataFrame(rows).to_csv(
    OUTPUT / "matching_results.tsv",
    sep="\t",
    index=False
)

print("-----------------------------------")
print("FINAL FILES CREATED")
print("-----------------------------------")
print("candidate_pairs.tsv")
print("matching_results.tsv")
print("Source1 rows:", len(rows))
print("Predicted matched Source1:", sum(bool(x) for x in matches.values()))
print("Total matches:", sum(len(x) for x in matches.values()))
