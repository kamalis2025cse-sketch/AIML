import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from rapidfuzz.fuzz import ratio, token_set_ratio

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"

MODEL = joblib.load(OUTPUT / "matching_model_10000.pkl")

with open(OUTPUT / "matching_threshold_10000.txt") as f:
    THRESHOLD = float(f.read().strip())

print("Loading test data...")

s1 = pd.read_csv(TEST / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s2 = pd.read_csv(TEST / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
s3 = pd.read_csv(TEST / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))
print("Threshold:", THRESHOLD)

def normalize(x):
    return str(x).lower().strip()

def features(a, b):
    n1 = normalize(a["business_name"])
    n2 = normalize(b["business_name"])
    ad1 = normalize(a["business_address"])
    ad2 = normalize(b["business_address"])
    c1 = normalize(a["country"])
    c2 = normalize(b["country"])

    return [
        ratio(n1, n2) / 100.0,
        token_set_ratio(n1, n2) / 100.0,
        ratio(ad1, ad2) / 100.0,
        token_set_ratio(ad1, ad2) / 100.0,
        int(n1 == n2 and n1 != ""),
        int(ad1 == ad2 and ad1 != ""),
        int(c1 == c2 and c1 != "")
    ]

print("Building test indexes...")

s2_by_id = s2.set_index("entity_id")
s3_by_id = s3.set_index("entity_id")

# Generate candidates using the same TF-IDF approach used during development.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def generate_candidates(source1, target, target_name):
    print("Generating", target_name, "candidates...")

    name_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=100000
    )

    address_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=100000
    )

    name_matrix = name_vectorizer.fit_transform(
        target["business_name"].fillna("")
    )

    address_matrix = address_vectorizer.fit_transform(
        target["business_address"].fillna("")
    )

    name_query = name_vectorizer.transform(
        source1["business_name"].fillna("")
    )

    address_query = address_vectorizer.transform(
        source1["business_address"].fillna("")
    )

    name_nn = NearestNeighbors(
        n_neighbors=min(40, len(target)),
        metric="cosine",
        n_jobs=-1
    )

    address_nn = NearestNeighbors(
        n_neighbors=min(40, len(target)),
        metric="cosine",
        n_jobs=-1
    )

    name_nn.fit(name_matrix)
    address_nn.fit(address_matrix)

    _, name_indices = name_nn.kneighbors(name_query)
    _, address_indices = address_nn.kneighbors(address_query)

    pairs = set()

    for i in range(len(source1)):
        for j in name_indices[i]:
            pairs.add((
                source1.iloc[i]["entity_id"],
                target.iloc[j]["entity_id"]
            ))

        for j in address_indices[i]:
            pairs.add((
                source1.iloc[i]["entity_id"],
                target.iloc[j]["entity_id"]
            ))

    print(target_name, "candidate pairs:", len(pairs))

    return list(pairs)

candidates2 = generate_candidates(s1, s2, "Source 2")
candidates3 = generate_candidates(s1, s3, "Source 3")

all_candidates = candidates2 + candidates3

print("Total test candidates:", len(all_candidates))

print("Scoring candidates...")

matches = {sid: [] for sid in s1["entity_id"]}

batch_size = 50000

for start in range(0, len(all_candidates), batch_size):

    batch = all_candidates[start:start + batch_size]

    X = []
    metadata = []

    for sid, tid in batch:

        a = s1.loc[s1["entity_id"] == sid].iloc[0]

        if tid.startswith("S2-"):
            b = s2_by_id.loc[tid]
        else:
            b = s3_by_id.loc[tid]

        X.append(features(a, b))
        metadata.append((sid, tid))

    X = np.asarray(X, dtype=np.float32)

    probabilities = MODEL.predict_proba(X)[:, 1]

    for (sid, tid), probability in zip(metadata, probabilities):

        if probability >= THRESHOLD:
            matches[sid].append(tid)

    print("Scored:", min(start + batch_size, len(all_candidates)),
          "/", len(all_candidates))

print("Removing duplicate matches...")

for sid in matches:
    matches[sid] = sorted(set(matches[sid]))

print("Writing matching_results.tsv...")

result_rows = []

for sid in s1["entity_id"]:
    result_rows.append({
        "source1_entity_id": sid,
        "matched_entity_ids": ",".join(matches[sid])
    })

result = pd.DataFrame(result_rows)

result.to_csv(
    OUTPUT / "matching_results.tsv",
    sep="\t",
    index=False
)

print("Writing candidate_pairs.tsv...")

candidate_df = pd.DataFrame(
    all_candidates,
    columns=["source1_entity_id", "target_entity_id"]
)

candidate_df["target_source"] = candidate_df["target_entity_id"].apply(
    lambda x: "S2" if str(x).startswith("S2-") else "S3"
)

candidate_df.to_csv(
    OUTPUT / "candidate_pairs.tsv",
    sep="\t",
    index=False
)

print("-----------------------------------")
print("FINAL MATCHING COMPLETED")
print("-----------------------------------")
print("Matching file:")
print(OUTPUT / "matching_results.tsv")
print("Candidate file:")
print(OUTPUT / "candidate_pairs.tsv")
print("Test Source1 records:", len(s1))
print("Total candidates:", len(all_candidates))
print("Source1 records with matches:",
      sum(1 for x in matches.values() if x))
print("Total predicted matches:",
      sum(len(x) for x in matches.values()))
