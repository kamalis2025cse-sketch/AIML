import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib

try:
    from rapidfuzz.fuzz import ratio, token_set_ratio
except ImportError:
    print("rapidfuzz is not installed.")
    print("Run: pip install rapidfuzz")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "train_10000_corrected"
OUTPUT = ROOT / "output"

print("Loading training data...")

s1 = pd.read_csv(
    DATA / "train_source1_corrected.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = pd.read_csv(
    DATA / "train_source2_corrected.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    DATA / "train_source3_corrected.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

gt = pd.read_csv(
    DATA / "train_ground_truth_corrected.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

candidates = pd.read_csv(
    OUTPUT / "candidate_pairs_10000.tsv",
    sep="\t",
    dtype=str
)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))
print("Candidates:", len(candidates))

s1 = s1.set_index("entity_id")
s2 = s2.set_index("entity_id")
s3 = s3.set_index("entity_id")

positive_pairs = set()

for _, row in gt.iterrows():
    sid = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"]).strip()

    if matches:
        for tid in matches.split(","):
            tid = tid.strip()
            if tid:
                positive_pairs.add((sid, tid))

print("Positive pairs:", len(positive_pairs))


def get_record(entity_id):
    if entity_id.startswith("S2-"):
        return s2.loc[entity_id]
    return s3.loc[entity_id]


def make_features(sid, tid):
    a = s1.loc[sid]
    b = get_record(tid)

    name1 = str(a["business_name"])
    name2 = str(b["business_name"])

    addr1 = str(a["business_address"])
    addr2 = str(b["business_address"])

    country1 = str(a["country"]).lower()
    country2 = str(b["country"]).lower()

    name_ratio = ratio(name1, name2) / 100.0
    name_token = token_set_ratio(name1, name2) / 100.0
    addr_ratio = ratio(addr1, addr2) / 100.0
    addr_token = token_set_ratio(addr1, addr2) / 100.0

    name_exact = int(
        name1.strip().lower() == name2.strip().lower()
        and name1.strip() != ""
    )

    addr_exact = int(
        addr1.strip().lower() == addr2.strip().lower()
        and addr1.strip() != ""
    )

    country_match = int(
        country1.strip() != ""
        and country1 == country2
    )

    return [
        name_ratio,
        name_token,
        addr_ratio,
        addr_token,
        name_exact,
        addr_exact,
        country_match
    ]


print("Creating training features...")

# Use all positive pairs and a controlled number of negative pairs.
candidate_pairs = list(
    zip(
        candidates["source1_entity_id"].astype(str),
        candidates["target_entity_id"].astype(str)
    )
)

positive_candidates = []
negative_candidates = []

for pair in candidate_pairs:
    if pair in positive_pairs:
        positive_candidates.append(pair)
    else:
        negative_candidates.append(pair)

print("Positive candidates:", len(positive_candidates))
print("Negative candidates:", len(negative_candidates))

# Keep a manageable negative sample while preserving all positives.
rng = np.random.default_rng(42)

negative_count = min(
    len(negative_candidates),
    max(len(positive_candidates) * 4, 100000)
)

negative_indices = rng.choice(
    len(negative_candidates),
    size=negative_count,
    replace=False
)

negative_sample = [
    negative_candidates[i]
    for i in negative_indices
]

training_pairs = positive_candidates + negative_sample

rng.shuffle(training_pairs)

print("Training pairs:", len(training_pairs))

X = []
y = []

for i, (sid, tid) in enumerate(training_pairs):
    if i % 50000 == 0:
        print("Features:", i, "/", len(training_pairs))

    X.append(make_features(sid, tid))
    y.append(1 if (sid, tid) in positive_pairs else 0)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int8)

print("Feature matrix:", X.shape)
print("Positive labels:", int(y.sum()))
print("Negative labels:", int((y == 0).sum()))

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training model...")

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")

prob = model.predict_proba(X_val)[:, 1]

best_threshold = 0.80
best_f05 = -1

print("Searching threshold...")

for threshold in np.arange(0.50, 0.991, 0.01):
    pred = (prob >= threshold).astype(int)

    score = fbeta_score(
        y_val,
        pred,
        beta=0.5,
        zero_division=0
    )

    if score > best_f05:
        best_f05 = score
        best_threshold = float(threshold)

pred = (prob >= best_threshold).astype(int)

print("-----------------------------------")
print("VALIDATION RESULTS")
print("-----------------------------------")
print("Best threshold:", best_threshold)
print("F0.5:", round(best_f05, 4))
print("Precision:", round(precision_score(y_val, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, pred, zero_division=0), 4))

MODEL_DIR = ROOT / "output"

joblib.dump(
    model,
    MODEL_DIR / "matching_model_10000.pkl"
)

with open(MODEL_DIR / "matching_threshold_10000.txt", "w") as f:
    f.write(str(best_threshold))

print("-----------------------------------")
print("MATCHING MODEL COMPLETED")
print("-----------------------------------")
print("Model saved to:")
print(MODEL_DIR / "matching_model_10000.pkl")
print("Threshold saved to:")
print(MODEL_DIR / "matching_threshold_10000.txt")
