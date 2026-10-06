import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "dataset" / "train_10000"
CANDIDATE_FILE = PROJECT_ROOT / "output" / "candidate_pairs_10000.tsv"


# --------------------------------------------------
# LOAD GROUND TRUTH
# --------------------------------------------------

def load_ground_truth():

    file_path = DATA_DIR / "train_ground_truth_10000.tsv"

    ground_truth = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return ground_truth


# --------------------------------------------------
# CONVERT GROUND TRUTH INTO PAIRS
# --------------------------------------------------

def create_true_pairs(ground_truth):

    true_pairs = set()

    for _, row in ground_truth.iterrows():

        source1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for target_id in matched_ids.split(","):

            target_id = target_id.strip()

            if target_id:
                true_pairs.add(
                    (
                        source1_id,
                        target_id
                    )
                )

    return true_pairs


# --------------------------------------------------
# EVALUATE CANDIDATES
# --------------------------------------------------

def evaluate_candidates():

    print("Loading ground truth...")

    ground_truth = load_ground_truth()

    print("Loading candidate pairs...")

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str
    )

    true_pairs = create_true_pairs(
        ground_truth
    )

    candidate_pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["target_entity_id"]
        )
    )

    # True matches that were generated as candidates
    recovered_pairs = true_pairs.intersection(
        candidate_pairs
    )

    total_true = len(true_pairs)

    recovered = len(recovered_pairs)

    if total_true > 0:
        recall = recovered / total_true
    else:
        recall = 0

    print("\n-----------------------------------")
    print("CANDIDATE GENERATION EVALUATION")
    print("-----------------------------------")

    print(
        "Total true matching pairs:",
        total_true
    )

    print(
        "True matches found in candidates:",
        recovered
    )

    print(
        "True matches missed:",
        total_true - recovered
    )

    print(
        f"Candidate recall: {recall:.4f}"
    )

    print(
        f"Candidate recall: {recall * 100:.2f}%"
    )

    # Show some missed examples
    missed_pairs = list(
        true_pairs - candidate_pairs
    )

    print("\nNumber of missed true pairs:", len(missed_pairs))

    if missed_pairs:

        print("\nFirst 20 missed true pairs:")

        for source1_id, target_id in missed_pairs[:20]:

            print(
                source1_id,
                "->",
                target_id
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    evaluate_candidates()