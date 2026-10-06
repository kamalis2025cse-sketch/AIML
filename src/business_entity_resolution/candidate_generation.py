import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from preprocessing import load_data, preprocess_dataframe


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "output"


# --------------------------------------------------
# CREATE TF-IDF CANDIDATES
# --------------------------------------------------

def generate_candidates(source1, target, top_k=40):

    print("  Building name TF-IDF index...")

    name_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=100000
    )

    target_names = target["name_normalized"].fillna("")
    source1_names = source1["name_normalized"].fillna("")

    target_name_matrix = name_vectorizer.fit_transform(
        target_names
    )

    source1_name_matrix = name_vectorizer.transform(
        source1_names
    )

    name_neighbors = NearestNeighbors(
        n_neighbors=min(top_k, len(target)),
        metric="cosine",
        algorithm="brute"
    )

    name_neighbors.fit(target_name_matrix)

    _, name_indices = name_neighbors.kneighbors(
        source1_name_matrix
    )

    print("  Building address TF-IDF index...")

    address_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=100000
    )

    target_addresses = target["address_normalized"].fillna("")
    source1_addresses = source1["address_normalized"].fillna("")

    target_address_matrix = address_vectorizer.fit_transform(
        target_addresses
    )

    source1_address_matrix = address_vectorizer.transform(
        source1_addresses
    )

    address_neighbors = NearestNeighbors(
        n_neighbors=min(top_k, len(target)),
        metric="cosine",
        algorithm="brute"
    )

    address_neighbors.fit(target_address_matrix)

    _, address_indices = address_neighbors.kneighbors(
        source1_address_matrix
    )

    # --------------------------------------------------
    # COMBINE NAME + ADDRESS CANDIDATES
    # --------------------------------------------------

    candidates = []

    for source1_idx in range(len(source1)):

        candidate_indices = set()

        # Name candidates
        candidate_indices.update(
            name_indices[source1_idx]
        )

        # Address candidates
        candidate_indices.update(
            address_indices[source1_idx]
        )

        source1_id = source1.iloc[source1_idx]["entity_id"]

        for target_idx in candidate_indices:

            target_id = target.iloc[target_idx]["entity_id"]

            candidates.append(
                {
                    "source1_entity_id": source1_id,
                    "target_entity_id": target_id
                }
            )

    return pd.DataFrame(candidates)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading data...")

    source1, source2, source3, ground_truth = load_data()

    print("Preprocessing data...")

    source1 = preprocess_dataframe(source1)
    source2 = preprocess_dataframe(source2)
    source3 = preprocess_dataframe(source3)

    # --------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------

    print("\nGenerating Source 2 candidates...")

    candidates_s2 = generate_candidates(
        source1,
        source2,
        top_k=40
    )

    candidates_s2["target_source"] = "S2"

    print(
        "Source 2 candidate pairs:",
        len(candidates_s2)
    )

    # --------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------

    print("\nGenerating Source 3 candidates...")

    candidates_s3 = generate_candidates(
        source1,
        source3,
        top_k=40
    )

    candidates_s3["target_source"] = "S3"

    print(
        "Source 3 candidate pairs:",
        len(candidates_s3)
    )

    # --------------------------------------------------
    # COMBINE
    # --------------------------------------------------

    all_candidates = pd.concat(
        [
            candidates_s2,
            candidates_s3
        ],
        ignore_index=True
    )

    all_candidates = all_candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "target_entity_id"
        ]
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / "candidate_pairs_10000.tsv"

    all_candidates.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print("\n-----------------------------------")
    print("CANDIDATE GENERATION COMPLETED")
    print("-----------------------------------")

    print(
        "Total candidate pairs:",
        len(all_candidates)
    )

    print(
        "Saved to:",
        output_file
    )

    print("\nCandidate sample:")
    print(all_candidates.head(10))