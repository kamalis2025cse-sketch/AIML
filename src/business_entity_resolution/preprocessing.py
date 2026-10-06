import pandas as pd
import re
import unicodedata
from pathlib import Path


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "dataset" / "train_10000_corrected"


# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    """
    Normalize business names and addresses.

    Example:
    "ABC Pvt. Ltd." -> "abc pvt ltd"
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert accented characters to normal characters
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Convert to lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    source1 = pd.read_csv(
        DATA_DIR / "train_source1_corrected.tsv",
        sep="\t",
        dtype=str
    )

    source2 = pd.read_csv(
        DATA_DIR / "train_source2_corrected.tsv",
        sep="\t",
        dtype=str
    )

    source3 = pd.read_csv(
        DATA_DIR / "train_source3_corrected.tsv",
        sep="\t",
        dtype=str
    )

    ground_truth = pd.read_csv(
        DATA_DIR / "train_ground_truth_corrected.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return source1, source2, source3, ground_truth
# --------------------------------------------------
# PREPROCESS DATA
# --------------------------------------------------

def preprocess_dataframe(df):

    df = df.copy()

    df["name_normalized"] = df["business_name"].apply(
        normalize_text
    )

    df["address_normalized"] = df["business_address"].apply(
        normalize_text
    )

    df["country_normalized"] = df["country"].apply(
        normalize_text
    )

    return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    source1, source2, source3, ground_truth = load_data()

    print("\nOriginal record counts:")
    print("Source 1:", len(source1))
    print("Source 2:", len(source2))
    print("Source 3:", len(source3))
    print("Ground truth:", len(ground_truth))

    source1 = preprocess_dataframe(source1)
    source2 = preprocess_dataframe(source2)
    source3 = preprocess_dataframe(source3)

    print("\nPreprocessing completed successfully!")

    print("\nSource 1 sample:")
    print(
        source1[
            [
                "entity_id",
                "business_name",
                "name_normalized",
                "business_address",
                "address_normalized",
                "country"
            ]
        ].head()
    )