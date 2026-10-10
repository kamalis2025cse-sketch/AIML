import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from rapidfuzz.fuzz import ratio, token_set_ratio
import pickle


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "dataset" / "train_10000_corrected"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_PATH = OUTPUT_DIR / "matching_model_10000.pkl"
THRESHOLD_PATH = OUTPUT_DIR / "matching_threshold_10000.txt"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Business Entity Resolution",
    page_icon="🏢",
    layout="wide"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKD", text
    ).encode(
        "ascii", "ignore"
    ).decode("ascii")

    text = text.lower()

    text = re.sub(r"[^a-z0-9]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    source1 = pd.read_csv(
        DATA_DIR / "train_source1_corrected.tsv",
        sep="\t",
        dtype=str
    ).fillna("")

    source2 = pd.read_csv(
        DATA_DIR / "train_source2_corrected.tsv",
        sep="\t",
        dtype=str
    ).fillna("")

    source3 = pd.read_csv(
        DATA_DIR / "train_source3_corrected.tsv",
        sep="\t",
        dtype=str
    ).fillna("")

    return source1, source2, source3


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    return model, threshold


# ============================================================
# PREPARE TARGET DATA
# ============================================================

@st.cache_data
def prepare_targets(source2, source3):

    s2 = source2.copy()
    s3 = source3.copy()

    s2["target_source"] = "S2"
    s3["target_source"] = "S3"

    targets = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    targets["name_norm"] = targets[
        "business_name"
    ].apply(normalize_text)

    targets["address_norm"] = targets[
        "business_address"
    ].apply(normalize_text)

    targets["country_norm"] = targets[
        "country"
    ].apply(normalize_text)

    return targets


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(
    input_name,
    input_address,
    input_country,
    target
):

    name = normalize_text(input_name)
    address = normalize_text(input_address)
    country = normalize_text(input_country)

    target_name = target["name_norm"]
    target_address = target["address_norm"]
    target_country = target["country_norm"]

    name_ratio = ratio(
        name,
        target_name
    )

    name_token_ratio = token_set_ratio(
        name,
        target_name
    )

    address_ratio = ratio(
        address,
        target_address
    )

    address_token_ratio = token_set_ratio(
        address,
        target_address
    )

    exact_name = int(
        name == target_name and name != ""
    )

    exact_address = int(
        address == target_address and address != ""
    )

    country_match = int(
        country == target_country and country != ""
    )

    return [
        name_ratio,
        name_token_ratio,
        address_ratio,
        address_token_ratio,
        exact_name,
        exact_address,
        country_match
    ]


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    input_name,
    input_address,
    input_country,
    targets,
    top_k=30
):

    name = normalize_text(input_name)
    address = normalize_text(input_address)
    country = normalize_text(input_country)

    scores = []

    for idx, row in targets.iterrows():

        # Country is used as a useful blocking signal
        country_score = 100

        if country and row["country_norm"]:
            if country != row["country_norm"]:
                country_score = 0

        name_score = 0
        address_score = 0

        if name:
            name_score = max(
                ratio(name, row["name_norm"]),
                token_set_ratio(name, row["name_norm"])
            )

        if address:
            address_score = max(
                ratio(address, row["address_norm"]),
                token_set_ratio(address, row["address_norm"])
            )

        # Combined candidate score
        combined = (
            0.60 * name_score
            + 0.30 * address_score
            + 0.10 * country_score
        )

        scores.append((idx, combined))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    selected = [
        idx for idx, score in scores[:top_k]
    ]

    return targets.loc[selected].copy()


# ============================================================
# PREDICTION
# ============================================================

def find_matches(
    input_name,
    input_address,
    input_country,
    candidates,
    model,
    threshold
):

    feature_rows = []

    for _, row in candidates.iterrows():

        features = create_features(
            input_name,
            input_address,
            input_country,
            row
        )

        feature_rows.append(features)

    X = np.array(feature_rows)

    probabilities = model.predict_proba(X)[:, 1]

    results = candidates.copy()

    results["match_probability"] = probabilities

    results["name_similarity"] = [
        ratio(
            normalize_text(input_name),
            x
        )
        for x in results["name_norm"]
    ]

    results["address_similarity"] = [
        ratio(
            normalize_text(input_address),
            x
        )
        for x in results["address_norm"]
    ]

    results["is_match"] = (
        results["match_probability"] >= threshold
    )

    results = results.sort_values(
        "match_probability",
        ascending=False
    )

    return results


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    source1, source2, source3 = load_data()

    targets = prepare_targets(
        source2,
        source3
    )

    model, threshold = load_model()

    data_loaded = True

except Exception as e:

    data_loaded = False
    load_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏢 Entity Resolution")

    st.markdown(
        """
        ### AI/ML Pipeline

        **1. Preprocessing**

        Normalize business names and addresses.

        **2. Candidate Generation**

        Find similar business records.

        **3. Feature Extraction**

        Calculate name, address and country similarities.

        **4. ML Matching**

        Predict whether records refer to the same business.

        **5. Entity Resolution**

        Display the most likely matching entity.
        """
    )

    st.divider()

    st.subheader("Development Metrics")

    st.metric(
        "Development Records",
        "10,000"
    )

    st.metric(
        "Candidate Recall",
        "99.78%"
    )

    st.metric(
        "Validation F0.5",
        "0.9425"
    )

    st.divider()

    st.caption(
        "AI Business Entity Resolution Challenge 2026"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🏢 AI Business Entity Resolution"
)

st.markdown(
    """
    ### Match business records across different data sources

    This system identifies whether business records from different
    databases refer to the same real-world business entity.
    """
)


if not data_loaded:

    st.error(
        "Unable to load the project model or dataset."
    )

    st.code(
        load_error
    )

    st.stop()


# ============================================================
# PROJECT STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Source 1 Records",
        f"{len(source1):,}"
    )

with col2:
    st.metric(
        "Source 2 Records",
        f"{len(source2):,}"
    )

with col3:
    st.metric(
        "Source 3 Records",
        f"{len(source3):,}"
    )

with col4:
    st.metric(
        "Model Threshold",
        f"{threshold:.2f}"
    )


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("🔍 Enter Business Information")

col1, col2 = st.columns(2)

with col1:

    business_name = st.text_input(
        "Business Name",
        placeholder="Example: ABC Private Limited"
    )

    business_address = st.text_area(
        "Business Address",
        placeholder="Example: 12 Main Road, Chennai"
    )

with col2:

    country = st.text_input(
        "Country",
        placeholder="Example: India"
    )

    st.info(
        "Enter as much information as possible for better matching."
    )


find_button = st.button(
    "🔎 Find Matching Business",
    type="primary",
    use_container_width=True
)


# ============================================================
# MATCHING
# ============================================================

if find_button:

    if not business_name.strip():

        st.warning(
            "Please enter a business name."
        )

    else:

        with st.spinner(
            "Generating candidates and running the ML model..."
        ):

            candidates = generate_candidates(
                business_name,
                business_address,
                country,
                targets,
                top_k=30
            )

            results = find_matches(
                business_name,
                business_address,
                country,
                candidates,
                model,
                threshold
            )

        st.divider()

        st.subheader(
            "🎯 Matching Results"
        )

        matches = results[
            results["is_match"]
        ]

        if len(matches) > 0:

            best = matches.iloc[0]

            st.success(
                "✓ Matching entity found"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Match Probability",
                    f"{best['match_probability']:.2%}"
                )

            with col2:

                st.metric(
                    "Name Similarity",
                    f"{best['name_similarity']:.1f}%"
                )

            with col3:

                st.metric(
                    "Address Similarity",
                    f"{best['address_similarity']:.1f}%"
                )

            st.markdown("### Matched Entity")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.write(
                    "**Entity ID:**",
                    best["entity_id"]
                )

                st.write(
                    "**Source:**",
                    best["target_source"]
                )

                st.write(
                    "**Business Name:**",
                    best["business_name"]
                )

            with result_col2:

                st.write(
                    "**Address:**",
                    best["business_address"]
                )

                st.write(
                    "**Country:**",
                    best["country"]
                )

            st.divider()

            st.subheader(
                "Other Candidate Matches"
            )

            display_columns = [
                "entity_id",
                "target_source",
                "business_name",
                "country",
                "match_probability",
                "name_similarity",
                "address_similarity"
            ]

            display = results[
                display_columns
            ].head(10).copy()

            display["match_probability"] = (
                display["match_probability"] * 100
            ).round(2)

            display["name_similarity"] = (
                display["name_similarity"]
            ).round(2)

            display["address_similarity"] = (
                display["address_similarity"]
            ).round(2)

            display.columns = [
                "Entity ID",
                "Source",
                "Business Name",
                "Country",
                "ML Probability (%)",
                "Name Similarity (%)",
                "Address Similarity (%)"
            ]

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No sufficiently confident matching entity was found."
            )

            st.info(
                "Try entering a more complete business name or address."
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("⚙️ How the System Works")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown(
        """
        ### 1️⃣
        **Preprocessing**

        Normalize names and addresses to reduce differences in formatting.
        """
    )

with step2:

    st.markdown(
        """
        ### 2️⃣
        **Candidate Generation**

        Retrieve the most similar possible business records.
        """
    )

with step3:

    st.markdown(
        """
        ### 3️⃣
        **Feature Extraction**

        Compare names, addresses and country information.
        """
    )

with step4:

    st.markdown(
        """
        ### 4️⃣
        **ML Matching**

        Predict whether candidate records represent the same entity.
        """
    )


st.divider()

st.caption(
    "AI Business Entity Resolution • Python • Pandas • "
    "Scikit-learn • RapidFuzz • Streamlit"
)