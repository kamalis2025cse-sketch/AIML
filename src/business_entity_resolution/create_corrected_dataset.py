import csv
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE1_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "train_10000"
    / "train_source1_10000.tsv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "train_10000"
    / "train_ground_truth_10000.tsv"
)

SOURCE2_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_source2.tsv"
)

SOURCE3_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_source3.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "train_10000_corrected"
)


# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD SOURCE 1 IDs
# --------------------------------------------------

print("Loading Source 1 IDs...")

source1_ids = set()

with open(
    SOURCE1_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:
        source1_ids.add(
            row["entity_id"]
        )

print(
    "Source 1 records:",
    len(source1_ids)
)


# --------------------------------------------------
# LOAD GROUND TRUTH
# --------------------------------------------------

print("Loading ground truth...")

required_s2_ids = set()
required_s3_ids = set()

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:

        if row["source1_entity_id"] not in source1_ids:
            continue

        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for entity_id in matched_ids.split(","):

            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                required_s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                required_s3_ids.add(entity_id)


print(
    "Required Source 2 IDs:",
    len(required_s2_ids)
)

print(
    "Required Source 3 IDs:",
    len(required_s3_ids)
)


# --------------------------------------------------
# EXTRACT REQUIRED SOURCE 2 RECORDS
# --------------------------------------------------

print("\nScanning original Source 2 file...")
print("This may take some time because the file is large.")

source2_output = (
    OUTPUT_DIR
    / "train_source2_corrected.tsv"
)

found_s2 = 0

with open(
    SOURCE2_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as infile, open(
    source2_output,
    "w",
    encoding="utf-8",
    newline=""
) as outfile:

    header = infile.readline()

    outfile.write(header)

    for line in infile:

        entity_id = line.split(
            "\t",
            1
        )[0]

        if entity_id in required_s2_ids:

            outfile.write(line)

            found_s2 += 1

print(
    "Source 2 records extracted:",
    found_s2
)


# --------------------------------------------------
# EXTRACT REQUIRED SOURCE 3 RECORDS
# --------------------------------------------------

print("\nScanning original Source 3 file...")
print("This may take some time because the file is large.")

source3_output = (
    OUTPUT_DIR
    / "train_source3_corrected.tsv"
)

found_s3 = 0

with open(
    SOURCE3_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as infile, open(
    source3_output,
    "w",
    encoding="utf-8",
    newline=""
) as outfile:

    header = infile.readline()

    outfile.write(header)

    for line in infile:

        entity_id = line.split(
            "\t",
            1
        )[0]

        if entity_id in required_s3_ids:

            outfile.write(line)

            found_s3 += 1

print(
    "Source 3 records extracted:",
    found_s3
)


# --------------------------------------------------
# COPY SOURCE 1
# --------------------------------------------------

source1_output = (
    OUTPUT_DIR
    / "train_source1_corrected.tsv"
)

with open(
    SOURCE1_FILE,
    "r",
    encoding="utf-8"
) as infile, open(
    source1_output,
    "w",
    encoding="utf-8"
) as outfile:

    for line in infile:
        outfile.write(line)


# --------------------------------------------------
# COPY GROUND TRUTH
# --------------------------------------------------

ground_truth_output = (
    OUTPUT_DIR
    / "train_ground_truth_corrected.tsv"
)

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8"
) as infile, open(
    ground_truth_output,
    "w",
    encoding="utf-8"
) as outfile:

    for line in infile:
        outfile.write(line)


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print("\n===================================")
print("CORRECTED DATASET CREATED")
print("===================================")

print(
    "Source 1:",
    source1_output
)

print(
    "Source 2:",
    source2_output
)

print(
    "Source 3:",
    source3_output
)

print(
    "Ground truth:",
    ground_truth_output
)

print("\nDone!")