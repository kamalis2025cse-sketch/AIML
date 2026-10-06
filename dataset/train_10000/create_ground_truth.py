import csv

source1_file = "train_source1_10000.tsv"
ground_truth_file = "..\\train\\train_ground_truth.tsv"
output_file = "train_ground_truth_10000.tsv"

# Read the 10,000 Source 1 IDs
source1_ids = set()

with open(source1_file, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        source1_ids.add(row["entity_id"])

print("Source 1 IDs loaded:", len(source1_ids))

# Read the large ground-truth file line by line
found = 0

with open(ground_truth_file, "r", encoding="utf-8", newline="") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:

    header = infile.readline()
    outfile.write(header)

    for line in infile:
        source1_id = line.split("\t", 1)[0]

        if source1_id in source1_ids:
            outfile.write(line)
            found += 1

print("Ground-truth records found:", found)
print("Created:", output_file)