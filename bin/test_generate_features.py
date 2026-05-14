#!/usr/bin/env python3

import pandas as pd
import argparse

# PARSE USER INFORMATION PASSED AT RUNTIME
parser = argparse.ArgumentParser(
    description="Feature generation for peptide-based sequences. Output file contains features for the sequence as well as the sequence itself (in the first column).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-i",
    "--input",
    action="store_true",
    help="location of file containing peptide sequences to generate features for",
)
parser.add_argument(
    "-o",
    "--output",
    action="store_true",
    help="location and name of output file containing feature data",
)
args = parser.parse_known_args()

# LINK PARAMETERS TO ARGUMENTS
read_in = args[1][0]
read_out = args[1][1]


# GENERATE PROTDCAL FEATURES FOR SEQUENCE
def protdcal_features(sequence, protdcal):
    # METHOD FOR GENERATING PROTDCAL MEAN VALUES FOR EACH SEQUENCE
    # FIRST: GENERATE PROTDCAL VALUES
    slist = list(sequence.upper())  # split sequence up into list
    # Go through sequence to get protdcal value
    t1 = []
    values = []
    for i in slist:
        t1.append(protdcal.loc[i].tolist())
    t2 = list(map(lambda *x: sum(x), *t1))  # add up values
    for t in t2:
        t = t / len(slist)  # get average (mean) of summed values
        values.append(t)
    headers = protdcal.columns.tolist()  # include headers

    return values, headers


# TRAINING FEATURE GENERATION: GENERATE OTHER FEATURES + FORMAT NICELY
# Import protdcal file
protdcal = pd.read_csv(
    "/home/t/thomasc/MaPra/Pipeline/dummy_pipeline/scripts/protdcal_table.csv",
    index_col=0,
)

# Import user file
file_in = str(read_in)
user_feat = pd.read_csv(read_in)

# Define seq_col input by user as a variable, ensure it's in string format for use with dataframe
seq_col = "domain_sequence"


# Check for invalid information in sequences that we can't handle - remove + output as error file
alphabet = "ARNDCQEGHILKMFPSTWYVX"  # 20 essential amino acids + X

print(user_feat)

sequences = user_feat[seq_col]

# NOW GET INTO FEATURE GENERATION!

# Create df for protdcal results to go into
print(sequences[0])
v, h = protdcal_features(sequences[0], protdcal)
features = pd.DataFrame(columns=h)
features.loc[len(features)] = v

i = 1

# Go through rest of sequences to generate protdcal features set
while i < len(sequences):
    ts = sequences[i]
    value, header = protdcal_features(ts, protdcal)
    features.loc[len(features)] = value
    i += 1

print(features)
