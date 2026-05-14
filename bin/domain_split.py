import numpy as np
import pandas as pd
import random
import sqlite3
from functools import partial
from tqdm import tqdm

ANNEALING_STEPS = 1

annealing_step_fractions = [
    (i + 1) / (ANNEALING_STEPS + 1) for i in range(ANNEALING_STEPS)
]

split_fractions = {"train": 0.8, "val": 0.1, "test": 0.1}

conn = sqlite3.connect("cobinet.sqlite3")
# Load the protein-domain mapping into a DataFrame (not used in the current implementation, but may be useful for future extensions)
# pd_mapping = pd.read_sql("""
#    SELECT domain_id, protein_id FROM domain_protein_map
#    WHERE domain_id IS NOT NULL AND protein_id IS NOT NULL LIMIT 20;
# """, conn)

ddi = pd.read_sql(
    """
    SELECT id AS ddi_id, domain_id_a, domain_id_b FROM domain_domain_interaction;
""",
    conn,
)

# cluster sequences are defined as <domain_id>-<protein_id>
cluster_df = pd.read_csv(
    "domain_clusters.tsv", sep="\t", header=None, names=["centroid", "member"]
)

# strip the protein_id from the sequence ids in the cluster dataframe
cluster_df = cluster_df.map(lambda x: x.split("-")[0]).map(int)

# Aggregate member domains for each centroid
domain_clusters = cluster_df.groupby("centroid")["member"].apply(set).tolist()

split_domains = {split_name: set() for split_name in split_fractions.keys()}
split_ddis = {split_name: set() for split_name in split_fractions.keys()}


def get_new_ddis_for_domains(domains_in_split, new_domains):
    prefiltered_ddi = ddi[
        (ddi["domain_id_a"].isin(new_domains)) | (ddi["domain_id_b"].isin(new_domains))
    ]
    new_ddis = {
        ddi_id
        for ddi_id, domain_a, domain_b in prefiltered_ddi.itertuples(index=False)
        if (domain_a in domains_in_split or domain_b in domains_in_split)
    }
    return new_ddis


def get_split_ddis(split_name, new_domains=None):
    if new_domains is None:
        # compute from scratch (more expensive)
        return {
            ddi_id
            for ddi_id, domain_a, domain_b in ddi.itertuples(index=False)
            if domain_a in split_domains[split_name]
            and domain_b in split_domains[split_name]
        }
    else:
        # compute incrementally (more efficient)
        new_ddis = get_new_ddis_for_domains(split_domains[split_name], new_domains)
        return split_ddis[split_name].union(new_ddis)


def cluster_rank_function(split_name, cluster_domains):
    # Calculate the total number of interactions that would be added to the split if this cluster were added
    # no need to normalize by split size since we are comparing clusters for the same split
    # return len(get_split_ddis(split_name, new_domains=cluster_domains))

    return len(get_new_ddis_for_domains(split_domains[split_name], cluster_domains))


def split(list, fraction):
    split_index = int(len(list) * fraction)
    return list[:split_index], list[split_index:]


# as initialization, assign clusters to splits according to the annealing fraction, then iteratively assign remaining clusters to the split that would gain the most interactions (normalized by split size)
for (split_name, fraction), clusters in zip(
    split_fractions.items(), np.array_split(domain_clusters, len(split_fractions))
):
    split_domains[split_name] = set.union(*clusters)

for annealing_step, annealing_fraction in enumerate(annealing_step_fractions):
    print(
        f"Starting annealing step {annealing_step + 1}/{ANNEALING_STEPS} with fraction {annealing_fraction:.2f}"
    )
    print("Initial DDI counts: ")
    for split_name in split_fractions.keys():
        split_ddis[split_name] = get_split_ddis(split_name)
        print(f"\t{split_name}: {len(split_ddis[split_name])} interactions")

    random.shuffle(domain_clusters)
    clusters_to_keep, clusters_queue = split(domain_clusters, annealing_fraction)
    domains_to_keep = set.union(*clusters_to_keep)

    # Remove domains from splits that are not in the clusters to keep
    for split_name in split_fractions.keys():
        split_domains[split_name] = split_domains[split_name].intersection(
            domains_to_keep
        )
        split_ddis[split_name] = get_split_ddis(split_name)

    print("DDI counts after initialization:")
    for split_name in split_fractions.keys():
        print(f"\t{split_name}: {len(split_ddis[split_name])} interactions")

    tqdm_ = tqdm(
        total=len(domain_clusters),
        initial=len(clusters_to_keep),
        desc=f"Annealing step {annealing_step + 1}/{ANNEALING_STEPS}",
    )
    while clusters_queue:
        # take the split that has the least number of interactions (normalized by split size)
        current_split_name = min(
            split_fractions.keys(),
            key=lambda split_name: (
                len(split_ddis[split_name]) / split_fractions[split_name]
            ),
        )

        # rank the clusters by the number of interactions they would add to the split
        cluster = max(
            clusters_queue, key=partial(cluster_rank_function, current_split_name)
        )
        clusters_queue.remove(cluster)

        split_domains[current_split_name].update(cluster)
        split_ddis[current_split_name] = get_split_ddis(
            current_split_name, new_domains=cluster
        )

        tqdm_.update(1)

print("Final DDI counts:")
for split_name in split_fractions.keys():
    print(f"\t{split_name}: {len(split_ddis[split_name])} interactions")

    # write out the domain ids in the split to a file
    with open(split_name, "w") as f:
        f.write("ddi_id\n")
        for ddi_id in split_ddis[split_name]:
            f.write(ddi_id)
            f.write("\n")
