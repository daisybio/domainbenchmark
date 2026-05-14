#! /usr/bin/env python

import sqlite3
import pandas as pd
from pathlib import Path
import os
import sys


def check_file_existence(file):
    if not os.path.exists(file):
        print(f"File {file} does not exist.")
        sys.exit(1)


def load_ddi(path_to_database: Path) -> pd.DataFrame:
    """
    Load domain-domain interaction data from the SQLite database and save to CSV.
    """
    with sqlite3.connect(path_to_database) as conn:
        ddi_df = pd.read_sql(
            """
            SELECT d1.pfam_id AS domain_1, d2.pfam_id AS domain_2,
                    NOT negative AS interaction, is_evaluation_relevant as eval_relevant
            FROM domain_domain_interaction, domain as d1, domain as d2
            WHERE domain_domain_interaction.domain_id_a = d1.id AND
                domain_domain_interaction.domain_id_b = d2.id;
        """,
            conn,
        )

        ddi_df["eval_relevant"] = ddi_df["eval_relevant"].fillna(0).astype(int)

        domain_a = []
        domain_b = []
        for a, b in zip(ddi_df["domain_1"], ddi_df["domain_2"]):
            sorted_pair = sorted([a, b])
            domain_a.append(sorted_pair[0])
            domain_b.append(sorted_pair[1])
        ddi_df["domain_a"] = domain_a
        ddi_df["domain_b"] = domain_b

        ddi_final = ddi_df[["domain_a", "domain_b", "interaction", "eval_relevant"]]
    return ddi_final


def load_ddi_new(path_to_database: Path) -> pd.DataFrame:
    with sqlite3.connect(path_to_database) as conn:
        ddi_df = pd.read_sql(
            """
                            SELECT domain_id_a AS domain_1, domain_id_b AS domain_2,
                                    NOT negative AS interaction
                            FROM domain_domain_interaction
                            WHERE is_evaluation_relevant;
                    """,
            conn,
        )

    return ddi_df


def load_pd_mapping(path_to_database: Path) -> pd.DataFrame:
    """
    Load protein-domain mapping data from the SQLite database and save to CSV.
    """
    with sqlite3.connect(path_to_database) as conn:
        protein_domain_df = pd.read_sql(
            """
            SELECT uniprot_id, pfam_id
            FROM domain_protein_map, domain, protein
            WHERE domain_id = domain.id AND
                protein_id = protein.id;
        """,
            conn,
        )
    return protein_domain_df


def load_ppi(path_to_database: Path) -> pd.DataFrame:
    """
    Load protein-protein interaction data from the SQLite database and save to CSV.
    """
    with sqlite3.connect(path_to_database) as conn:
        ppi_df = pd.read_sql(
            """
            SELECT p1.uniprot_id AS protein_1, p2.uniprot_id AS protein_2,
                   score
            FROM protein_protein_interaction, protein AS p1, protein AS p2
            WHERE protein_protein_interaction.protein_id_a = p1.id AND
                  protein_protein_interaction.protein_id_b = p2.id;
        """,
            conn,
        )
    return ppi_df


def load_pgo(path_to_database: Path) -> pd.DataFrame:
    """
    Load protein-GO term mapping data from the SQLite database and save to CSV.
    """
    with sqlite3.connect(path_to_database) as conn:
        protein_go_terms = pd.read_sql(
            """
            SELECT uniprot_id, go_accession
            FROM protein_go_terms, protein
            WHERE protein_id = protein.id;
            """,
            conn,
        )
        # protein_go_terms.to_csv(output_folder / "PGO.csv", index=False)
    return protein_go_terms


def create_debug_dataset(ddi_df, pd_mapping, ppi_df) -> tuple:
    """
    Create a small debug dataset by sampling a subset of the data.
    """

    # Number of negative samples in DDI
    n_negative_samples = ddi_df[ddi_df["interaction"] == 0].shape[0]
    debug_ddi = pd.concat(
        [
            ddi_df[ddi_df["interaction"] == 1].sample(
                n=n_negative_samples, random_state=42
            ),
            ddi_df[ddi_df["interaction"] == 0].sample(
                n=n_negative_samples // 10, random_state=42
            ),
        ]
    ).reset_index(drop=True)
    involved_domains = set(debug_ddi["domain_a"]).union(set(debug_ddi["domain_b"]))
    debug_pd_mapping = pd_mapping[
        pd_mapping["pfam_id"].isin(involved_domains)
    ].reset_index(drop=True)

    involved_proteins = set(debug_pd_mapping["uniprot_id"])
    debug_ppi = ppi_df[
        ppi_df["protein_1"].isin(involved_proteins)
        & ppi_df["protein_2"].isin(involved_proteins)
    ].reset_index(drop=True)

    # Repeat occurring DDIS by repeating rows with the same amount as in the original dataset
    original_ddi_counts = (
        ddi_df.groupby(["domain_a", "domain_b"]).size().reset_index(name="counts")
    )
    debug_ddi = debug_ddi.merge(
        original_ddi_counts, on=["domain_a", "domain_b"], how="left"
    )
    debug_ddi = debug_ddi.loc[debug_ddi.index.repeat(debug_ddi["counts"])].reset_index(
        drop=True
    )
    debug_ddi = debug_ddi.drop(columns=["counts"])

    return debug_ddi, debug_pd_mapping, debug_ppi


def create_debug_dataset_kgiddi(
    ddi_df, pd_mapping, ppi_df, pgo_df, go_graph_nx
) -> tuple:
    """
    Create a small debug dataset by sampling a subset of the data, ensuring all proteins have GO terms in the GO graph.
    """

    # 1. Filter PGO for GO terms present in the GO graph
    valid_go_terms = set(go_graph_nx.nodes())
    pgo_valid = pgo_df[pgo_df["go_accession"].isin(valid_go_terms)].reset_index(
        drop=True
    )

    # 2. Find proteins with at least one valid GO term
    valid_proteins = set(pgo_valid["uniprot_id"])

    # 3. Restrict PD mapping to these proteins
    pd_mapping_valid = pd_mapping[
        pd_mapping["uniprot_id"].isin(valid_proteins)
    ].reset_index(drop=True)

    # 4. Restrict PPI to these proteins
    ppi_valid = ppi_df[
        ppi_df["protein_1"].isin(valid_proteins)
        & ppi_df["protein_2"].isin(valid_proteins)
    ].reset_index(drop=True)

    # 5. Get all domains present in the valid PD mapping
    valid_domains = set(pd_mapping_valid["pfam_id"])

    # 6. Restrict DDI to these domains
    ddi_valid = ddi_df[
        ddi_df["domain_a"].isin(valid_domains) & ddi_df["domain_b"].isin(valid_domains)
    ].reset_index(drop=True)

    # 7. Sample DDIs for debug set
    n_negative_samples = ddi_valid[ddi_valid["interaction"] == 0].shape[0]
    # As the number of samples is now likely smaller I multiply by 2
    n_negative_samples *= 2

    debug_ddi = pd.concat(
        [
            ddi_valid[ddi_valid["interaction"] == 1].sample(
                n=min(
                    n_negative_samples,
                    ddi_valid[ddi_valid["interaction"] == 1].shape[0],
                ),
                random_state=42,
            ),
            ddi_valid[ddi_valid["interaction"] == 0].sample(
                n=max(1, n_negative_samples // 10), random_state=42
            ),
        ]
    ).reset_index(drop=True)

    # 8. Get domains in debug_ddi
    debug_domains = set(debug_ddi["domain_a"]).union(set(debug_ddi["domain_b"]))

    # 9. Restrict PD mapping to these domains
    debug_pd_mapping = pd_mapping_valid[
        pd_mapping_valid["pfam_id"].isin(debug_domains)
    ].reset_index(drop=True)

    # 10. Restrict PPI to proteins in debug_pd_mapping
    debug_proteins = set(debug_pd_mapping["uniprot_id"])
    debug_ppi = ppi_valid[
        ppi_valid["protein_1"].isin(debug_proteins)
        & ppi_valid["protein_2"].isin(debug_proteins)
    ].reset_index(drop=True)

    # 11. Restrict PGO to proteins in debug_pd_mapping
    debug_pgo_df = pgo_valid[pgo_valid["uniprot_id"].isin(debug_proteins)].reset_index(
        drop=True
    )

    return debug_ddi, debug_pd_mapping, debug_ppi, debug_pgo_df


if __name__ == "__main__":
    # Testing
    # Check the two load ddi functions
    database_path = Path(
        "/nfs/data/CoBiNet_Masterpraktikum/databases/random_ddi/test.sqlite3"
    )
    ddi1 = load_ddi(database_path)
    ddi2 = load_ddi_new(database_path)
    print(f"DDI1 shape: {ddi1.shape}, DDI2 shape: {ddi2.shape}")
    # Write both to tmp csv files
    ddi1.to_csv("tmp/tmp_ddi1.csv", index=False)
    ddi2.to_csv("tmp/tmp_ddi2.csv", index=False)
