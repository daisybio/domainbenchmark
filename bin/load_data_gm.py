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


