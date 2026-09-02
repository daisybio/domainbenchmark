#! /usr/bin/env python

import sqlite3
import pandas as pd
from pathlib import Path
import os
import sys

# STRING's "medium confidence" band, which is also STRING's own default. Used
# only when neither the pipeline (`params.ppi_score_cutoff`) nor a model JSON
# names a value -- both graph models share it so a standalone invocation can
# never silently score a different interactome than a pipeline run.
#
# Not 900 ("highest confidence"): a split database carries only its own
# proteins' interactome, so that band can be a few percent of it. See the note
# in kgiddi.py's preprocessing for the counts that forced this.
DEFAULT_PPI_SCORE_CUTOFF = 400


def canonical_pair(domain_a, domain_b):
    """`(lo, hi)` for a domain pair, so `domain_a` is always the smaller id.

    A DDI is undirected, so the two orientations name the same thing and the
    published `predictions_*.parquet` should not depend on which one a model's
    internal iteration happened to produce. Every writer canonicalises here.

    Plain string comparison is the right order for a Pfam accession: they are
    `PF` plus a zero-padded five-digit number, so lexicographic and numeric order
    coincide (`PF00099` < `PF00100`). It would not hold for a bare integer id --
    another reason the surrogate `domain.id` is not what anything is keyed on.
    """
    a, b = str(domain_a), str(domain_b)
    return (a, b) if a <= b else (b, a)


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
                    NOT negative AS interaction
            FROM domain_domain_interaction, domain as d1, domain as d2
            WHERE domain_domain_interaction.domain_id_a = d1.id AND
                domain_domain_interaction.domain_id_b = d2.id;
        """,
            conn,
        )

        # domainsplit's SUBSET_SPLIT_DB copies only the rows this split owns,
        # so every DDI row in the file is evaluation-relevant by construction.
        # The old `is_evaluation_relevant` column no longer exists; the flag is
        # kept so downstream consumers (ddi_dict tuples) keep their shape.
        ddi_df["eval_relevant"] = 1

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

    `score` comes back as a **numeric** column or not at all. It was not always
    numeric in the database: domainsplit used to declare
    `protein_protein_interaction.score` with no type (`score,` in
    INIT_DOMAINSPLIT_DB), giving the column BLOB affinity so SQLite stored
    whatever class it was handed -- and INSERT_PPI handed it `parts[2]`, the raw
    string it split out of STRING's links file. pandas then read an `object`
    column and both graph models' confidence filter died on
    `'>=' not supported between instances of 'str' and 'int'`.

    domainsplit types the column REAL and parses the score now, so a fresh
    database needs nothing from this function. It stays because a database
    already on disk does not get retyped, and because the check belongs
    somewhere every reader of this column shares: they all compare it to
    `params.ppi_score_cutoff`.
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

    raw_score = ppi_df["score"]
    score = pd.to_numeric(raw_score, errors="coerce")
    # Any row that does not parse is fatal, not dropped. Such a row cannot pass
    # a confidence cutoff, so tolerating it means silently scoring a smaller
    # interactome than the caller asked for -- and one unparseable score means
    # the column holds something other than a number, which is a property of the
    # whole database, not of that row. A NULL is the same thing: domainsplit's
    # INSERT_PPI has no path that writes one, so it is a corrupt or foreign
    # database rather than a missing measurement.
    bad = raw_score[score.isna()]
    if len(bad):
        raise ValueError(
            f"{path_to_database}: {len(bad)} of {len(ppi_df)} "
            "protein_protein_interaction rows have a score that is not a number "
            f"(examples: {sorted(set(bad.astype(str)))[:5]}), so no confidence "
            "cutoff can be applied to this interactome. Check "
            "`SELECT typeof(score), count(*) FROM protein_protein_interaction "
            "GROUP BY 1` -- the column was typeless in older domainsplit "
            "schemas, so it holds whatever class the loader inserted."
        )
    ppi_df["score"] = score
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


