process DDI_EXTRACTION {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database_dir)

    output:
        // NOT `${meta.id}/DDI`: the input database directory is staged under its
        // own basename, which for a directory input *is* `${meta.id}`, so writing
        // there followed the stage symlink and dumped the CSVs straight into the
        // caller's domainsplit output (and into the test fixture, whose stale
        // copies are still tracked). `ddi_out/` cannot collide with a staged input.
        tuple val(meta), path("ddi_out/DDI"), emit: ddi

    script:
        // Every DDI row in a split database *is* that split: domainsplit's
        // SUBSET_SPLIT_DB copies only the rows `ddi_split_membership` assigns
        // to (method, split). There is no `is_evaluation_relevant` column to
        // filter on any more.
        //
        // Alongside the domain-pair CSV each split also gets an
        // `<split>_instances.csv` listing the concrete domain-instance pairs
        // the splitter assigned, which is what the ML loader instantiates
        // instead of a full cross-product over every instance of each family.
        //
        // ---------------------------------------------------------------------
        // Every query below joins `domain` and reports `pfam_id`, never
        // `domain.id`.
        //
        // `domain.id` is a **per-run surrogate integer**: domainsplit's
        // SUBSET_SPLIT_DB copies it verbatim and PRUNE_UNREPRESENTED_DDIS
        // deletes without renumbering, so the same integer names a different
        // domain in the next run. Anything keyed on it therefore cannot be
        // compared between runs -- and the report is exactly a cross-run
        // artefact. `pfam_id` is stable, and `domain` is UNIQUE(pfam_id) with
        // `id INTEGER PRIMARY KEY`, so the two are in bijection *within* a
        // database: swapping the key loses nothing.
        //
        // This also closes a silent join failure. The graph models never went
        // through these CSVs -- `bin/load_data_gm.py` reads the database
        // directly and has always joined `domain` for `pfam_id` -- so their
        // `predictions_*.parquet` carried Pfam accessions while
        // `<split>_sources.csv` carried surrogate integers. `eval_one.py` joins
        // the two on the domain pair, so *every* graph-model pair missed and
        // landed in the `unknown` per-source bucket. The feature h5 files and
        // the ML loader are keyed on `pfam_id` for the same reason.
        // ---------------------------------------------------------------------
        def splits = (meta.splits ?: ['test']).join(' ')

        // Single-file db inputs only support a 'test' split.
        // Directory inputs carry train/validation/test* sqlite splits.
        if (database_dir.isFile()) {
            """
            #!/usr/bin/env bash
            python3 <<'PYEOF'
            import pandas as pd
            import sqlite3
            import os
            import sys

            DDI_QUERY = '''
                SELECT d1.pfam_id AS domain_1, d2.pfam_id AS domain_2,
                       NOT ddi.negative AS interaction
                FROM domain_domain_interaction AS ddi
                JOIN domain AS d1 ON ddi.domain_id_a = d1.id
                JOIN domain AS d2 ON ddi.domain_id_b = d2.id;
            '''

            SOURCES_QUERY = '''
                SELECT d1.pfam_id AS domain_1, d2.pfam_id AS domain_2,
                       NOT ddi.negative AS interaction, ddi.source AS source
                FROM domain_domain_interaction AS ddi
                JOIN domain AS d1 ON ddi.domain_id_a = d1.id
                JOIN domain AS d2 ON ddi.domain_id_b = d2.id;
            '''

            with sqlite3.connect('${database_dir}') as conn:
                # pfam_id is the join key for the whole downstream report, so a
                # NULL one would quietly merge unrelated domains into a single
                # row instead of failing. UNIQUE(pfam_id) permits exactly one.
                n_null = conn.execute(
                    "SELECT COUNT(*) FROM domain "
                    "WHERE pfam_id IS NULL OR TRIM(pfam_id) = ''"
                ).fetchone()[0]
                if n_null:
                    sys.exit(
                        f"[ddi_extraction] ${database_dir}: {n_null} domain row(s) "
                        "have no pfam_id. pfam_id is the key every CSV, feature "
                        "h5 and prediction file is written under -- a NULL would "
                        "collapse unrelated domains into one row."
                    )
                ddi_df = pd.read_sql(DDI_QUERY, conn)
                source_df = pd.read_sql(SOURCES_QUERY, conn)

            os.makedirs("ddi_out/DDI/", exist_ok=True)
            ddi_df.to_csv("ddi_out/DDI/test.csv", index=False)
            source_df.to_csv("ddi_out/DDI/test_sources.csv", index=False)
            PYEOF
            """
        } else {
            """
            #!/usr/bin/env bash
            python3 <<'PYEOF'
            import pandas as pd
            import sqlite3
            import os
            import sys

            DDI_QUERY = '''
                SELECT d1.pfam_id AS domain_1, d2.pfam_id AS domain_2,
                       NOT ddi.negative AS interaction
                FROM domain_domain_interaction AS ddi
                JOIN domain AS d1 ON ddi.domain_id_a = d1.id
                JOIN domain AS d2 ON ddi.domain_id_b = d2.id;
            '''

            # Same rows plus the provenance list. `source` is a comma-joined list
            # of every source that contributed the pair (domainsplit keys
            # `domain_domain_interaction` UNIQUE on the pair), so a DDI with
            # "single_domain_ppi,PPIDM,PPIDM_Gold" counts towards all three
            # per-source performance rows in the report.
            SOURCES_QUERY = '''
                SELECT d1.pfam_id AS domain_1, d2.pfam_id AS domain_2,
                       NOT ddi.negative AS interaction, ddi.source AS source
                FROM domain_domain_interaction AS ddi
                JOIN domain AS d1 ON ddi.domain_id_a = d1.id
                JOIN domain AS d2 ON ddi.domain_id_b = d2.id;
            '''

            # `ddi_split_membership` canonicalises each pair by sorting the two
            # instance ids, so instance_id_a does not necessarily belong to
            # domain_id_a. Resolve the side through domain_protein_map.
            # Homodimers satisfy both branches -- either orientation is correct.
            INSTANCE_QUERY = '''
                SELECT d1.pfam_id AS domain_1,
                       d2.pfam_id AS domain_2,
                       CASE WHEN pa.domain_id = ddi.domain_id_a
                            THEN m.instance_id_a ELSE m.instance_id_b END AS instance_1,
                       CASE WHEN pa.domain_id = ddi.domain_id_a
                            THEN m.instance_id_b ELSE m.instance_id_a END AS instance_2,
                       NOT ddi.negative AS interaction
                FROM ddi_split_membership AS m
                JOIN domain_domain_interaction AS ddi ON ddi.id = m.ddi_id
                JOIN domain_protein_map AS pa ON pa.instance_id = m.instance_id_a
                JOIN domain AS d1 ON ddi.domain_id_a = d1.id
                JOIN domain AS d2 ON ddi.domain_id_b = d2.id;
            '''

            NULL_PFAM_QUERY = (
                "SELECT COUNT(*) FROM domain "
                "WHERE pfam_id IS NULL OR TRIM(pfam_id) = ''"
            )

            def has_table(conn, name):
                return conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
                ).fetchone() is not None

            def assert_pfam_ids(conn, path):
                # pfam_id is the join key for the whole downstream report -- the
                # CSVs here, the feature h5 group names, the prediction files and
                # eval_one.py's per-source join all use it. A NULL would quietly
                # merge unrelated domains into one row rather than fail, so check
                # it once per split instead. UNIQUE(pfam_id) permits exactly one.
                n_null = conn.execute(NULL_PFAM_QUERY).fetchone()[0]
                if n_null:
                    sys.exit(
                        f"[ddi_extraction] {path}: {n_null} domain row(s) have no "
                        "pfam_id. pfam_id is the key every CSV, feature h5 and "
                        "prediction file is written under -- a NULL would collapse "
                        "unrelated domains into one row."
                    )

            out_dir = "ddi_out/DDI"
            os.makedirs(out_dir, exist_ok=True)

            for split in "${splits}".split():
                path = f"${database_dir}/{split}.sqlite3"
                if not os.path.isfile(path):
                    continue
                with sqlite3.connect(path) as conn:
                    assert_pfam_ids(conn, path)
                    pd.read_sql(DDI_QUERY, conn).to_csv(f"{out_dir}/{split}.csv", index=False)
                    pd.read_sql(SOURCES_QUERY, conn).to_csv(
                        f"{out_dir}/{split}_sources.csv", index=False
                    )
                    if has_table(conn, "ddi_split_membership"):
                        pd.read_sql(INSTANCE_QUERY, conn).to_csv(
                            f"{out_dir}/{split}_instances.csv", index=False
                        )
            PYEOF
            """
        }
}
