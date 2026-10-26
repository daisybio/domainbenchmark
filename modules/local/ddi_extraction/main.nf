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
        //
        // `database_dir` is always a directory: datasetTuple() runs every input
        // through discoverSplits(), which lists it and errors without
        // train/validation/test*.sqlite3. The single-file branch that used to
        // sit here could never be reached.
        def splits = meta.splits.join(' ')
        """
        #!/usr/bin/env bash
        python3 <<'PYEOF'
        import pandas as pd
        import sqlite3
        import os
        import sys

        # ORDER BY on every query: the CSVs are snapshotted and published, so
        # their bytes must not depend on SQLite's scan order. Nothing
        # downstream depends on the order itself -- the ML loader sorts.
        DDI_QUERY = '''
            SELECT d1.pfam_id AS domain_1, d2.pfam_id AS domain_2,
                   NOT ddi.negative AS interaction
            FROM domain_domain_interaction AS ddi
            JOIN domain AS d1 ON ddi.domain_id_a = d1.id
            JOIN domain AS d2 ON ddi.domain_id_b = d2.id
            ORDER BY domain_1, domain_2;
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
            JOIN domain AS d2 ON ddi.domain_id_b = d2.id
            ORDER BY domain_1, domain_2;
        '''

        # `ddi_split_membership` canonicalises each pair by sorting the two
        # instance ids, so instance_id_a does not necessarily belong to
        # domain_id_a. The orientation is resolved from *both* instances'
        # domains; assert_membership() has already failed the task on any row
        # whose instances match the DDI in neither orientation, so the ELSE
        # branch is the verified swap, never a guess. A homodimer satisfies
        # the first branch -- either orientation is correct.
        #
        # instance_1/instance_2 are the raw `instance_id`s. They equal the h5
        # dataset key `COALESCE(instance_id, 'r' || rowid)` the encoders write,
        # because a membership row can only name an instance through a
        # non-NULL instance_id.
        INSTANCE_QUERY = '''
            SELECT d1.pfam_id AS domain_1,
                   d2.pfam_id AS domain_2,
                   CASE WHEN pa.domain_id = ddi.domain_id_a AND pb.domain_id = ddi.domain_id_b
                        THEN m.instance_id_a ELSE m.instance_id_b END AS instance_1,
                   CASE WHEN pa.domain_id = ddi.domain_id_a AND pb.domain_id = ddi.domain_id_b
                        THEN m.instance_id_b ELSE m.instance_id_a END AS instance_2,
                   NOT ddi.negative AS interaction
            FROM ddi_split_membership AS m
            JOIN domain_domain_interaction AS ddi ON ddi.id = m.ddi_id
            JOIN domain_protein_map AS pa ON pa.instance_id = m.instance_id_a
            JOIN domain_protein_map AS pb ON pb.instance_id = m.instance_id_b
            JOIN domain AS d1 ON ddi.domain_id_a = d1.id
            JOIN domain AS d2 ON ddi.domain_id_b = d2.id
            ORDER BY domain_1, domain_2, instance_1, instance_2;
        '''

        NULL_PFAM_QUERY = (
            "SELECT COUNT(*) FROM domain "
            "WHERE pfam_id IS NULL OR TRIM(pfam_id) = ''"
        )

        # `negative` is declared without a type, so it stores whatever class
        # it was handed. SQLite reads any text as 0 in `NOT negative`, so
        # 'True' would come out a positive. Only integer 0/1 is accepted;
        # nothing is guessed.
        BAD_NEGATIVE_QUERY = '''
            SELECT id, negative, typeof(negative) FROM domain_domain_interaction
            WHERE typeof(negative) != 'integer' OR negative NOT IN (0, 1)
            ORDER BY id;
        '''

        # domainsplit's SUBSET_SPLIT_DB copies exactly one (method, split) into
        # each split database. More than one would mix splits into one CSV.
        MEMBERSHIP_SPLITS_QUERY = '''
            SELECT method, split, COUNT(*) FROM ddi_split_membership
            GROUP BY method, split ORDER BY method, split;
        '''

        # INSTANCE_QUERY's inner joins would drop these rows without a word,
        # shrinking the split instead of failing.
        DANGLING_MEMBERSHIP_QUERY = '''
            SELECT m.ddi_id, m.instance_id_a, m.instance_id_b,
                   ddi.id IS NULL, pa.instance_id IS NULL, pb.instance_id IS NULL
            FROM ddi_split_membership AS m
            LEFT JOIN domain_domain_interaction AS ddi ON ddi.id = m.ddi_id
            LEFT JOIN domain_protein_map AS pa ON pa.instance_id = m.instance_id_a
            LEFT JOIN domain_protein_map AS pb ON pb.instance_id = m.instance_id_b
            WHERE ddi.id IS NULL OR pa.instance_id IS NULL OR pb.instance_id IS NULL
            ORDER BY m.ddi_id, m.instance_id_a, m.instance_id_b;
        '''

        # Instance pairs whose domains are not the DDI's two sides in either
        # orientation. The old CASE looked at instance_a alone and emitted
        # these swapped, i.e. under a domain pair neither instance belongs to.
        MISMATCHED_MEMBERSHIP_QUERY = '''
            SELECT m.ddi_id, d1.pfam_id, d2.pfam_id,
                   m.instance_id_a, da.pfam_id, m.instance_id_b, db.pfam_id
            FROM ddi_split_membership AS m
            JOIN domain_domain_interaction AS ddi ON ddi.id = m.ddi_id
            JOIN domain_protein_map AS pa ON pa.instance_id = m.instance_id_a
            JOIN domain_protein_map AS pb ON pb.instance_id = m.instance_id_b
            LEFT JOIN domain AS d1 ON ddi.domain_id_a = d1.id
            LEFT JOIN domain AS d2 ON ddi.domain_id_b = d2.id
            LEFT JOIN domain AS da ON pa.domain_id = da.id
            LEFT JOIN domain AS db ON pb.domain_id = db.id
            WHERE NOT COALESCE(
                (pa.domain_id = ddi.domain_id_a AND pb.domain_id = ddi.domain_id_b)
                OR (pa.domain_id = ddi.domain_id_b AND pb.domain_id = ddi.domain_id_a),
                0)
            ORDER BY m.ddi_id, m.instance_id_a, m.instance_id_b;
        '''

        def has_table(conn, name):
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone() is not None

        def fail(path, problem, rows, explain):
            lines = [f"[ddi_extraction] {path}: {problem}"]
            lines += [f"    {tuple(row)}" for row in rows[:5]]
            if len(rows) > 5:
                lines.append(f"    ... and {len(rows) - 5} more")
            lines.append(explain)
            sys.exit("\\n".join(lines))

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

        def assert_labels(conn, path):
            rows = conn.execute(BAD_NEGATIVE_QUERY).fetchall()
            if rows:
                fail(path,
                     f"{len(rows)} DDI row(s) have a `negative` that is not integer "
                     "0/1 (id, negative, typeof):",
                     rows,
                     "    SQLite reads any text as 0 in `NOT negative`, so 'True' "
                     "would become a positive. Fix the database rather than "
                     "guess the label.")

        def assert_membership(conn, path):
            rows = conn.execute(MEMBERSHIP_SPLITS_QUERY).fetchall()
            if len(rows) > 1:
                fail(path,
                     f"ddi_split_membership holds {len(rows)} (method, split) "
                     "combinations, expected one (method, split, rows):",
                     rows,
                     "    A split database carries exactly one split; more would "
                     "merge several splits' instance pairs into one CSV.")
            rows = conn.execute(DANGLING_MEMBERSHIP_QUERY).fetchall()
            if rows:
                fail(path,
                     f"{len(rows)} membership row(s) reference a DDI or instance "
                     "missing from the database (ddi_id, instance_a, instance_b, "
                     "ddi missing, instance_a missing, instance_b missing):",
                     rows,
                     "    The join would drop these pairs and silently shrink "
                     "the split.")
            rows = conn.execute(MISMATCHED_MEMBERSHIP_QUERY).fetchall()
            if rows:
                fail(path,
                     f"{len(rows)} membership row(s) pair instances whose domains "
                     "are not their DDI's two sides (ddi_id, ddi domain_a, ddi "
                     "domain_b, instance_a, its domain, instance_b, its domain):",
                     rows,
                     "    No orientation of these pairs instantiates their DDI.")

        out_dir = "ddi_out/DDI"
        os.makedirs(out_dir, exist_ok=True)

        for split in "${splits}".split():
            path = f"${database_dir}/{split}.sqlite3"
            if not os.path.isfile(path):
                continue
            with sqlite3.connect(path) as conn:
                assert_pfam_ids(conn, path)
                assert_labels(conn, path)
                pd.read_sql(DDI_QUERY, conn).to_csv(f"{out_dir}/{split}.csv", index=False)
                pd.read_sql(SOURCES_QUERY, conn).to_csv(
                    f"{out_dir}/{split}_sources.csv", index=False
                )
                if has_table(conn, "ddi_split_membership"):
                    assert_membership(conn, path)
                    pd.read_sql(INSTANCE_QUERY, conn).to_csv(
                        f"{out_dir}/{split}_instances.csv", index=False
                    )
        PYEOF
        """
}
