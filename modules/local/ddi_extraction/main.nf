process DDI_EXTRACTION {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database_dir)

    output:
        // NOT `${meta.id}/DDI`: the input database directory is staged under its
        // own basename, which for a directory input *is* `${meta.id}`, so writing
        // there followed the stage symlink and dumped the CSVs straight into the
        // caller's domainsplit output (and into the test fixture, whose stale
        // copies are still tracked). `ddi_out/` cannot collide with a staged input.
        tuple val(meta), path("ddi_out/DDI"), emit: ddi
        path "versions.yml",                  emit: versions

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

            with sqlite3.connect('${database_dir}') as conn:
                ddi_df = pd.read_sql('''
                        SELECT domain_id_a AS domain_1, domain_id_b AS domain_2,
                                NOT negative AS interaction
                        FROM domain_domain_interaction;
                ''', conn)
                source_df = pd.read_sql('''
                        SELECT domain_id_a AS domain_1, domain_id_b AS domain_2,
                                NOT negative AS interaction, source
                        FROM domain_domain_interaction;
                ''', conn)

            os.makedirs("ddi_out/DDI/", exist_ok=True)
            ddi_df.to_csv("ddi_out/DDI/test.csv", index=False)
            source_df.to_csv("ddi_out/DDI/test_sources.csv", index=False)
            PYEOF

            cat <<-END_VERSIONS > versions.yml
            "${task.process}":
                python: \$(python --version 2>&1 | sed 's/Python //')
                pandas: \$(python -c 'import pandas as pd; print(pd.__version__)')
            END_VERSIONS
            """
        } else {
            """
            #!/usr/bin/env bash
            python3 <<'PYEOF'
            import pandas as pd
            import sqlite3
            import os

            DDI_QUERY = '''
                SELECT domain_id_a AS domain_1, domain_id_b AS domain_2,
                       NOT negative AS interaction
                FROM domain_domain_interaction;
            '''

            # Same rows plus the provenance list. `source` is a comma-joined list
            # of every source that contributed the pair (domainsplit keys
            # `domain_domain_interaction` UNIQUE on the pair), so a DDI with
            # "single_domain_ppi,PPIDM,PPIDM_Gold" counts towards all three
            # per-source performance rows in the report.
            SOURCES_QUERY = '''
                SELECT domain_id_a AS domain_1, domain_id_b AS domain_2,
                       NOT negative AS interaction, source
                FROM domain_domain_interaction;
            '''

            # `ddi_split_membership` canonicalises each pair by sorting the two
            # instance ids, so instance_id_a does not necessarily belong to
            # domain_id_a. Resolve the side through domain_protein_map.
            # Homodimers satisfy both branches -- either orientation is correct.
            INSTANCE_QUERY = '''
                SELECT ddi.domain_id_a AS domain_1,
                       ddi.domain_id_b AS domain_2,
                       CASE WHEN pa.domain_id = ddi.domain_id_a
                            THEN m.instance_id_a ELSE m.instance_id_b END AS instance_1,
                       CASE WHEN pa.domain_id = ddi.domain_id_a
                            THEN m.instance_id_b ELSE m.instance_id_a END AS instance_2,
                       NOT ddi.negative AS interaction
                FROM ddi_split_membership AS m
                JOIN domain_domain_interaction AS ddi ON ddi.id = m.ddi_id
                JOIN domain_protein_map AS pa ON pa.instance_id = m.instance_id_a;
            '''

            def has_table(conn, name):
                return conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
                ).fetchone() is not None

            out_dir = "ddi_out/DDI"
            os.makedirs(out_dir, exist_ok=True)

            for split in "${splits}".split():
                path = f"${database_dir}/{split}.sqlite3"
                if not os.path.isfile(path):
                    continue
                with sqlite3.connect(path) as conn:
                    pd.read_sql(DDI_QUERY, conn).to_csv(f"{out_dir}/{split}.csv", index=False)
                    pd.read_sql(SOURCES_QUERY, conn).to_csv(
                        f"{out_dir}/{split}_sources.csv", index=False
                    )
                    if has_table(conn, "ddi_split_membership"):
                        pd.read_sql(INSTANCE_QUERY, conn).to_csv(
                            f"{out_dir}/{split}_instances.csv", index=False
                        )
            PYEOF

            cat <<-END_VERSIONS > versions.yml
            "${task.process}":
                python: \$(python --version 2>&1 | sed 's/Python //')
                pandas: \$(python -c 'import pandas as pd; print(pd.__version__)')
            END_VERSIONS
            """
        }
}
