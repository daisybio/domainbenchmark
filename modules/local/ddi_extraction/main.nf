process DDI_EXTRACTION {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database_dir)

    output:
        tuple val(meta), path("${meta.id}/DDI"), emit: ddi
        path "versions.yml",                     emit: versions

    script:
        // Single-file db inputs only support a 'test' split.
        // Directory inputs are expected to contain train/test/optimization sqlite splits.
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

            os.makedirs(f"${meta.id}/DDI/", exist_ok=True)
            ddi_df.to_csv(f"${meta.id}/DDI/test.csv", index=False)
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

            os.makedirs(f"${meta.id}/DDI/", exist_ok=True)
            for dbtype in ('train', 'test', 'optimization'):
                path = f"${database_dir}/{dbtype}.sqlite3"
                if os.path.isfile(path):
                    with sqlite3.connect(path) as conn:
                        ddi_df = pd.read_sql('''
                                SELECT domain_id_a AS domain_1, domain_id_b AS domain_2,
                                        NOT negative AS interaction
                                FROM domain_domain_interaction
                                WHERE is_evaluation_relevant;
                        ''', conn)
                    ddi_df.to_csv(f"${meta.id}/DDI/{dbtype}.csv", index=False)
            PYEOF

            cat <<-END_VERSIONS > versions.yml
            "${task.process}":
                python: \$(python --version 2>&1 | sed 's/Python //')
                pandas: \$(python -c 'import pandas as pd; print(pd.__version__)')
            END_VERSIONS
            """
        }
}
