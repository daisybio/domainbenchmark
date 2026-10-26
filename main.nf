#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    daisybio/domainbenchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Github : https://github.com/daisybio/domainbenchmark
----------------------------------------------------------------------------------------
*/

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TYPED PARAMETER DECLARATIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Types only -- the *defaults* stay in nextflow.config, which remains the single
    source of truth for them.

    Nextflow's v2 (strict) parser, the default since 26.x, stopped inferring types
    for command-line parameters: `--seed 7` arrives as the String "7", and
    nf-schema then rejects it against the schema's `"type": "integer"` with
    "Value is [string] but should be [integer]". Declaring the type here makes
    Nextflow coerce the CLI value before validation runs, which is the v2-native
    fix -- `validation.lenientMode` is not it (it only widens the other
    direction, letting an integer satisfy a string type), and
    `NXF_SYNTAX_PARSER=v1` only works by reverting to the retired parser.

    Every numeric or boolean param a user might override on the command line
    belongs here. `Float`, not `Double`/`BigDecimal`/`Number`: those reject the
    String outright instead of coercing it. Booleans are `Boolean` -- a bare
    `--allow_cpu_ml` was always fine, `--allow_cpu_ml true` was not.

    Only the *entry* script's declarations count: a `params { }` block inside an
    included script is ignored, so a pipeline embedding this one would have to
    repeat these.

    The nf-core boolean flags are declared too. They are usually passed bare,
    but `--validate_params false` arrived as the String "false" -- truthy in
    Groovy -- so it left validation *on*. `help` stays undeclared: it is a
    string-or-boolean union a type here would break.
----------------------------------------------------------------------------------------
*/
params {
    seed: Integer
    ppi_score_cutoff: Integer
    min_embedding_coverage: Float
    allow_cpu_ml: Boolean
    validate_params: Boolean
    monochrome_logs: Boolean
    plaintext_email: Boolean
    help_full: Boolean
    show_hidden: Boolean
    version: Boolean
}

include { PIPELINE_INITIALISATION } from './subworkflows/local/utils_nfcore_domainbenchmark_pipeline'
include { PIPELINE_COMPLETION     } from './subworkflows/local/utils_nfcore_domainbenchmark_pipeline'
include { PER_DB_BENCHMARK        } from './subworkflows/local/per_db_benchmark/main.nf'
include { AGGREGATE_EVAL          } from './subworkflows/local/aggregate_eval/main.nf'
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline depending on type of input
//
workflow DAISYBIO_DOMAINBENCHMARK {

    take:
        db_ch   // channel: tuple(meta, db_path)

    main:
        PER_DB_BENCHMARK(db_ch)
        AGGREGATE_EVAL(PER_DB_BENCHMARK.out.report)

    emit:
        per_db_report = PER_DB_BENCHMARK.out.report
        combined      = AGGREGATE_EVAL.out.report
}
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    main:
    //
    // SUBWORKFLOW: Run initialisation tasks
    //
    PIPELINE_INITIALISATION (
        params.version,
        params.validate_params,
        params.monochrome_logs,
        args,
        params.outdir,
        params.input,
        params.help,
        params.help_full,
        params.show_hidden
    )

    //
    // WORKFLOW: Run main workflow
    //
    DAISYBIO_DOMAINBENCHMARK (
        PIPELINE_INITIALISATION.out.db_ch
    )
    //
    // SUBWORKFLOW: Run completion tasks
    //
    PIPELINE_COMPLETION (
        params.email,
        params.email_on_fail,
        params.plaintext_email,
        params.outdir,
        params.monochrome_logs,
        // The report file, not COMBINE_EVAL's (meta, dir) tuple: completionEmail
        // attaches whatever it is handed and turned the tuple into a bogus path.
        DAISYBIO_DOMAINBENCHMARK.out.combined.map { _meta, dir -> dir / 'ddi_report.html' }
    )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
