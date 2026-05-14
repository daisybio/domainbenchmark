#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    daisybio/cobinet-benchmarking-pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Github : https://github.com/daisybio/cobinet-benchmarking-pipeline
----------------------------------------------------------------------------------------
*/

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { COBINET-BENCHMARKING-PIPELINE  } from './workflows/cobinet-benchmarking-pipeline'
include { PIPELINE_INITIALISATION } from './subworkflows/local/utils_nfcore_cobinet-benchmarking-pipeline_pipeline'
include { PIPELINE_COMPLETION     } from './subworkflows/local/utils_nfcore_cobinet-benchmarking-pipeline_pipeline'
include { PER_DB_BENCHMARK } from './subworkflows/local/per_db_benchmark/main.nf'
include { AGGREGATE_EVAL   } from './subworkflows/local/aggregate_eval/main.nf'
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline depending on type of input
//
workflow DAISYBIO_COBINET-BENCHMARKING-PIPELINE {

    take:
        db_ch   // channel: tuple(meta, db_path)

    main:
        PER_DB_BENCHMARK(db_ch)
        AGGREGATE_EVAL(PER_DB_BENCHMARK.out.report)

    emit:
        per_db_report = PER_DB_BENCHMARK.out.report
        combined      = AGGREGATE_EVAL.out.report
        versions      = PER_DB_BENCHMARK.out.versions.mix(AGGREGATE_EVAL.out.versions)
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
        // pipeline-specific input
        params.db_list,
        params.db
    )

    //
    // WORKFLOW: Run main workflow
    //
    DAISYBIO_COBINET-BENCHMARKING-PIPELINE (
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
        DAISYBIO_COBINET-BENCHMARKING-PIPELINE.out.combined
    )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
