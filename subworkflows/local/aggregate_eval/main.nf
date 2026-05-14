/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    AGGREGATE_EVAL -- combine per-DB evaluation reports across DBs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Replaces wrapper.nf. Takes a channel of per-DB evaluation directories,
    folds them into a single COMBINE_EVAL invocation, and emits the combined
    report directory.
----------------------------------------------------------------------------*/

include { COMBINE_EVAL } from '../../../modules/local/evaluation/main.nf'


workflow AGGREGATE_EVAL {

    take:
        per_db_reports   // channel: tuple(meta, evaluation_dir) — emitted by PER_DB_BENCHMARK

    main:
        def reports_collected = per_db_reports
            .map { _meta, dir -> dir }
            .collect()

        def combine_input_ch = reports_collected.map { dirs ->
            def m = [ id: 'combined' ]
            tuple(m, dirs)
        }

        COMBINE_EVAL(combine_input_ch)

    emit:
        report   = COMBINE_EVAL.out.combined_report
        versions = COMBINE_EVAL.out.versions
}
