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
        // Collect (db_id, dir) pairs as a single list so COMBINE_EVAL can stage
        // each per-DB report under a parent dir named by db_id. Flat collect
        // would lose the pairing; flat:false preserves the per-entry tuples.
        def combine_input_ch = per_db_reports
            .map { meta, dir -> [ meta.id, dir ] }
            .collect(flat: false)
            .map { entries ->
                def m = [ id: 'combined' ]
                // collect() preserves arrival order, i.e. whichever database
                // finished evaluating first. Sorting by db_id keeps the staged
                // `src/<n>/` numbering, the task hash and the combined report's
                // byte content the same from run to run.
                def sorted = entries.toSorted { a, b -> a[0] <=> b[0] }
                def ids  = sorted.collect { it[0] }
                def dirs = sorted.collect { it[1] }
                tuple(m, dirs, ids)
            }

        COMBINE_EVAL(combine_input_ch)

    emit:
        report = COMBINE_EVAL.out.combined_report
}
