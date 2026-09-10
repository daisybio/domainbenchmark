//
// Subworkflow with functionality specific to the daisybio/domainbenchmark pipeline
//

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { UTILS_NFSCHEMA_PLUGIN     } from '../../nf-core/utils_nfschema_plugin'
include { paramsSummaryMap          } from 'plugin/nf-schema'
include { completionEmail           } from '../../nf-core/utils_nfcore_pipeline'
include { completionSummary         } from '../../nf-core/utils_nfcore_pipeline'
include { UTILS_NFCORE_PIPELINE     } from '../../nf-core/utils_nfcore_pipeline'
include { UTILS_NEXTFLOW_PIPELINE   } from '../../nf-core/utils_nextflow_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SUBWORKFLOW TO INITIALISE PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow PIPELINE_INITIALISATION {

    take:
    version           // boolean: Display version and exit
    validate_params   // boolean: Boolean whether to validate parameters against the schema at runtime
    monochrome_logs   // boolean: Do not use coloured log outputs
    nextflow_cli_args //   array: List of positional nextflow CLI args
    outdir            //  string: The output directory where the results will be saved
    input             //  string: Path to input samplesheet
    help              // boolean: Display help message and exit
    help_full         // boolean: Show the full help message
    show_hidden       // boolean: Show hidden parameters in the help message

    main:

    //
    // Print version and exit if required and dump pipeline parameters to JSON file
    //
    UTILS_NEXTFLOW_PIPELINE (
        version,
        true,
        outdir,
        workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1
    )

    //
    // Validate parameters and generate parameter summary to stdout
    //

    def before_text = ""
    def after_text = ""
    if (monochrome_logs) {
        before_text = before_text.replaceAll(/\033\[[0-9;]*m/, '')
    }

    command = "nextflow run ${workflow.manifest.name} -profile <docker/singularity/.../institute> --input samplesheet.csv --outdir <OUTDIR>"

    UTILS_NFSCHEMA_PLUGIN (
        workflow,
        validate_params,
        null,
        help,
        help_full,
        show_hidden,
        before_text,
        after_text,
        command
    )

    //
    // Check config provided to the pipeline
    //
    UTILS_NFCORE_PIPELINE (
        nextflow_cli_args
    )

    //
    // Custom tests
    //
    if (params.input) {
        def input_path = file(params.input, checkIfExists: true)

        if (input_path.isDirectory()) {
            // Directory mode: `--input databases/` as published by
            // daisybio/domainsplit. Every immediate subdirectory holding a
            // train.sqlite3 is a dataset, named after the directory.
            def dataset_dirs = input_path
                .listFiles()
                .findAll { it.isDirectory() && it.resolve('train.sqlite3').exists() }
                .sort { it.getName() }

            if (!dataset_dirs) {
                error("No dataset directories containing train.sqlite3 found under ${input_path}")
            }

            db_ch = Channel.fromList(
                dataset_dirs.collect { dir -> datasetTuple(dir.getName(), dir) }
            )
        } else {
            db_ch = Channel
                .fromPath(params.input, checkIfExists: true)
                .splitCsv(header: true)
                .map { row ->
                    def db_path = row.db_path.startsWith('/')
                        ? file(row.db_path, checkIfExists: true)
                        : file("${workflow.projectDir}/${row.db_path}", checkIfExists: true)
                    datasetTuple(row.id ?: db_path.getName(), db_path)
                }
        }
    } else {
        error "No input provided: set --input <samplesheet.csv|databases/>"
    }

    emit:
    db_ch = db_ch
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SUBWORKFLOW FOR PIPELINE COMPLETION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow PIPELINE_COMPLETION {

    take:
    email           //  string: email address
    email_on_fail   //  string: email address sent on pipeline failure
    plaintext_email // boolean: Send plain-text email instead of HTML
    outdir          //    path: Path to output directory where results will be published
    monochrome_logs // boolean: Disable ANSI colour codes in log output
    multiqc_report  //  string: Path to MultiQC report

    main:
    summary_params = paramsSummaryMap(workflow, parameters_schema: "nextflow_schema.json")
    def multiqc_reports = multiqc_report.toList()

    //
    // Completion email and summary
    //
    workflow.onComplete {
        if (email || email_on_fail) {
            completionEmail(
                summary_params,
                email,
                email_on_fail,
                plaintext_email,
                outdir,
                monochrome_logs,
                multiqc_reports.getVal(),
            )
        }

        completionSummary(monochrome_logs)

    }

    workflow.onError {
        log.error "Pipeline failed. Please refer to troubleshooting docs for common issues: https://nf-co.re/docs/running/troubleshooting"
    }
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// Discover the sqlite splits inside one database directory.
//
// daisybio/domainsplit publishes `databases/<dataset>/<split>.sqlite3` with
// `train`, `validation`, and one or more `test*` splits: `test` for a dataset
// whose test set comes from BUILD_EXTERNAL_TEST, `test_balanced` +
// `test_realistic` for one with an internal test set. Each test split is
// benchmarked separately against the same trained models.
//
// Returns [ splits, tests ]:
//   splits — ordered split names, i.e. the sqlite stems
//   tests  — variant -> split name (`test_balanced` -> `balanced`, `test` -> `test`)
//
def discoverSplits(db_dir) {
    def names = db_dir.list().findAll { it.endsWith('.sqlite3') }.collect { it - '.sqlite3' }

    ['train', 'validation'].each { required ->
        if (!names.contains(required)) {
            error("Database directory ${db_dir} is missing ${required}.sqlite3 (found: ${names.sort().join(', ')})")
        }
    }

    def test_splits = names.findAll { it == 'test' || it.startsWith('test_') }.sort()
    if (!test_splits) {
        error("Database directory ${db_dir} contains no test*.sqlite3 split (found: ${names.sort().join(', ')})")
    }

    def tests = test_splits.collectEntries { split ->
        [ (split == 'test' ? 'test' : split - 'test_'), split ]
    }

    return [ ['train', 'validation'] + test_splits, tests ]
}

//
// Build the (meta, db_path) tuple for one database directory.
//
def datasetTuple(id, db_dir) {
    def (splits, tests) = discoverSplits(db_dir)
    return tuple([ id: id, db: id, splits: splits, tests: tests ], db_dir)
}
