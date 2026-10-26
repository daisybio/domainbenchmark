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
        def datasets

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

            datasets = dataset_dirs.collect { dir -> datasetTuple(dir.getName(), dir) }
        } else {
            // Read eagerly rather than through a channel: the checks below need
            // every row at once, and they have to fail at launch -- not after
            // the models have trained.
            //
            // `assets/schema_input.json` is deliberately not wired into the
            // `input` schema entry: `--input` may also be a directory, which a
            // samplesheet schema would reject, and relative `db_path`s resolve
            // against projectDir here but against launchDir in nf-schema.
            datasets = input_path.splitCsv(header: true).withIndex().collect { row, i ->
                def where = "Samplesheet ${input_path}, row ${i + 2}"
                if (!row.db_path?.trim()) {
                    error("${where}: db_path is empty or the column is missing.")
                }
                if (row.id != null && !(row.id ==~ /\S+/)) {
                    error("${where}: id '${row.id}' must be a non-empty string with no spaces.")
                }
                def db_path = row.db_path.startsWith('/')
                    ? file(row.db_path, checkIfExists: true)
                    : file("${workflow.projectDir}/${row.db_path}", checkIfExists: true)
                if (!db_path.isDirectory()) {
                    error("${where}: db_path ${db_path} is not a directory.")
                }
                datasetTuple(row.id ?: db_path.getName(), db_path)
            }
        }

        validateDatasets(datasets)
        validateModelParams()
        db_ch = Channel.fromList(datasets)
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

//
// Accept either a List or a comma-separated String. Filter out blanks and the
// sentinel 'none' so callers can disable a list cleanly with
// `--graph_models none` (or `--graph_models ''` once nf-schema allows it).
//
def csvToList(s) {
    def items = s instanceof List ? s : (s ? s.tokenize(',') : [])
    items*.trim().findAll { it && it.toLowerCase() != 'none' }
}

//
// A database with an internal test set produces two runs (`random_balanced`,
// `random_realistic`); one with a single `test` keeps its bare name. The label
// identifies the run everywhere downstream -- process tags, the evaluation
// directory, and the entry in the combined cross-database report.
//
def runLabel(db_id, variant) {
    variant == 'test' ? "${db_id}".toString() : "${db_id}_${variant}".toString()
}

//
// Launch-time checks over the whole dataset list.
//
// Every one of these used to surface late or not at all: two samplesheet rows
// with one id merged their `<feature>__<split>.h5` files under one key, two
// datasets sharing a run label (`random` + `test_balanced` next to a database
// named `random_balanced`) collided in COMBINE_EVAL's staging and one of them
// vanished from the combined report, and an `--mqc_order` typo was only caught
// by COMBINE_EVAL -- the last task of the run, after every model had trained.
//
def validateDatasets(datasets) {
    def ids = datasets.collect { it[0].id }
    def dup_ids = ids.countBy { it }.findAll { _id, n -> n > 1 }.keySet().sort()
    if (dup_ids) {
        error("Duplicate dataset id(s): ${dup_ids.join(', ')}. Every database needs its own id.")
    }

    def labels = datasets.collectMany { ds ->
        def meta = ds[0]
        meta.tests.keySet().collect { variant -> [runLabel(meta.id, variant), "${meta.id}/${variant}"] }
    }
    def clashes = labels.groupBy { it[0] }.findAll { _label, owners -> owners.size() > 1 }
    if (clashes) {
        def detail = clashes.collect { label, owners -> "'${label}' (${owners*.get(1).join(' and ')})" }
        error("Run label collision: ${detail.sort().join('; ')}. Rename one of the databases.")
    }

    // Same normalisation as eval_multiqc_functions.parse_dataset_order: split
    // on commas, trim, drop empties.
    def requested = params.mqc_order ? params.mqc_order.toString().tokenize(',')*.trim().findAll { it } : []
    def repeated = requested.countBy { it }.findAll { _name, n -> n > 1 }.keySet().sort()
    if (repeated) {
        error("--mqc_order names dataset(s) more than once: ${repeated.join(', ')}.")
    }
    def known = labels.collect { it[0] }
    def unknown = requested.findAll { !known.contains(it) }
    if (unknown) {
        error(
            "--mqc_order names dataset(s) that do not exist: ${unknown.join(', ')}. " +
            "Available: ${known.sort().join(', ')}."
        )
    }
}

//
// Every model, feature and --skip name must refer to something that exists.
// csvToList used to accept any token, so `--machine_learning_models
// random_forrest` silently trained nothing and the run "succeeded" without a
// report.
//
def validateModelParams() {
    def ml_known  = ['neural_network', 'random_forest']
    def ml_models = csvToList(params.machine_learning_models)
    def bad_ml    = ml_models.findAll { !ml_known.contains(it) }
    if (bad_ml) {
        error("Unknown machine_learning_models: ${bad_ml.join(', ')}. Known: ${ml_known.join(', ')}.")
    }

    def graph_models = csvToList(params.graph_models)
    def bad_graph    = graph_models.findAll { !file("${params.modeljson}/${it}.json").exists() }
    if (bad_graph) {
        error("Unknown graph_models: ${bad_graph.join(', ')} (no <name>.json under ${params.modeljson}).")
    }

    def features  = csvToList(params.machine_learning_features)
    def published = csvToList(params.published_features)
    def bad_feat  = features.findAll {
        !published.contains(it) && !file("${workflow.projectDir}/bin/features/${it}.py").exists()
    }
    if (bad_feat) {
        error(
            "Unknown machine_learning_features: ${bad_feat.join(', ')}. An extracted feature " +
            "needs bin/features/<name>.py; a published one must be listed in published_features."
        )
    }

    def skippable = features + ml_models + graph_models
    def bad_skip  = csvToList(params.skip).findAll { !skippable.contains(it) }
    if (bad_skip) {
        error("--skip names nothing that would run: ${bad_skip.join(', ')}. Skippable: ${skippable.join(', ')}.")
    }
}
