import h5py
import numpy as np
import pandas as pd
import sqlite3

# HDF5 layout is `h5[pfam_id][instance_key]`.
#
# The group name is the domain's **Pfam accession**, not `domain.id`.
# `domain.id` is a per-run surrogate integer -- domainsplit's SUBSET_SPLIT_DB
# copies it verbatim and PRUNE_UNREPRESENTED_DDIS deletes without renumbering,
# so the same integer names a different domain in the next run and nothing keyed
# on it can be compared across runs. `domain` is UNIQUE(pfam_id) with
# `id INTEGER PRIMARY KEY`, so the two are in bijection within a database and
# the swap loses nothing. DDI_EXTRACTION's CSVs, the graph models
# (`bin/load_data_gm.py`) and the prediction files all speak Pfam accessions, so
# the feature files have to as well or the ML loader resolves nothing.
#
# The dataset name inside the group is the *instance*, not the protein:
# `domain_protein_map` is unique on (domain_id, protein_id, start_pos, end_pos),
# so a protein carrying two copies of one family yields two rows that would
# otherwise collide on protein_id. `instance_id` is domainsplit's own opaque
# identifier -- never parse it. The rowid fallback only has to be unique within
# the file the .h5 is built from, which is the same file the DDI CSVs come from.
def instance_key_sql(alias: str = "domain_protein_map") -> str:
    """The instance-key expression for one `domain_protein_map` alias."""
    return f"COALESCE({alias}.instance_id, 'r' || {alias}.rowid)"


INSTANCE_KEY_SQL = f"{instance_key_sql()} AS instance_key"


def domain_key_sql(alias: str = "domain") -> str:
    """The HDF5 group-name expression for one `domain` alias."""
    return f"{alias}.pfam_id"


DOMAIN_KEY_SQL = f"{domain_key_sql()} AS domain_key"

#: Join that resolves `domain_protein_map.domain_id` to its Pfam accession.
#: Every extractor needs it, so it lives here rather than being retyped.
DOMAIN_JOIN_SQL = "JOIN domain ON domain_protein_map.domain_id = domain.id"


def write_instance(out_file: h5py.File, domain_key: str, instance_key: str, value):
    """Write one instance vector under its domain group, creating it if needed.

    `domain_key` is the Pfam accession (see the note above), `instance_key` the
    opaque domain-instance identifier.

    Cast to float32, the only precision the pipeline carries -- the ML loader
    assembles every row into a float32 array, so anything wider is truncated on
    the way in anyway, and the rounding is the same either way. Without the cast
    h5py takes the dtype from what it is handed, so the `list[float]` an encoder
    naturally returns lands as float64 and doubles the file (aacomp and aaencode
    both did).
    """
    if domain_key not in out_file:
        domain_group = out_file.create_group(domain_key)
    else:
        domain_group = out_file[domain_key]

    domain_group[instance_key] = np.asarray(value, dtype=np.float32)


# Interaction encodings
# HDF5 layout is `h5[<pfam_id_a>_<pfam_id_b>][<instance_id_a>_<instance_id_b>]`.
# 
# The group name is the concatenation of the two Pfam accessions, as they occurr in ddi_instance table
# The table ddi_split_membership is structured as follows:
# CREATE TABLE ddi_split_membership (
#             ddi_id REFERENCES domain_domain_interaction ON DELETE CASCADE,
#             method, split,
#             instance_id_a, instance_id_b,
#             z_score REAL,
#             is_mock INTEGER DEFAULT 0,
#             UNIQUE(ddi_id, method, split, instance_id_a, instance_id_b)
#         );
# For each ddi_id, there are multiple instances
# Get for each ddi_id, the pfam ids via a join with domain_domain_interaction table and with domain table
# domain_domain_interaction only contains domain_id_a and domain_id_b, so we need to join with domain table to get the pfam ids

# join operation

DDI_JOIN_SQL = """
    SELECT ddi.id, d1.pfam_id AS pfam_id_a, d2.pfam_id AS pfam_id_b
    FROM domain_domain_interaction ddi
    JOIN domain d1 ON ddi.domain_id_a = d1.id
    JOIN domain d2 ON ddi.domain_id_b = d2.id
"""




# # File to load structures from is created with
# group_key = f"{row.pfam_id_a}_{row.pfam_id_b}"
# dset_key = f"{row.instance_id_a}__{row.instance_id_b}"
# grp = h5file.require_group(group_key)
# if dset_key not in grp:
#     # pdb_gz is already gzip-compressed (see
#     # utils_struct.ddi_pair_to_bytes) -- no h5 compression
#     # filter here, or we'd be gzipping gzip for no benefit.
#     grp.create_dataset(dset_key, data=np.frombuffer(pdb_gz, dtype="uint8"),
#                         compression=None)
# IF there is no entry in the file, we return None, and the feature extractor should skip that instance pair.

def interaction_group_name(pfam_id_a: str, pfam_id_b: str) -> str:
    """Return the HDF5 group name for a pair of Pfam accessions."""
    return f"{pfam_id_a}_{pfam_id_b}"

def interaction_dataset_name(instance_id_a: str, instance_id_b: str) -> str:
    """Dataset name for an interaction-encoding *feature* file (out_file).

    Single underscore -- this is what machine_learning.py's
    `combination_available` builds when checking a feature file for a
    candidate instance pair (`joined = f"{instance_a}_{instance_b}"`). Used by
    write_interaction_instance, i.e. every feature encoder's output.
    """
    return f"{instance_id_a}_{instance_id_b}"


def structure_dataset_name(instance_id_a: str, instance_id_b: str) -> str:
    """Dataset name inside the (external, already-built) structures.h5.

    Double underscore -- this matches utils_struct's existing on-disk format
    (`dset_key = f"{row.instance_id_a}__{row.instance_id_b}"`), which we only
    read from and don't control. Used by read_interaction_instance.
    """
    return f"{instance_id_a}__{instance_id_b}"


def read_interaction_instance(
    h5_file: h5py.File,
    pfam_id_a: str,
    pfam_id_b: str,
    instance_id_a: str,
    instance_id_b: str,
):
    group_key = interaction_group_name(pfam_id_a, pfam_id_b)
    dset_key = structure_dataset_name(instance_id_a, instance_id_b)  # changed

    group = h5_file.get(group_key)
    if group is None:
        return None
    dset = group.get(dset_key)
    if dset is None:
        return None
    return dset[()]


def write_interaction_instance(
    out_file: h5py.File,
    pfam_id_a: str,
    pfam_id_b: str,
    instance_id_a: str,
    instance_id_b: str,
    value,
):
    """Write one interaction-instance vector under its DDI group, creating it if needed.

    `pfam_id_a`/`pfam_id_b` form the group name (see `interaction_group_name`),
    `instance_id_a`/`instance_id_b` the dataset name within it (see
    `interaction_dataset_name`).

    Cast to float32, the only precision the pipeline carries -- same reasoning
    as `write_instance`.
    """
    group_key = interaction_group_name(pfam_id_a, pfam_id_b)
    dset_key = interaction_dataset_name(instance_id_a, instance_id_b)

    if group_key not in out_file:
        group = out_file.create_group(group_key)
    else:
        group = out_file[group_key]

    group[dset_key] = np.asarray(value, dtype=np.float32)


def load_structure_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load every DDI instance pair with its Pfam accessions resolved.

    Joins ddi_split_membership (the actual instance_id_a/instance_id_b pairs)
    against domain_domain_interaction/domain (via DDI_JOIN_SQL) to attach
    pfam_id_a/pfam_id_b -- the same id-instability reasoning as
    DOMAIN_JOIN_SQL applies here: ddi.id and domain.id are per-run surrogates,
    so anything cached or compared across runs has to key on Pfam accessions
    instead.

    Every interaction encoder (aacomp_interface, and any future one) needs
    this exact merge, so it lives here once rather than being reimplemented
    per encoder.

    Returns a DataFrame with columns ddi_id, pfam_id_a, pfam_id_b,
    instance_id_a, instance_id_b -- all cast to str, ready to pass straight
    into interaction_group_name/interaction_dataset_name or
    read_interaction_instance/write_interaction_instance.
    """
    ddi_pfam_df = pd.read_sql(DDI_JOIN_SQL, conn)

    # TODO: later add is_mock = 0, to only load real instances
    # current db is not yet updated with is_mock column, so we load all instances for now
    membership_df = pd.read_sql(
        """
        SELECT ddi_id, instance_id_a, instance_id_b
        FROM ddi_split_membership
        WHERE is_mock = 0
        """,
        conn,
    )

    ddi_df = membership_df.merge(
        ddi_pfam_df, left_on="ddi_id", right_on="id", how="inner"
    ).drop(columns=["id"])

    ddi_df["pfam_id_a"] = ddi_df["pfam_id_a"].astype(str)
    ddi_df["pfam_id_b"] = ddi_df["pfam_id_b"].astype(str)
    ddi_df["instance_id_a"] = ddi_df["instance_id_a"].astype(str)
    ddi_df["instance_id_b"] = ddi_df["instance_id_b"].astype(str)

    return ddi_df