import h5py
import numpy as np

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
