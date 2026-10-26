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
# identifier -- never parse it. It is `<family>_<accession>_<start>_<end>`, so
# it is stable across runs as well as unique within one.
#
# The `'r' || rowid` fallback is not: a rowid is local to the database file it
# was read from, so it can only ever match an .h5 extracted from that same split
# database. A per-run published file is built from the run's own database and
# never carries one. In practice the loader never asks for it either --
# `ddi_split_membership` names instances by `instance_id`, so every instance it
# assigns has a non-NULL one and the COALESCE is the identity there.
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


#: The 20 standard amino acids. Everything else (X, B, Z, J, U, O, gaps, ...) is
#: stripped by `clean_sequence` before a sequence-derived encoder sees it.
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence, domain_key, instance_key, min_length: int = 1) -> str:
    """Upper-case `sequence` and drop every non-standard residue, or raise.

    Shared by every sequence encoder so they agree on what a residue is.
    Stripping rather than mapping: an ambiguity code carries no property value
    any encoder could look up, and substituting one would invent signal.

    Raises when nothing usable is left (NULL, empty, or shorter than
    `min_length` after stripping). Each encoder used to fail differently here
    -- a ZeroDivisionError, a KeyError, or worse a silent all-zero vector that
    trains like real data -- so fail once, naming the instance.
    """
    if sequence is None or (isinstance(sequence, float) and sequence != sequence):
        raise ValueError(f"{domain_key}/{instance_key}: domain_sequence is NULL")
    raw = str(sequence).upper()
    cleaned = "".join(residue for residue in raw if residue in STANDARD_AA)
    if len(cleaned) < max(min_length, 1):
        raise ValueError(
            f"{domain_key}/{instance_key}: {len(cleaned)} standard residue(s) left "
            f"of {len(raw)} after stripping non-standard ones, need at least "
            f"{max(min_length, 1)} (sequence starts {raw[:30]!r})"
        )
    return cleaned


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

    A "/" in either key is rejected: h5py reads it as a path separator and
    would silently create nested groups the loader never looks up.
    """
    for key in (domain_key, instance_key):
        if "/" in key:
            raise ValueError(f"HDF5 key {key!r} contains '/', which h5py treats as a group separator")
    if domain_key not in out_file:
        domain_group = out_file.create_group(domain_key)
    else:
        domain_group = out_file[domain_key]

    domain_group[instance_key] = np.asarray(value, dtype=np.float32)
