"""Shared plumbing for the structure-derived (interaction) encoders.

Input is domainsplit's ENRICH_STRUCTURAL `structures.h5`: group
`<pfamA>_<pfamB>`, dataset `<instA>__<instB>`, a uint8 array of gzip'd PDB text
with chain A = instance A and chain B = instance B, keyed in whichever
orientation domainsplit stored the membership row.

Every encoder is `encode(cx, **params) -> 1-D vector` over a `Complex`, wrapped
by `make_extractor`, which owns the loop: it walks the split's membership pairs
that have a structure (`iter_structures`), computes the vector for both chain
orders and writes both orientations in the interaction layout protdcal uses.

Degenerate input follows one rule: a value that is genuinely zero (no
interface contact, no clash) is written as zero; anything undefined (missing
chain, non-standard residue, missing atom, empty denominator) raises naming
the pair. Nothing is filled with NaN or 0.0 to keep going.
"""
import gzip
import io
from collections import defaultdict

import h5py
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial.distance import cdist

from features import embeddings

# Tien et al. 2013, "Maximum allowed solvent accessibilities of residues in
# proteins" (https://doi.org/10.1002/prot.24286), theoretical values.
MAX_SASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 223.0, "GLU": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}
BACKBONE = ("N", "CA", "C", "O")
CHAINS = ("A", "B")

PAIR_QUERY = f"""
    SELECT m.ddi_id, ddi.negative,
           m.instance_id_a, m.instance_id_b, m.has_structure,
           {embeddings.domain_key_sql("d1")} AS pfam_a,
           {embeddings.domain_key_sql("d2")} AS pfam_b
    FROM ddi_split_membership AS m
    JOIN domain_domain_interaction AS ddi ON ddi.id = m.ddi_id
    LEFT JOIN domain_protein_map AS dpm_1 ON dpm_1.instance_id = m.instance_id_a
    LEFT JOIN domain_protein_map AS dpm_2 ON dpm_2.instance_id = m.instance_id_b
    LEFT JOIN domain AS d1 ON dpm_1.domain_id = d1.id
    LEFT JOIN domain AS d2 ON dpm_2.domain_id = d2.id
    ORDER BY m.instance_id_a, m.instance_id_b, m.ddi_id
"""


def structure_rows(conn, struct_h5: h5py.File):
    """Membership pairs with a structure, resolved to their structures.h5 key.

    Returns a sorted list of `(pfam_a, pfam_b, inst_a, inst_b, group, dataset,
    swapped)`; `swapped` means the structure is stored as (b, a), so its chain
    A is `inst_b`. Prints the split's coverage and raises on inconsistent input
    or a split without a single structure.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(ddi_split_membership)")}
    if not columns:
        raise ValueError("structure features need ddi_split_membership; this database has none")
    if "has_structure" not in columns:
        raise ValueError(
            "ddi_split_membership has no has_structure column: this database predates "
            "domainsplit's ENRICH_STRUCTURAL and cannot be paired with a structures.h5"
        )
    rows = conn.execute(PAIR_QUERY).fetchall()
    (n_membership,) = conn.execute("SELECT COUNT(*) FROM ddi_split_membership").fetchone()
    if len(rows) != n_membership:
        raise ValueError(
            f"ddi_split_membership has {n_membership} rows but {len(rows)} resolved through "
            "domain_domain_interaction/domain_protein_map: a DDI is missing or an instance_id is duplicated"
        )

    pairs = {0: [0, 0], 1: [0, 0]}  # negative -> [total, with structure]
    ddis = {}  # ddi_id -> (negative, has any structure)
    resolved, seen = [], set()
    for ddi_id, negative, inst_a, inst_b, has_structure, pfam_a, pfam_b in rows:
        if pfam_a is None or pfam_b is None:
            raise ValueError(f"membership pair {inst_a}/{inst_b} (ddi {ddi_id}) has no domain_protein_map/domain row")
        if negative not in (0, 1) or has_structure not in (0, 1):
            raise ValueError(
                f"membership pair {inst_a}/{inst_b}: negative={negative!r}, has_structure={has_structure!r}; both must be integer 0/1"
            )
        pairs[negative][0] += 1
        pairs[negative][1] += has_structure
        ddis[ddi_id] = (negative, ddis.get(ddi_id, (negative, False))[1] or bool(has_structure))
        if not has_structure:
            continue
        for group, dataset, swapped in (
            (f"{pfam_a}_{pfam_b}", f"{inst_a}__{inst_b}", False),
            (f"{pfam_b}_{pfam_a}", f"{inst_b}__{inst_a}", True),
        ):
            if group in struct_h5 and dataset in struct_h5[group]:
                break
        else:
            raise ValueError(
                f"membership pair {inst_a}/{inst_b} (ddi {ddi_id}) has has_structure = 1 but "
                f"structures.h5 holds neither {pfam_a}_{pfam_b}/{inst_a}__{inst_b} nor the swapped key"
            )
        # The same instance pair can only describe one complex; a second
        # membership row naming it (other orientation) adds nothing.
        if frozenset((inst_a, inst_b)) in seen:
            continue
        seen.add(frozenset((inst_a, inst_b)))
        resolved.append((pfam_a, pfam_b, inst_a, inst_b, group, dataset, swapped))

    ddi_by_label = defaultdict(lambda: [0, 0])
    for negative, covered in ddis.values():
        ddi_by_label[negative][0] += 1
        ddi_by_label[negative][1] += covered
    for what, counts in (("membership pairs", pairs), ("DDIs", ddi_by_label)):
        total = sum(c[0] for c in counts.values())
        covered = sum(c[1] for c in counts.values())
        print(
            f"structure coverage: {what} {covered}/{total} "
            f"(positive {counts[0][1]}/{counts[0][0]}, negative {counts[1][1]}/{counts[1][0]})"
        )
    if not resolved:
        raise ValueError("this split has no membership pair with a structure (has_structure = 1)")
    return resolved


def parse_complex(blob, label):
    """Parse one stored complex and return its validated (chain A, chain B)."""
    text = gzip.decompress(np.asarray(blob, dtype=np.uint8).tobytes()).decode("utf-8")
    structure = PDBParser(QUIET=True).get_structure(label, io.StringIO(text))
    model = next(iter(structure), None)
    chain_ids = sorted(chain.id for chain in model) if model is not None else []
    if chain_ids != list(CHAINS):
        raise ValueError(f"{label}: expected exactly chains A and B, found {chain_ids}")
    for chain in model:
        residues = list(chain)
        if not residues:
            raise ValueError(f"{label}: chain {chain.id} has no residues")
        for res in residues:
            if not is_aa(res, standard=True) or res.id[0] != " ":
                raise ValueError(f"{label}: chain {chain.id} residue {res.id} {res.get_resname()} is not a standard amino acid")
            missing = [name for name in BACKBONE if name not in res]
            if missing:
                raise ValueError(f"{label}: chain {chain.id} residue {res.id} lacks backbone atom(s) {missing}")
    return model["A"], model["B"]


def atom_depth(coords, accessible):
    """Distance of each atom to the nearest solvent-accessible atom (0 if accessible)."""
    if not accessible.any():
        raise ValueError("no solvent-accessible atom")
    return np.where(accessible, 0.0, cdist(coords, coords[accessible]).min(axis=1))


def compute_sasa(chain_a, chain_b):
    """Shrake-Rupley per storage chain: {"A": {...}, "B": {...}}.

    Per residue `iso` (chain alone) and `cplx` (inside A+B); per atom `depth`
    in the complex. float64, so the cache reproduces it bit for bit.
    """
    sr = ShrakeRupley()
    out = {}
    for chain in (chain_a, chain_b):
        sr.compute(chain, level="A")
        out[chain.id] = {"iso": np.array([sum(a.sasa for a in res) for res in chain], dtype=np.float64)}
    model = chain_a.get_parent()
    sr.compute(model, level="A")
    atoms = [a for chain in (chain_a, chain_b) for a in chain.get_atoms()]
    coords = np.array([a.coord for a in atoms], dtype=np.float64)
    depth = atom_depth(coords, np.array([a.sasa for a in atoms]) > 0)
    n_a = sum(1 for _ in chain_a.get_atoms())
    for chain, chain_depth in ((chain_a, depth[:n_a]), (chain_b, depth[n_a:])):
        out[chain.id]["cplx"] = np.array([sum(a.sasa for a in res) for res in chain], dtype=np.float64)
        out[chain.id]["depth"] = chain_depth
    return out


class Complex:
    """One membership pair's complex, `first` = the row's instance a.

    `swapped()` exchanges the chains (and their SASA), which is how the
    reverse orientation's vector is computed.
    """

    def __init__(self, label, first, second, sasa_source):
        self.label = label
        self.first, self.second = first, second
        self._sasa_source = sasa_source  # callable -> {"A": ..., "B": ...}
        self._sasa = None

    def sasa(self):
        """(first, second) SASA dicts: keys iso, cplx (per residue), depth (per atom)."""
        if self._sasa is None:
            by_chain = self._sasa_source()
            self._sasa = (by_chain[self.first.id], by_chain[self.second.id])
        return self._sasa

    def swapped(self):
        other = Complex(self.label, self.second, self.first, self._sasa_source)
        if self._sasa is not None:
            other._sasa = self._sasa[::-1]
        return other

    def residues(self):
        return list(self.first) + list(self.second)

    def atoms(self):
        return list(self.first.get_atoms()) + list(self.second.get_atoms())

    def totals(self):
        """(total SASA first alone, second alone, complex)."""
        s1, s2 = self.sasa()
        return s1["iso"].sum(), s2["iso"].sum(), s1["cplx"].sum() + s2["cplx"].sum()

    def fail(self, message):
        raise ValueError(f"{self.label}: {message}")


def cached_sasa(cache: h5py.File, group, dataset, chain_a, chain_b, label):
    if group not in cache or dataset not in cache[group]:
        raise ValueError(f"{label}: SASA cache has no entry {group}/{dataset}")
    entry = cache[group][dataset]
    out = {}
    for chain in (chain_a, chain_b):
        values = {key: entry[chain.id][key][()] for key in ("iso", "cplx", "depth")}
        n_res, n_atoms = len(chain), sum(1 for _ in chain.get_atoms())
        if len(values["iso"]) != n_res or len(values["cplx"]) != n_res or len(values["depth"]) != n_atoms:
            raise ValueError(f"{label}: SASA cache entry for chain {chain.id} does not match the structure")
        out[chain.id] = values
    return out


def iter_structures(conn, structures, sasa_cache=None):
    """Yield one `Complex` per membership pair with a structure, sorted by instance pair."""
    with h5py.File(structures, "r") as struct_h5:
        cache = h5py.File(sasa_cache, "r") if sasa_cache is not None else None
        try:
            for pfam_a, pfam_b, inst_a, inst_b, group, dataset, swapped in structure_rows(conn, struct_h5):
                label = f"{pfam_a}_{pfam_b}/{inst_a}_{inst_b}"
                chain_a, chain_b = parse_complex(struct_h5[group][dataset][()], label)
                if cache is None:
                    def source(a=chain_a, b=chain_b):
                        return compute_sasa(a, b)
                else:
                    def source(a=chain_a, b=chain_b, g=group, d=dataset, lbl=label):
                        return cached_sasa(cache, g, d, a, b, lbl)
                first, second = (chain_b, chain_a) if swapped else (chain_a, chain_b)
                yield (pfam_a, pfam_b, inst_a, inst_b), Complex(label, first, second, source)
        finally:
            if cache is not None:
                cache.close()


def make_extractor(encode):
    """The `extract_features` entrypoint for one `encode(cx, **params)`."""

    def extract_features(conn, out_file, seed, *, structures, sasa_cache=None, **params):
        # seed is part of the encoder ABI; every structure encoder is deterministic.
        out_file.attrs["layout"] = "interaction"
        n = 0
        for (pfam_a, pfam_b, inst_a, inst_b), cx in iter_structures(conn, structures, sasa_cache):
            def vector(c):
                vec = np.asarray(encode(c, **params), dtype=np.float64).ravel()
                if vec.size == 0 or not np.all(np.isfinite(vec)):
                    cx.fail(f"{encode.__module__} produced a non-finite or empty vector")
                return vec

            if inst_a == inst_b:
                # An instance paired with itself: both orientations are one key,
                # and two AF3 copies of one chain need not be identical, so a
                # chain-ordered feature is averaged over the two chain orders.
                writes = [((f"{pfam_a}_{pfam_b}", f"{inst_a}_{inst_b}"),
                           (vector(cx) + vector(cx.swapped())) / 2)]
            else:
                writes = [((f"{pfam_a}_{pfam_b}", f"{inst_a}_{inst_b}"), vector(cx)),
                          ((f"{pfam_b}_{pfam_a}", f"{inst_b}_{inst_a}"), vector(cx.swapped()))]
            for key, vec in writes:
                if key[0] in out_file and key[1] in out_file[key[0]]:
                    # A pair listed in both orientations writes each key twice.
                    if not np.array_equal(out_file[key[0]][key[1]][()], vec.astype(np.float32)):
                        cx.fail("the pair's two membership orientations encode differently")
                    continue
                embeddings.write_instance(out_file, *key, vec)
            n += 1
        print(f"{encode.__module__}: wrote {n} instance pairs in both orientations")

    return extract_features


# ---------------------------------------------------------------- encodings


def pool(values, label="values"):
    """Fixed-size summary of a variable-length vector: mean, std, min, max, median, sum."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError(f"cannot pool an empty {label}")
    return np.array([v.mean(), v.std(), v.min(), v.max(), np.median(v), v.sum()])


def ca_coords(chain):
    return np.array([res["CA"].coord for res in chain], dtype=np.float64)


def resnames(chain):
    return [res.get_resname() for res in chain]


def rsa(chain, sasa):
    return sasa / np.array([MAX_SASA[name] for name in resnames(chain)])


AVG_LEVELS = (2, 4)  # k=1 is M.mean(), derivable from the row-sum mean
MAX_LEVELS = (2,)
SEP_EDGES = (1, 3, 5, 9, 17, 33, 65)  # |i-j| bands [1,3) [3,5) ... [65,inf)


def n_matrix_features(symmetric, banded=False):
    core = 9 if symmetric else 14
    grids = sum(k * (k + 1) // 2 if symmetric else k * k for k in AVG_LEVELS + MAX_LEVELS)
    return core + grids + (len(SEP_EDGES) if banded else 0)


def _adaptive_pool(M, k, reduce, upper_only):
    r, c = M.shape
    P = np.empty((k, k))
    for i in range(k):
        r0, r1 = (i * r) // k, -((-(i + 1) * r) // k)
        for j in range(k):
            c0, c1 = (j * c) // k, -((-(j + 1) * c) // k)
            P[i, j] = reduce(M[r0:r1, c0:c1])
    return P[np.triu_indices(k)] if upper_only else P.ravel()


def encode_matrix(M, symmetric, banded=False):
    """Encode a variable-size 2-D matrix into n_matrix_features(symmetric, banded) floats.

    symmetric: square symmetric map (one row-sum profile, upper-triangle grids);
    otherwise rectangular (row- and column-sum profiles). banded adds the mean
    per |i-j| band, symmetric only; a band no residue pair reaches (chain
    shorter than the band's lower edge) is 0 by construction of the encoding.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.size == 0 or (symmetric and M.shape[0] != M.shape[1]):
        raise ValueError(f"encode_matrix needs a non-empty {'square ' if symmetric else ''}2-D matrix, got {M.shape}")
    r, c = M.shape
    rs = M.sum(axis=1)
    stats5 = [rs.mean(), rs.std(), rs.min(), rs.max(), np.median(rs)]
    if symmetric:
        core = [np.log1p(r), M.std(), M.max(), (M > 0).mean(), *stats5]
    else:
        cs = M.sum(axis=0)
        core = [np.log1p(r), np.log1p(c), M.std(), M.max(), (M > 0).mean(), *stats5,
                cs.std(), cs.min(), cs.max(), np.median(cs)]
    parts = [np.array(core)]
    parts += [_adaptive_pool(M, k, np.mean, symmetric) for k in AVG_LEVELS]
    parts += [_adaptive_pool(M, k, np.max, symmetric) for k in MAX_LEVELS]
    if banded:
        idx = np.arange(r)
        D = np.abs(idx[:, None] - idx[None, :])
        edges = list(SEP_EDGES) + [np.inf]
        band = np.zeros(len(SEP_EDGES))
        for b in range(len(SEP_EDGES)):
            mask = (D >= edges[b]) & (D < edges[b + 1])
            if mask.any():
                band[b] = M[mask].mean()
        parts.append(band)
    return np.concatenate(parts)
