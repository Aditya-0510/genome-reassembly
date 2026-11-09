from __future__ import annotations

import json
from pathlib import Path


def load_chrom_maps(sequence_report_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Build chromosome-name lookups from an NCBI datasets sequence_report.jsonl.

    Returns (ucsc_name_to_refseq, bare_chr_name_to_refseq), e.g.
    {"chr1": "NC_000001.11", ...}, {"1": "NC_000001.11", ...}
    """
    ucsc_to_refseq: dict[str, str] = {}
    chrname_to_refseq: dict[str, str] = {}

    with Path(sequence_report_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("role") != "assembled-molecule":
                continue  # skip unlocalized/unplaced scaffolds that reuse the same chrName
            refseq = record.get("refseqAccession")
            if not refseq:
                continue
            ucsc_name = record.get("ucscStyleName")
            chr_name = record.get("chrName")
            if ucsc_name:
                ucsc_to_refseq[ucsc_name] = refseq
            if chr_name:
                chrname_to_refseq[chr_name] = refseq

    return ucsc_to_refseq, chrname_to_refseq
