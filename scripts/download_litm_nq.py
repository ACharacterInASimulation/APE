#!/usr/bin/env python
"""Download the prebuilt Lost-in-the-Middle NQ multi-document QA files."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

POSITION_FILES = {
    10: {"start": [0], "middle": [4], "end": [9], "all": [0, 4, 9]},
    20: {"start": [0], "middle": [9], "end": [19], "all": [0, 4, 9, 14, 19]},
    30: {"start": [0], "middle": [14], "end": [29], "all": [0, 4, 9, 14, 19, 24, 29]},
}

RAW_BASE = "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data"
STANFORD_RETRIEVAL_URL = "https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz"


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def download(url: str, output_path: Path, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"exists {output_path}")
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"download {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/litm_nq")
    parser.add_argument("--document-counts", default="10,20,30")
    parser.add_argument(
        "--positions",
        default="start,middle,end",
        help="Comma-separated from start,middle,end,all or explicit gold indices.",
    )
    parser.add_argument("--download-retrieval-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    counts = parse_csv_ints(args.document_counts)
    position_keys = [part.strip() for part in args.positions.split(",") if part.strip()]
    manifest = []

    for doc_count in counts:
        if doc_count not in POSITION_FILES:
            raise ValueError(f"Unsupported document count {doc_count}; choose from {sorted(POSITION_FILES)}")
        positions = []
        for key in position_keys:
            if key in POSITION_FILES[doc_count]:
                positions.extend(POSITION_FILES[doc_count][key])
            else:
                positions.append(int(key))
        for gold_idx in sorted(set(positions)):
            name = f"nq-open-{doc_count}_total_documents_gold_at_{gold_idx}.jsonl.gz"
            url = f"{RAW_BASE}/{doc_count}_total_documents/{name}"
            output_path = output_dir / f"{doc_count}_total_documents" / name
            download(url, output_path, force=args.force)
            manifest.append(
                {
                    "document_count": doc_count,
                    "gold_index": gold_idx,
                    "path": str(output_path),
                    "url": url,
                }
            )

    if args.download_retrieval_source:
        retrieval_path = output_dir / "nq-open-contriever-msmarco-retrieved-documents.jsonl.gz"
        download(STANFORD_RETRIEVAL_URL, retrieval_path, force=args.force)
        manifest.append({"kind": "retrieval_source", "path": str(retrieval_path), "url": STANFORD_RETRIEVAL_URL})

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"manifest {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
