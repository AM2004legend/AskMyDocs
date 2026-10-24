"""Ingest sample documents at startup so a fresh deployment isn't empty.

    python -m app.preload samples            # every supported file in samples/
    python -m app.preload samples --map      # also build the document map (needs an LLM key)

Safe to run on every start: a file that is already in the database (same SHA-256) is skipped.
"""

import argparse
import shutil
from pathlib import Path

from app import db
from app.ingest.parsers import SUPPORTED_TYPES
from app.ingest.pipeline import UPLOAD_DIR, create_document, find_by_hash, process_document, sha256_of


def preload(folder: Path, build_map: bool = False) -> list[str]:
    db.init_schema()
    report = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower().lstrip(".") in SUPPORTED_TYPES):
        data = path.read_bytes()
        if find_by_hash(sha256_of(data)):
            report.append(f"skip   {path.name} (already loaded)")
            continue
        file_type = path.suffix.lower().lstrip(".")
        doc = create_document(path.name, file_type, sha256_of(data))
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        work_copy = UPLOAD_DIR / f"{doc['id']}.{file_type}"  # the pipeline deletes its input when done
        shutil.copyfile(path, work_copy)
        process_document(doc["id"], work_copy, file_type, build_map=build_map)
        report.append(f"loaded {path.name} as document {doc['id']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--map", action="store_true", help="also build the document map (uses LLM quota)")
    args = parser.parse_args()
    if not args.folder.is_dir():
        print(f"preload: {args.folder} not found, nothing to do")
        return
    for line in preload(args.folder, build_map=args.map):
        print(f"preload: {line}")
    db.close_pool()


if __name__ == "__main__":
    main()
