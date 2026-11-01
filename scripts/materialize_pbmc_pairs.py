"""Materialize original PBMC train/test paths from nine deduplicated platform inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


FILES = ("matrix.mtx", "barcodes.tsv", "features.tsv", "metadata.csv")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data/experiments/cross_platform_pbmc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> str:
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid pair or platform name: {name!r}")
    return name


def copy_if_missing(source: Path, destination: Path) -> bool:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
            raise FileExistsError(f"Existing file differs from packaged input: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pbmc_pair_", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        Path(temporary).replace(destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT,
                        help="Extracted cross_platform_pbmc directory")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pair", help="One reference_to_query name from experiments_index.csv")
    selection.add_argument("--all", action="store_true", help="Materialize all 72 directed pairs")
    args = parser.parse_args()

    root = args.data_root.resolve()
    with (root / "experiments_index.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 72 or len({row["name"] for row in pairs}) != 72:
        raise RuntimeError("Expected exactly 72 unique PBMC experiment pairs")
    selected = pairs if args.all else [row for row in pairs if row["name"] == args.pair]
    if not selected:
        raise ValueError(f"Pair not found in experiments_index.csv: {args.pair}")

    copied = 0
    for row in selected:
        pair = safe_name(row["name"])
        reference = safe_name(row["train"])
        query = safe_name(row["test"])
        if pair != f"{reference}_to_{query}":
            raise RuntimeError(f"Invalid pair mapping: {pair}")
        for side, platform in (("train", reference), ("test", query)):
            for filename in FILES:
                source = root / "data/platform_inputs" / platform / filename
                destination = root / "original_paths/data/pairs" / pair / side / filename
                copied += copy_if_missing(source, destination)
        print(f"Ready: {pair}")
    print(f"Materialized {len(selected)} pair(s); copied {copied} file(s).")


if __name__ == "__main__":
    main()
