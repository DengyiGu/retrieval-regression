"""Check that a source distribution contains no untracked local project files."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {
    "PKG-INFO",
    "setup.cfg",
    *(
        f"src/retrieval_regression.egg-info/{name}"
        for name in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        )
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.archive.is_dir():
        (args.archive,) = args.archive.glob("*.tar.gz")
    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    )
    included = set()
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                raise ValueError(f"unexpected archive member: {member.name}")
            if member.isfile():
                included.add(str(PurePosixPath(*path.parts[1:])))
    unexpected = included - tracked - GENERATED
    required = {"README.md", "LICENSE", "pyproject.toml"} | {
        path for path in tracked if path.startswith("src/retrieval_regression/")
    }
    if unexpected or required - included:
        raise ValueError(
            f"untracked archive files: {sorted(unexpected)}; "
            f"missing required files: {sorted(required - included)}"
        )
    print(f"PASS: {len(included)} source-distribution files checked against the Git index")


if __name__ == "__main__":
    main()
