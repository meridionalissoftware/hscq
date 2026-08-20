#!/usr/bin/env python3
import os
import shutil
import sys

# Resolve paths against the repo root (parent of scripts/), so the installer
# works regardless of the caller's cwd.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, "src", "hsc")

default_dest_dir = "/usr/include/hsc"

HEADER_EXTS = (".h", ".hh", ".hpp")


def ensure_dir(path):
    if not os.path.isdir(path):
        print("location doesnt exist, creating " + path, file=sys.stderr)
        os.makedirs(path, exist_ok=True)


def install_tree(src, dest, extra_names=()):
    """Copy every header (matched by extension or explicit name) under `src`
    into `dest`, preserving the relative directory layout."""
    src = os.path.realpath(src)
    ensure_dir(dest)
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(HEADER_EXTS) or f in extra_names:
                src_file = os.path.join(root, f)
                rel_path = os.path.relpath(src_file, src)
                dest_file = os.path.join(dest, rel_path)
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy2(src_file, dest_file)
                os.chmod(dest_file, 0o644)


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else default_dest_dir

    if os.geteuid() != 0:
        print("Must be root", file=sys.stderr)
        sys.exit(1)

    # hsc headers -> <dest> (default /usr/include/hsc), so <hsc/...> resolves.
    install_tree(src_path, dest)
    print("installed hsc headers -> " + dest)


if __name__ == "__main__":
    main()
