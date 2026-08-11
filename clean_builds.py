"""Clean older training runs and keep the most recent.

Usage examples:
  python clean_builds.py --project runs/train --keep 1 --yes
  python clean_builds.py --project runs/train --keep 2 --dry-run

This script is conservative by default. Use `--yes` to actually delete.
"""
import argparse
import shutil
from pathlib import Path
import sys


def find_run_dirs(project_dir: Path):
    if not project_dir.exists():
        return []
    return [p for p in project_dir.iterdir() if p.is_dir()]


def sorted_by_mtime(dirs):
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def remove_paths(paths, dry_run=True):
    for p in paths:
        if dry_run:
            print(f"Would remove: {p}")
        else:
            print(f"Removing: {p}")
            shutil.rmtree(p)


def copy_best_from_run(run_dir: Path, dest_dir: Path, dry_run=True):
    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        return None
    candidates = sorted(weights_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    best = candidates[0]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / best.name
    if dry_run:
        print(f"Would copy {best} -> {dest}")
    else:
        print(f"Copying {best} -> {dest}")
        shutil.copy2(best, dest)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("runs/train"), help="Project runs directory")
    parser.add_argument("--keep", type=int, default=1, help="Number of most recent runs to keep")
    parser.add_argument("--yes", action="store_true", help="Actually delete files (default is dry-run)")
    parser.add_argument("--backup-model-dir", type=Path, default=Path("model"), help="Directory to copy the latest weights to")
    args = parser.parse_args()

    project_dir: Path = args.project
    keep = max(0, args.keep)
    dry_run = not args.yes

    run_dirs = find_run_dirs(project_dir)
    if not run_dirs:
        print(f"No run directories found in {project_dir}")
        return

    sorted_runs = sorted_by_mtime(run_dirs)
    to_keep = sorted_runs[:keep]
    to_remove = sorted_runs[keep:]

    print(f"Found {len(sorted_runs)} runs. Keeping {len(to_keep)} most recent.")
    for p in to_keep:
        print(f"Keeping: {p}")

    # Optionally copy best weights from newest kept run to model/
    if to_keep:
        latest = to_keep[0]
        copy_best_from_run(latest, args.backup_model_dir, dry_run=dry_run)

    if to_remove:
        print("Will remove these run dirs:")
        for p in to_remove:
            print(f" - {p}")
        remove_paths(to_remove, dry_run=dry_run)
    else:
        print("No runs to remove.")

    print("Done.")


if __name__ == '__main__':
    main()
