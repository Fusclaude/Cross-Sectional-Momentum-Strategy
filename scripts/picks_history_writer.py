"""
Idempotent writer for data/picks_history.jsonl

Problem this fixes:
  The old code did `open(path, "a").write(json.dumps(row))`, so every workflow
  re-run (manual dispatch, retry after a merge conflict, etc.) appended another
  row for the same date instead of replacing it. That's what caused 5 duplicate
  2026-06-20 rows in the file, and it's also what makes git merges conflict
  destructively (both sides "add" different tail lines).

Fix:
  Load existing rows into a dict keyed by date, overwrite/insert today's row,
  then rewrite the whole file sorted by date. Re-running the script twice for
  the same date now produces an identical file (idempotent), so there's
  nothing for git to conflict about, and if a conflict ever does happen the
  file can be safely merged by unioning + re-running this dedup pass.
"""
import json
import sys
from pathlib import Path


def load_rows(path: Path) -> dict:
    """Read existing picks_history.jsonl into {date: row}. Last row per date wins."""
    rows = {}
    if path.exists():
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[row["date"]] = row
    return rows


def write_rows(path: Path, rows: dict) -> None:
    """Write rows back out, one per date, sorted ascending by date."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for date in sorted(rows.keys()):
            f.write(json.dumps(rows[date], separators=(",", ":")) + "\n")


def upsert_picks(path: Path, new_row: dict) -> None:
    """Insert or replace the row for new_row['date'], then rewrite the file."""
    rows = load_rows(path)
    rows[new_row["date"]] = new_row
    write_rows(path, rows)


def merge_files(*paths: Path) -> dict:
    """
    Union multiple picks_history.jsonl files (e.g. 'ours' vs 'theirs' from a
    git conflict) into one deduped {date: row} dict. Used by the conflict
    resolution helper below — not needed in normal operation once this
    writer is in place, since duplicate commits stop happening.
    """
    merged: dict = {}
    for p in paths:
        for date, row in load_rows(p).items():
            merged[date] = row  # last file wins on collision
    return merged


if __name__ == "__main__":
    # CLI helper for the merge-conflict case:
    #   python picks_history_writer.py merge out.jsonl ours.jsonl theirs.jsonl
    if len(sys.argv) >= 2 and sys.argv[1] == "merge":
        out_path = Path(sys.argv[2])
        in_paths = [Path(p) for p in sys.argv[3:]]
        merged = merge_files(*in_paths)
        write_rows(out_path, merged)
        print(f"Merged {len(in_paths)} files -> {out_path} ({len(merged)} unique dates)")
    else:
        print(__doc__)
