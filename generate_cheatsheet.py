#!/usr/bin/env python3
"""
generate_cheatsheet.py  —  CLI-Wrapper fuer modules/cheatsheet.py

Fuehrt einen Full Rebuild des Cheatsheets aus allen perfect_matches durch.
Erzeugt:
  - data/cheatsheet.md           (Markdown fuer Menschen)
  - data/cheatsheet_cache.json   (JSON-Cache fuer die Pipeline)

Aufruf:
    python generate_cheatsheet.py [--pm-dir output/perfect_matches/nonmatchings]
                                  [--out data/cheatsheet.md]
"""

import argparse
from pathlib import Path
from modules.cheatsheet import (
    full_rebuild, save_cache, generate_cheatsheet_md,
    generate_prompt_context, CACHE_PATH,
)


def main():
    ap = argparse.ArgumentParser(description="Generate decompilation cheatsheet from perfect matches")
    ap.add_argument("--pm-dir", default="output/perfect_matches/nonmatchings",
                    help="Path to perfect_matches/nonmatchings directory")
    ap.add_argument("--out", default="data/cheatsheet.md",
                    help="Output path for the cheatsheet")
    args = ap.parse_args()

    pm_dir = Path(args.pm_dir)
    if not pm_dir.exists():
        print(f"ERROR: {pm_dir} not found.")
        return

    print(f"Scanning {pm_dir} ...")
    results = full_rebuild(str(pm_dir))

    print("Generating cheatsheet ...")
    cheatsheet = generate_cheatsheet_md(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cheatsheet, encoding="utf-8")

    save_cache(results, CACHE_PATH)

    # Beispiel-Prompt-Kontext
    sample_result = generate_prompt_context(results, "bsbtrot_entrypoint_3", "bs")
    sample_ctx = sample_result.get("cheatsheet", "") or sample_result.get("similar", "")
    sample_out = out.with_suffix(".prompt_sample.txt")
    sample_out.write_text(sample_ctx, encoding="utf-8")

    print(f"\nCheatsheet written to {out}")
    print(f"Cache written to {CACHE_PATH}")
    print(f"Sample prompt context written to {sample_out}")
    print(f"  {results['total_files']} total files analyzed")
    print(f"  {len(results['entrypoint_slots'])} entrypoint slot patterns")
    print(f"  {len(results['known_signatures'])} unique function signatures")
    print(f"  {sum(results['overlay_prefix_stats'].values())} overlay functions across "
          f"{len(results['overlay_prefix_stats'])} prefixes")


if __name__ == "__main__":
    main()
