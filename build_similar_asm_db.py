#!/usr/bin/env python3
"""
build_similar_asm_db.py
=======================
Standalone-Script: Baut eine JSONL-Datenbank, die fuer jede Target-ASM-Funktion
den strukturell aehnlichsten Referenz-C-Code findet.

Quellen fuer Referenz-C-Code:
  1. output/perfect_matches/nonmatchings/**/*.c
     -> Hex wird aus der zugehoerigen target .s gelesen (kein Compile noetig!)
     -> C-Code wird aus der .c Datei gelesen
  2. data/refdata_original/*.c
     -> Wird mit IDO kompiliert, Hex aus .o extrahiert
     -> Header stehen schon im Code, .h Dateien liegen im selben Ordner

Target-ASM: data/target_asm/nonmatchings/**/*.s

Ablauf:
  Phase 1: Hex-Cache fuer alle Referenzen aufbauen (einmalig)
  Phase 2: Hex aller Target-ASMs extrahieren
  Phase 3: Vergleich & beste Matches finden

Usage:
  python3 build_similar_asm_db.py                    # Full run (auto-detect workers)
  python3 build_similar_asm_db.py -j4                 # 4 parallele Prozesse
  python3 build_similar_asm_db.py --test 20           # Test mit 20 Targets
  python3 build_similar_asm_db.py --rebuild-cache     # Nur Hex-Cache neu bauen
  python3 build_similar_asm_db.py --skip-refdata      # Nur perfect_matches als Referenz

Output: data/similar_asm_db.jsonl
Cache:  data/similar_asm_hex_cache.jsonl
"""

import os
import re
import sys
import json
import time
import signal
import shutil
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:
        """Minimaler tqdm-Fallback: kein Balken, nur Log-Zeilen."""
        def __init__(self, iterable=None, total=None, desc="", unit="it", leave=True, **kw):
            self._it = iterable
            self._total = total or (len(iterable) if iterable and hasattr(iterable, '__len__') else None)
            self._desc = desc
            self._n = 0
            self._postfix = {}
        def __iter__(self):
            for item in self._it:
                yield item
                self._n += 1
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
        def update(self, n=1): self._n += n
        def set_postfix(self, refresh=True, **kw): self._postfix.update(kw)
        def close(self):
            pf = " ".join(f"{k}={v}" for k, v in self._postfix.items())
            print(f"  {self._desc}: {self._n}" + (f"/{self._total}" if self._total else "") + (f"  [{pf}]" if pf else ""))
        @staticmethod
        def write(s, **kw): print(s)

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from modules.ido_comparison import (
    extract_hex_from_original_s,
    mask_mips_hex_advanced,
)

# Logging: nur Datei-Handler auf INFO, Konsole bleibt ruhig (tqdm uebernimmt)
_file_handler = logging.FileHandler(_SCRIPT_DIR / "build_similar_asm.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PFADE
# ══════════════════════════════════════════════════════════════════════════════

TARGET_ASM_DIR = _SCRIPT_DIR / "data" / "target_asm" 
PERFECT_MATCHES_DIR = _SCRIPT_DIR / "output" / "perfect_matches" / "nonmatchings"
REFDATA_DIR = _SCRIPT_DIR / "data" / "refdata_original"

HEX_CACHE_PATH = _SCRIPT_DIR / "data" / "similar_asm_hex_cache.jsonl"
OUTPUT_DB_PATH = _SCRIPT_DIR / "data" / "similar_asm_db.jsonl"

# IDO Compiler Pfade (fuer refdata Kompilierung)
IDO_DIR = _SCRIPT_DIR / "IDO_compiler"
IDO_CC = IDO_DIR / "cc"
HEADER_ROOT = _SCRIPT_DIR / "data" / "header"

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Laengenfilter: Nur vergleichen wenn Instruktionsanzahl im Verhaeltnis liegt
LENGTH_RATIO_MIN = 0.4
LENGTH_RATIO_MAX = 2.5

# Minimale Instruktionsanzahl fuer sinnvollen Vergleich
MIN_INSTRUCTIONS = 3


# ══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def _run_cmd(cmd, cwd=None, env=None, timeout=10):
    """Subprocess mit Timeout und Process-Group-Kill."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return -1, b"", b"Timeout"
    except Exception as e:
        return -1, b"", str(e).encode()


def _build_include_dirs():
    """Sammelt Include-Pfade fuer GCC Preprocessing."""
    dirs = []
    if HEADER_ROOT.is_dir():
        dirs.append(str(HEADER_ROOT))
        for root, subdirs, files in os.walk(HEADER_ROOT):
            if any(f.endswith(".h") for f in files):
                if root not in dirs:
                    dirs.append(root)

    for extra in [
        IDO_DIR / "include",
        IDO_DIR / "src",
        IDO_DIR / "include" / "PR",
        IDO_DIR / "lib" / "ultralib" / "include",
    ]:
        if extra.is_dir() and str(extra) not in dirs:
            dirs.append(str(extra))

    return dirs


INCLUDE_DIRS = _build_include_dirs()


def func_key_name(name: str) -> str:
    """Normalisiert Funktionsnamen (ohne .c/.s Suffix)."""
    for suffix in (".c", ".s", ".o"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def compile_refdata(c_path: Path) -> list:
    """
    Kompiliert eine refdata_original C-Datei mit IDO und extrahiert Hex-Opcodes.

    refdata_original Dateien haben ihre #include Zeilen schon drin,
    und die .h Dateien liegen im selben Ordner (data/refdata_original/).

    Returns: Liste von Hex-Strings oder leere Liste bei Fehler.
    """
    func_name = c_path.stem
    c_code = c_path.read_text(encoding="utf-8", errors="replace")

    temp_dir = tempfile.mkdtemp(prefix=f"simasm_{func_name}_")
    c_file = os.path.join(temp_dir, f"{func_name}.c")
    i_file = os.path.join(temp_dir, f"{func_name}.i")
    o_file = os.path.join(temp_dir, f"{func_name}.o")

    try:
        with open(c_file, "w", encoding="utf-8") as f:
            f.write(c_code)

        # GCC Preprocessing — refdata_original Ordner als extra Include-Pfad
        gcc_cmd = [
            "gcc", "-E", "-P", "-xc",
            "-D_LANGUAGE_C", "-D_MIPS_SZLONG=32",
            "-D__attribute__(x)=", "-D__extension__=",
            f"-I{REFDATA_DIR}",
        ]
        for inc in INCLUDE_DIRS:
            gcc_cmd.extend(["-I", inc])
        gcc_cmd.extend([c_file, "-o", i_file])

        rc, _, stderr = _run_cmd(gcc_cmd, timeout=5)
        if rc != 0:
            return []

        # IDO Compile
        env = os.environ.copy()
        env["COMPILER_PATH"] = str(IDO_DIR)
        env["LD_LIBRARY_PATH"] = f"{IDO_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

        ido_cmd = [
            str(IDO_CC), "-c", "-O2", "-mips2", "-G", "0",
            "-non_shared", "-w", i_file, "-o", o_file,
        ]
        rc, _, stderr = _run_cmd(ido_cmd, cwd=temp_dir, env=env, timeout=10)

        if rc != 0 or not os.path.exists(o_file):
            return []

        # Hex extrahieren via objdump
        objdump_cmd = ["mips-linux-gnu-objdump", "-d", "-z", o_file]
        try:
            result = subprocess.run(objdump_cmd, capture_output=True, text=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        hex_codes = []
        for line in result.stdout.splitlines():
            match = re.match(r"\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+", line)
            if match:
                hex_codes.append(match.group(1).lower())
        return hex_codes

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER-FUNKTIONEN (Module-Level fuer multiprocessing Pickling)
# ══════════════════════════════════════════════════════════════════════════════

def _worker_compile_refdata(c_path_str: str):
    """
    Pool-Worker: Kompiliert eine refdata-Datei und gibt den Cache-Eintrag zurueck.
    Laeuft in einem eigenen Prozess mit eigenem temp-dir.
    Returns: dict (Cache-Eintrag) oder None bei Fehler.
    """
    c_path = Path(c_path_str)
    func_name = c_path.stem

    hex_codes = compile_refdata(c_path)
    if len(hex_codes) < MIN_INSTRUCTIONS:
        return None

    c_code = c_path.read_text(encoding="utf-8", errors="replace").strip()
    hex_masked = [mask_mips_hex_advanced(h) for h in hex_codes]

    return {
        "func_name": func_key_name(func_name),
        "cache_key": f"ref_{func_name}",
        "source": "refdata_original",
        "c_path": str(c_path.relative_to(_SCRIPT_DIR)),
        "hex_masked": hex_masked,
        "hex_count": len(hex_masked),
        "c_code": c_code,
    }


# Globale Referenz-Liste fuer Phase-3-Worker (via fork COW geteilt)
_worker_refs = None


def _init_comparison_worker(refs):
    """Initializer fuer Phase-3-Pool: setzt die Referenz-Liste im Worker-Prozess."""
    global _worker_refs
    _worker_refs = refs


def _worker_compare_target(target: dict) -> dict:
    """
    Pool-Worker: Vergleicht ein Target gegen alle Referenzen.
    Laeuft in einem eigenen Prozess, liest _worker_refs via COW.

    Returns: dict mit result, stats
    """
    t_name = target["func_name"]
    t_hex = target["hex_masked"]
    t_count = target["hex_count"]

    best_score = 0.0
    best_ref = None
    comparisons = 0
    length_skipped = 0
    self_skipped = 0

    for ref in _worker_refs:
        r_name = ref["func_name"]
        r_count = ref["hex_count"]

        # Self-Skip
        if r_name == t_name:
            self_skipped += 1
            continue

        # Laengenfilter
        if t_count > 0 and r_count > 0:
            ratio = r_count / t_count
            if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
                length_skipped += 1
                continue

        # Struct-Score
        score = SequenceMatcher(None, t_hex, ref["hex_masked"]).ratio() * 100.0
        comparisons += 1

        if score > best_score:
            best_score = score
            best_ref = ref
            # Early termination bei 100%
            if score >= 99.99:
                break

    result = None
    if best_ref and best_score > 0:
        result = {
            "target_asm": t_name,
            "target_asm_path": target["s_path"],
            "target_hex_count": t_count,
            "best_ref": best_ref["func_name"],
            "best_ref_source": best_ref["source"],
            "best_ref_path": best_ref["c_path"],
            "struct_score": round(best_score, 2),
            "ref_hex_count": best_ref["hex_count"],
            "c_code": best_ref["c_code"],
        }

    return {
        "result": result,
        "comparisons": comparisons,
        "length_skipped": length_skipped,
        "self_skipped": self_skipped,
        "perfect": best_score >= 99.99,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: HEX-CACHE AUFBAUEN
# ══════════════════════════════════════════════════════════════════════════════

def _load_hex_cache() -> dict:
    """Laedt den bestehenden Hex-Cache."""
    cache = {}
    if HEX_CACHE_PATH.exists():
        with open(HEX_CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry.get("cache_key", entry["func_name"])
                    cache[key] = entry
                except (json.JSONDecodeError, KeyError):
                    pass
    return cache


def _save_hex_cache(cache: dict):
    """Schreibt den Hex-Cache."""
    HEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HEX_CACHE_PATH, "w", encoding="utf-8") as f:
        for entry in cache.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_hex_cache_perfect_matches(cache: dict) -> dict:
    """
    Phase 1a: Hex-Cache fuer perfect_matches aufbauen.
    Kein Compile noetig — Hex wird aus der zugehoerigen target .s gelesen.
    """
    if not PERFECT_MATCHES_DIR.exists():
        log.warning(f"Perfect matches Ordner nicht gefunden: {PERFECT_MATCHES_DIR}")
        return cache

    c_files = sorted(PERFECT_MATCHES_DIR.rglob("*.c"))
    new_count = 0
    skip_count = 0

    log.info(f"[Phase 1a] Perfect Matches: {len(c_files)} Dateien")

    pbar = tqdm(c_files, desc="Phase 1a │ perfect_matches", unit="fn", leave=True)
    for c_path in pbar:
        func_name = c_path.stem
        cache_key = f"pm_{func_name}"

        if cache_key in cache:
            skip_count += 1
            continue

        rel_path = c_path.relative_to(PERFECT_MATCHES_DIR)
        s_path = TARGET_ASM_DIR / rel_path.with_suffix(".s")

        if not s_path.exists():
            continue

        hex_codes = extract_hex_from_original_s(str(s_path))
        if len(hex_codes) < MIN_INSTRUCTIONS:
            continue

        c_code = c_path.read_text(encoding="utf-8", errors="replace").strip()
        hex_masked = [mask_mips_hex_advanced(h) for h in hex_codes]

        cache[cache_key] = {
            "func_name": func_key_name(func_name),
            "cache_key": cache_key,
            "source": "perfect_matches",
            "c_path": str(c_path.relative_to(_SCRIPT_DIR)),
            "hex_masked": hex_masked,
            "hex_count": len(hex_masked),
            "c_code": c_code,
        }
        new_count += 1
        pbar.set_postfix(neu=new_count, cache=skip_count, refresh=False)

    pbar.close()
    log.info(f"[Phase 1a] Fertig: {new_count} neu, {skip_count} aus Cache")
    return cache


def build_hex_cache_refdata(cache: dict, workers: int = 1) -> dict:
    """
    Phase 1b: Hex-Cache fuer refdata_original aufbauen.
    Kompiliert jede .c Datei mit IDO — parallelisiert mit Pool.
    """
    if not REFDATA_DIR.exists():
        log.warning(f"Refdata Ordner nicht gefunden: {REFDATA_DIR}")
        return cache

    c_files = sorted(REFDATA_DIR.glob("*.c"))

    # Nur Dateien die noch nicht im Cache sind
    todo = []
    skip_count = 0
    for c_path in c_files:
        cache_key = f"ref_{c_path.stem}"
        if cache_key in cache:
            skip_count += 1
        else:
            todo.append(str(c_path))

    log.info(
        f"[Phase 1b] Refdata: {len(c_files)} gesamt, "
        f"{len(todo)} zu kompilieren, {skip_count} aus Cache"
    )

    if not todo:
        return cache

    new_count = 0
    fail_count = 0
    save_interval = 200

    desc = f"Phase 1b │ compile refdata"
    if workers > 1:
        desc += f" ({workers}w)"

    pbar = tqdm(total=len(todo), desc=desc, unit="fn", leave=True)

    if workers <= 1:
        # ── Single-Process ──
        for c_path_str in todo:
            entry = _worker_compile_refdata(c_path_str)
            if entry:
                cache[entry["cache_key"]] = entry
                new_count += 1
            else:
                fail_count += 1

            pbar.update(1)
            pbar.set_postfix(ok=new_count, fehl=fail_count, refresh=False)

            if (new_count + fail_count) % save_interval == 0:
                _save_hex_cache(cache)
    else:
        # ── Multi-Process ──
        pool = Pool(workers)
        try:
            for entry in pool.imap_unordered(
                _worker_compile_refdata, todo, chunksize=8
            ):
                if entry:
                    cache[entry["cache_key"]] = entry
                    new_count += 1
                else:
                    fail_count += 1

                pbar.update(1)
                pbar.set_postfix(ok=new_count, fehl=fail_count, refresh=False)

                if (new_count + fail_count) % save_interval == 0:
                    _save_hex_cache(cache)
        except KeyboardInterrupt:
            pbar.close()
            tqdm.write("\nCtrl+C — speichere bisherigen Cache...")
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()

    pbar.close()
    log.info(f"[Phase 1b] Fertig: {new_count} neu, {skip_count} Cache, {fail_count} fehl")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TARGET-ASM HEX EXTRAHIEREN
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_target_hex() -> list:
    """
    Extrahiert Hex aus allen Target-ASM-Dateien.
    (Kein Compile, nur .s Parsen — schnell genug single-threaded)
    """
    if not TARGET_ASM_DIR.exists():
        log.error(f"Target ASM Ordner nicht gefunden: {TARGET_ASM_DIR}")
        return []

    s_files = sorted(TARGET_ASM_DIR.rglob("*.s"))
    targets = []

    log.info(f"[Phase 2] Target ASMs: {len(s_files)} Dateien")

    pbar = tqdm(s_files, desc="Phase 2  │ extract targets ", unit="fn", leave=True)
    for s_path in pbar:
        func_name = func_key_name(s_path.stem)
        hex_codes = extract_hex_from_original_s(str(s_path))

        if len(hex_codes) < MIN_INSTRUCTIONS:
            continue

        hex_masked = [mask_mips_hex_advanced(h) for h in hex_codes]

        targets.append({
            "func_name": func_name,
            "s_path": str(s_path.relative_to(_SCRIPT_DIR)),
            "hex_masked": hex_masked,
            "hex_count": len(hex_masked),
        })

        pbar.set_postfix(valid=len(targets), refresh=False)

    pbar.close()
    log.info(f"[Phase 2] Fertig: {len(targets)} Targets mit >= {MIN_INSTRUCTIONS} Instruktionen")
    return targets


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: VERGLEICH & BEST MATCHES
# ══════════════════════════════════════════════════════════════════════════════

def find_best_matches(
    targets: list,
    ref_cache: dict,
    existing_db: dict = None,
    workers: int = 1,
) -> dict:
    """
    Phase 3: Fuer jedes Target den besten Referenz-Match finden.

    Bei workers > 1: Targets werden parallel ueber einen Pool verteilt.
    Die Referenz-Liste wird via fork() COW geteilt (kein extra RAM).
    """
    refs = list(ref_cache.values())
    results = existing_db if existing_db else {}

    # Nur Targets die noch nicht berechnet sind
    todo = [t for t in targets if t["func_name"] not in results]
    skipped = len(targets) - len(todo)

    log.info(
        f"[Phase 3] {len(todo)} Targets vs {len(refs)} Referenzen"
        + (f" ({skipped} Resume-skip)" if skipped else "")
    )

    if not todo:
        tqdm.write("Phase 3: Alles aus Resume — nichts zu tun.")
        return results

    stats = {"comparisons": 0, "perfect": 0, "done": 0}
    save_interval = 500

    desc = f"Phase 3  │ find matches    "
    if workers > 1:
        desc = f"Phase 3  │ matching ({workers}w)   "

    pbar = tqdm(total=len(todo), desc=desc, unit="fn", leave=True)

    def _process_result(wr: dict):
        if wr["result"]:
            results[wr["result"]["target_asm"]] = wr["result"]
        stats["comparisons"] += wr["comparisons"]
        if wr["perfect"]:
            stats["perfect"] += 1
        stats["done"] += 1
        pbar.update(1)
        pbar.set_postfix(
            cmp=f'{stats["comparisons"] // 1000}k',
            perf=stats["perfect"],
            refresh=False,
        )

    if workers <= 1:
        # ── Single-Process ──
        global _worker_refs
        _worker_refs = refs

        for target in todo:
            wr = _worker_compare_target(target)
            _process_result(wr)

            if stats["done"] % save_interval == 0:
                _save_output_db(results)
    else:
        # ── Multi-Process ──
        chunksize = max(1, min(50, len(todo) // (workers * 4)))

        pool = Pool(
            workers,
            initializer=_init_comparison_worker,
            initargs=(refs,),
        )
        try:
            for wr in pool.imap_unordered(
                _worker_compare_target, todo, chunksize=chunksize
            ):
                _process_result(wr)

                if stats["done"] % save_interval == 0:
                    _save_output_db(results)
        except KeyboardInterrupt:
            pbar.close()
            tqdm.write("\nCtrl+C — speichere bisherige Ergebnisse...")
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()

    pbar.close()
    log.info(
        f"[Phase 3] Fertig: {stats['comparisons']:,} Vergleiche, "
        f"{stats['perfect']} perfekte Matches"
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _save_output_db(results: dict):
    """Schreibt die Ergebnis-Datenbank."""
    OUTPUT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DB_PATH, "w", encoding="utf-8") as f:
        for entry in results.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_existing_db() -> dict:
    """Laedt bestehende Ergebnisse fuer Resume."""
    db = {}
    if OUTPUT_DB_PATH.exists():
        with open(OUTPUT_DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    db[entry["target_asm"]] = entry
                except (json.JSONDecodeError, KeyError):
                    pass
    return db


def _print_score_block(label: str, entries: list, width: int = 60):
    """Gibt einen Statistik-Block fuer eine Gruppe von Ergebnissen aus."""
    if not entries:
        print(f"\n  {label}: keine Eintraege")
        return

    scores = [r["struct_score"] for r in entries]
    sources = {}
    for r in entries:
        src = r["best_ref_source"]
        sources[src] = sources.get(src, 0) + 1

    buckets = [
        ("90-100%", [s for s in scores if s >= 90]),
        ("70-90%",  [s for s in scores if 70 <= s < 90]),
        ("50-70%",  [s for s in scores if 50 <= s < 70]),
        ("30-50%",  [s for s in scores if 30 <= s < 50]),
        ("<30%",    [s for s in scores if s < 30]),
    ]

    n = len(scores)
    median = sorted(scores)[n // 2]

    print(f"\n  {label} ({n:,} Funktionen)")
    print(f"  {'-' * (width - 4)}")
    print(f"    Durchschn. Score:  {sum(scores)/n:.1f}%")
    print(f"    Median Score:      {median:.1f}%")
    print(f"    Bester / Schlecht: {max(scores):.1f}% / {min(scores):.1f}%")
    print()
    for bucket_label, bucket_scores in buckets:
        cnt = len(bucket_scores)
        bar = "█" * (cnt * 30 // n) if n > 0 else ""
        print(f"    {bucket_label:>8}: {cnt:>5}  {bar}")
    print()
    src_parts = [f"{s}: {c:,}" for s, c in sources.items()]
    print(f"    Quellen: {', '.join(src_parts)}")


def print_summary(results: dict, matched_funcs: set):
    """
    Gibt eine Zusammenfassung der Ergebnisse aus.
    Trennt "offene" Funktionen (noch kein eigener Perfect Match)
    von "bereits geloesten" (haben schon einen Perfect Match).
    """
    if not results:
        print("\nKeine Ergebnisse.")
        return

    # Aufteilen: offen vs. bereits geloest
    offen = []
    geloest = []
    for r in results.values():
        if r["target_asm"] in matched_funcs:
            geloest.append(r)
        else:
            offen.append(r)

    all_scores = [r["struct_score"] for r in results.values()]

    print(f"\n{'='*60}")
    print(f"  SIMILAR-ASM DATABASE — ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print(f"  Eintraege gesamt:      {len(results):,}")
    print(f"  Davon offen:           {len(offen):,}")
    print(f"  Davon bereits geloest: {len(geloest):,}")

    # Hauptblock: nur offene Funktionen (der relevante Teil)
    _print_score_block("OFFENE FUNKTIONEN (noch kein Perfect Match)", offen)

    # Nebenblock: bereits geloeste (zur Vollstaendigkeit)
    _print_score_block("BEREITS GELOEST (haben eigenen Perfect Match)", geloest)

    print(f"\n  Output: {OUTPUT_DB_PATH}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _default_workers() -> int:
    """Sicherer Default: Haelfte der CPUs, min 1, max 6."""
    try:
        n = cpu_count()
    except (NotImplementedError, ValueError):
        n = 2
    return max(1, min(n // 2, 6))


def main():
    parser = argparse.ArgumentParser(
        description="Baut Similar-ASM Datenbank fuer die Uebersetzungspipeline"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=0, metavar="N",
        help=f"Parallele Prozesse (default: auto={_default_workers()}). "
             f"-j1 = single-process."
    )
    parser.add_argument(
        "--test", type=int, default=0, metavar="N",
        help="Testmodus: Nur N Targets verarbeiten"
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Hex-Cache komplett neu aufbauen (ignoriert bestehenden Cache)"
    )
    parser.add_argument(
        "--skip-refdata", action="store_true",
        help="Refdata-Kompilierung ueberspringen (nur perfect_matches als Referenz)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Bestehende Ergebnisse ignorieren, komplett neu berechnen"
    )
    args = parser.parse_args()

    workers = args.jobs if args.jobs > 0 else _default_workers()

    print(f"{'='*60}")
    print(f"  BUILD SIMILAR-ASM DATABASE")
    print(f"  -j{workers}" + (f"  --test {args.test}" if args.test else ""))
    print(f"{'='*60}")
    log.info(f"Start: workers={workers}, test={args.test or 'Full'}")

    t_total = time.time()

    # ── Phase 1: Hex-Cache ──
    if args.rebuild_cache:
        cache = {}
        tqdm.write("Hex-Cache wird komplett neu aufgebaut.")
    else:
        cache = _load_hex_cache()
        if cache:
            tqdm.write(f"Hex-Cache geladen: {len(cache)} Eintraege")

    cache = build_hex_cache_perfect_matches(cache)
    _save_hex_cache(cache)

    if not args.skip_refdata:
        cache = build_hex_cache_refdata(cache, workers=workers)
        _save_hex_cache(cache)
    else:
        tqdm.write("Refdata-Kompilierung uebersprungen (--skip-refdata)")

    tqdm.write(f"Hex-Cache: {len(cache)} Referenzen bereit.\n")

    # ── Phase 2: Target-Hex ──
    targets = extract_all_target_hex()

    if args.test > 0:
        targets = targets[:args.test]
        tqdm.write(f"TESTMODUS: {args.test} Targets\n")

    # ── Phase 3: Vergleich ──
    existing_db = {} if args.no_resume else _load_existing_db()
    if existing_db:
        tqdm.write(f"Resume: {len(existing_db)} bestehende Eintraege\n")

    results = find_best_matches(targets, cache, existing_db, workers=workers)
    _save_output_db(results)

    # ── Zusammenfassung ──
    # Set aller Funktionsnamen die einen eigenen Perfect Match haben
    matched_funcs = {
        entry["func_name"]
        for entry in cache.values()
        if entry.get("cache_key", "").startswith("pm_")
    }

    elapsed = time.time() - t_total
    print_summary(results, matched_funcs)
    print(f"  Gesamtdauer: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"Gesamtdauer: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
