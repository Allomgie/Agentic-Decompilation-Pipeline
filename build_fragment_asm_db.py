#!/usr/bin/env python3
"""
build_fragment_asm_db.py
========================
Zweite, eigenstaendige Datenbank fuer den geplanten "Frankenstein"-Experten.

Unterschied zur similar_asm_db:
  similar_asm_db  -> 1 bester Donor pro Target (globaler struct_score).
  fragment_asm_db -> bis zu N Donoren pro Target, ausgewaehlt nach
                     ABDECKUNG des Target-Instruktionsstroms (Set-Cover/MMR),
                     plus die exakten Alignment-Bloecke (Transplant-Map).

Kernideen:
  * Prefilter mit GROBER Maske (mask_mips_hex_advanced(is_relocated=True))
    -> hoher Recall: findet auch Donoren mit anderen Immediates/Offsets.
  * Block-Alignment mit FEINER, reloc-bewusster Maske (build_skeleton)
    -> Praezision: ein "Match-Block" ist nur dann einer, wenn die Felder
       semantisch gleich sind (gleiche Konstanten/Offsets/Branch-Distanzen),
       sonst wuerde das Transplantat andere Immediates erzeugen.
  * autojunk=False ueberall -> grosse Funktionen (genau die Frankenstein-
    Targets) verlieren keine haeufigen Tokens an die SequenceMatcher-Heuristik.
  * Untere Laengengrenze gelockert -> kleine Spezial-Donoren, die nur einen
    Slice perfekt abdecken, sind fuer Transplantation Gold.

Auswahl pro Target:
  1. Globalen (groben) Score fuer alle Refs, Top PREFILTER_N behalten.
  2. Fuer die Top-N: feine Match-Bloecke + Coverage-Maske berechnen.
  3. Nur Donoren mit mind. einem Block >= MIN_FRAGMENT kommen in Frage.
  4. SCORE_SLOTS Plaetze fix an die hoechsten (feinen) Scores
     -> ein einzelner Donor, der fast alles abdeckt, schlaegt jedes Puzzle.
  5. Restliche Plaetze greedy nach MARGINALER Abdeckung
     -> jeder weitere Donor deckt moeglichst viel NOCH OFFENES ab.
  6. union_coverage = wie viel % des Targets die gewaehlte Menge zusammen
     abdeckt -> Routing-Signal: niedrig => Funktion genuin neu, Frankenstein
     lohnt nicht.

Usage:
  python3 build_fragment_asm_db.py                  # Full run (auto workers)
  python3 build_fragment_asm_db.py -j4
  python3 build_fragment_asm_db.py --test 20
  python3 build_fragment_asm_db.py --skip-refdata   # nur perfect_matches als Ref
  python3 build_fragment_asm_db.py --min-target-instr 40   # nur grosse Targets
  python3 build_fragment_asm_db.py --rebuild-cache

Output: data/fragment_asm_db.jsonl
Cache:  data/fragment_asm_hex_cache.jsonl
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
    extract_words_from_original_s,
    extract_words_from_objdump,
    mask_mips_hex_advanced,
)

_file_handler = logging.FileHandler(_SCRIPT_DIR / "build_fragment_asm.log", encoding="utf-8")
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

HEX_CACHE_PATH = _SCRIPT_DIR / "data" / "fragment_asm_hex_cache.jsonl"
OUTPUT_DB_PATH = _SCRIPT_DIR / "data" / "fragment_asm_db.jsonl"

IDO_DIR = _SCRIPT_DIR / "IDO_compiler"
IDO_CC = IDO_DIR / "cc"
HEADER_ROOT = _SCRIPT_DIR / "data" / "header"
OBJDUMP = "mips-linux-gnu-objdump"

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Laengenfilter (gelockert ggue. similar_asm_db): kleine Slice-Donoren erlaubt,
# grosse Donoren die das Target als Teilstueck enthalten ebenso.
LENGTH_RATIO_MIN = 0.10
LENGTH_RATIO_MAX = 4.0

# Minimale Instruktionsanzahl fuer sinnvollen Vergleich
MIN_INSTRUCTIONS = 3

# Ein zusammenhaengender Match-Block muss mind. so lang sein, um als
# transplantierbares Fragment zu zaehlen. Kuerzere Treffer sind Rauschen.
MIN_FRAGMENT = 4

# Wie viele Donoren maximal pro Target gespeichert werden.
MAX_SIMILARS = 10

# So viele Plaetze gehen FIX an die hoechsten globalen Scores (bevor die
# greedy Coverage-Auswahl uebernimmt). Empirisch (2026-06-15, 1000 grosse Targets):
# SS=2 ggue. SS=1 bringt +0.1% union_coverage, macht aber 45% aller Donor-Slots
# redundant (Coverage komplett in anderen enthalten) -> reiner Orakel-Ballast fuer
# Frankenstein. SS=1 behaelt den besten Single-Donor (Greedy-Slot 1 = hoechster
# Score) und fuellt den Rest rein nach marginaler Coverage. Default = 1.
SCORE_SLOTS = 1

# Wie viele Kandidaten der grobe Prefilter an die teure Block-Analyse weiterreicht.
PREFILTER_N = 40


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


def _skeletons_from_words(words: list):
    """
    Baut aus (hex, is_reloc)-Paaren beide Skelette:
      coarse: alles Maskierbare maskiert (Recall, fuer Prefilter)
      fine:   nur echt relocatable Felder maskiert (Praezision, fuer Bloecke)
    Beide Listen sind gleich lang (1 Token pro Instruktion).
    """
    coarse = [mask_mips_hex_advanced(h, True) for h, _ in words]
    fine = [mask_mips_hex_advanced(h, r) for h, r in words]
    return coarse, fine


def compile_refdata_to_o(c_path: Path):
    """Kompiliert eine refdata_original C-Datei mit IDO zu einer .o.
    Returns: (o_path | None, temp_dir). Der AUFRUFER raeumt temp_dir auf
    (shutil.rmtree), sobald die .o nicht mehr gebraucht wird."""
    func_name = c_path.stem
    c_code = c_path.read_text(encoding="utf-8", errors="replace")

    temp_dir = tempfile.mkdtemp(prefix=f"fragasm_{func_name}_")
    c_file = os.path.join(temp_dir, f"{func_name}.c")
    i_file = os.path.join(temp_dir, f"{func_name}.i")
    o_file = os.path.join(temp_dir, f"{func_name}.o")

    with open(c_file, "w", encoding="utf-8") as f:
        f.write(c_code)

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
        return None, temp_dir

    env = os.environ.copy()
    env["COMPILER_PATH"] = str(IDO_DIR)
    env["LD_LIBRARY_PATH"] = f"{IDO_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

    ido_cmd = [
        str(IDO_CC), "-c", "-O2", "-mips2", "-G", "0",
        "-non_shared", "-w", i_file, "-o", o_file,
    ]
    rc, _, stderr = _run_cmd(ido_cmd, cwd=temp_dir, env=env, timeout=10)
    if rc != 0 or not os.path.exists(o_file):
        return None, temp_dir
    return o_file, temp_dir


def compile_refdata_words(c_path: Path) -> list:
    """
    Kompiliert eine refdata_original C-Datei mit IDO und gibt (hex, is_reloc)
    Paare via objdump -dr zurueck (reloc-bewusst, im Gegensatz zur similar-DB).

    Returns: Liste von (hex, is_reloc) oder leere Liste bei Fehler.
    """
    o_file, temp_dir = compile_refdata_to_o(c_path)
    try:
        if not o_file:
            return []
        return extract_words_from_objdump(o_file)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER: REFDATA KOMPILIEREN
# ══════════════════════════════════════════════════════════════════════════════

def _worker_compile_refdata(c_path_str: str):
    """Pool-Worker: Kompiliert eine refdata-Datei -> Cache-Eintrag (oder None)."""
    c_path = Path(c_path_str)
    func_name = c_path.stem

    words = compile_refdata_words(c_path)
    if len(words) < MIN_INSTRUCTIONS:
        return None

    coarse, fine = _skeletons_from_words(words)

    return {
        "func_name": func_key_name(func_name),
        "cache_key": f"ref_{func_name}",
        "source": "refdata_original",
        "c_path": str(c_path.relative_to(_SCRIPT_DIR)),
        "sk_coarse": coarse,
        "sk_fine": fine,
        "hex_count": len(fine),
    }


# ══════════════════════════════════════════════════════════════════════════════
# WORKER: FRAGMENT-MATCHING
# ══════════════════════════════════════════════════════════════════════════════

_worker_refs = None


def _init_comparison_worker(refs):
    global _worker_refs
    _worker_refs = refs


def _matching_blocks(t_fine: list, r_fine: list):
    """
    Liefert (blocks, cover_set) auf Basis der feinen Skelette.
      blocks: [[t_start, n, r_start], ...] nur fuer n >= MIN_FRAGMENT
      cover_set: Menge der abgedeckten Target-Indizes
    autojunk=False ist hier kritisch (grosse Funktionen).
    """
    sm = SequenceMatcher(None, t_fine, r_fine, autojunk=False)
    blocks = []
    cover = set()
    for i, j, n in sm.get_matching_blocks():
        if n >= MIN_FRAGMENT:
            blocks.append([i, n, j])
            cover.update(range(i, i + n))
    return blocks, cover


def _greedy_select(t_count: int, cands: list):
    """
    Set-Cover/MMR-Auswahl.
      cands: list[dict] mit Schluesseln score, cover (set), blocks, ...
    Gibt (gewaehlte_liste, union_coverage) zurueck.
    """
    # Nur transplantierbare Donoren (mind. ein Fragment) sind ueberhaupt erlaubt.
    cands = [c for c in cands if c["cover"]]
    if not cands:
        return [], 0.0

    cands.sort(key=lambda c: c["score"], reverse=True)

    chosen = []
    covered = set()

    # Fixe Score-Plaetze.
    for c in cands[:SCORE_SLOTS]:
        chosen.append(c)
        covered |= c["cover"]

    pool = cands[SCORE_SLOTS:]

    # Greedy nach marginaler Abdeckung.
    while len(chosen) < MAX_SIMILARS and pool:
        best_i, best_gain = -1, 0
        for i, c in enumerate(pool):
            gain = len(c["cover"] - covered)
            if gain > best_gain:
                best_gain, best_i = gain, i
        # Lohnt sich nur, wenn mind. ein Fragment NEU dazukommt.
        if best_i < 0 or best_gain < MIN_FRAGMENT:
            break
        c = pool.pop(best_i)
        chosen.append(c)
        covered |= c["cover"]

    union_cov = len(covered) / t_count if t_count else 0.0
    return chosen, union_cov


def _build_candidates(target: dict):
    """
    Phase A+B: grober Prefilter (Recall) -> feine Bloecke/Coverage (Praezision).
    Gibt (cands, length_skipped, self_skipped) zurueck. Von der Auswahl getrennt,
    damit verschiedene SCORE_SLOTS-Strategien auf identischen Kandidaten
    verglichen werden koennen.
    """
    t_name = target["func_name"]
    t_coarse = target["sk_coarse"]
    t_fine = target["sk_fine"]
    t_count = target["hex_count"]

    # ── Phase A: grober Prefilter (Recall) ──
    prelim = []
    length_skipped = 0
    self_skipped = 0
    for ref in _worker_refs:
        if ref["func_name"] == t_name:
            self_skipped += 1
            continue
        r_count = ref["hex_count"]
        if t_count > 0 and r_count > 0:
            ratio = r_count / t_count
            if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
                length_skipped += 1
                continue
        coarse_score = SequenceMatcher(
            None, t_coarse, ref["sk_coarse"], autojunk=False
        ).ratio() * 100.0
        if coarse_score > 0:
            prelim.append((coarse_score, ref))

    prelim.sort(key=lambda x: x[0], reverse=True)
    prelim = prelim[:PREFILTER_N]

    # ── Phase B: feine Bloecke + Coverage (Praezision) ──
    cands = []
    for coarse_score, ref in prelim:
        blocks, cover = _matching_blocks(t_fine, ref["sk_fine"])
        if not cover:
            continue
        fine_score = SequenceMatcher(
            None, t_fine, ref["sk_fine"], autojunk=False
        ).ratio() * 100.0
        cands.append({
            "ref": ref["func_name"],
            "source": ref["source"],
            "ref_path": ref["c_path"],
            "ref_hex_count": ref["hex_count"],
            "score": round(fine_score, 2),
            "score_coarse": round(coarse_score, 2),
            "coverage": round(len(cover) / t_count, 4) if t_count else 0.0,
            "blocks": blocks,
            "cover": cover,
        })

    return cands, length_skipped, self_skipped


def _worker_compare_target(target: dict) -> dict:
    """
    Pool-Worker: findet bis zu MAX_SIMILARS abdeckungs-optimale Donoren.
    """
    t_count = target["hex_count"]

    cands, length_skipped, self_skipped = _build_candidates(target)

    # ── Phase C: Set-Cover Auswahl ──
    chosen, union_cov = _greedy_select(t_count, cands)

    similars = []
    for c in chosen:
        similars.append({
            "ref": c["ref"],
            "source": c["source"],
            "ref_path": c["ref_path"],
            "ref_hex_count": c["ref_hex_count"],
            "struct_score": c["score"],
            "struct_score_coarse": c["score_coarse"],
            "coverage": c["coverage"],
            "blocks": c["blocks"],
        })

    result = None
    if similars:
        result = {
            "target_asm": target["func_name"],
            "target_asm_path": target["s_path"],
            "target_hex_count": t_count,
            "union_coverage": round(union_cov, 4),
            "n_similars": len(similars),
            "similars": similars,
        }

    return {
        "result": result,
        "length_skipped": length_skipped,
        "self_skipped": self_skipped,
        "n_similars": len(similars),
        "union_coverage": union_cov,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: HEX-CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _load_hex_cache() -> dict:
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
    HEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HEX_CACHE_PATH, "w", encoding="utf-8") as f:
        for entry in cache.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_hex_cache_perfect_matches(cache: dict) -> dict:
    """
    Phase 1a: pm-Referenzen. Kein Compile — Hex aus zugehoeriger target .s,
    reloc-bewusst via extract_words_from_original_s.

    pm-Eintraege werden JEDES Mal frisch gebaut (kein Skip) und veraltete/
    geloeschte Eintraege geprunt — der Cache fuer pm ist nur ein Schreib-Cache,
    nicht maszgeblich.
    """
    if not PERFECT_MATCHES_DIR.exists():
        log.warning(f"Perfect matches Ordner nicht gefunden: {PERFECT_MATCHES_DIR}")
        return cache

    # Alte pm-Eintraege verwerfen -> frisch aufbauen (kein Stale-Code/-Pfad).
    for k in [k for k in cache if k.startswith("pm_")]:
        del cache[k]

    c_files = sorted(PERFECT_MATCHES_DIR.rglob("*.c"))
    new_count = 0

    log.info(f"[Phase 1a] Perfect Matches: {len(c_files)} Dateien")

    pbar = tqdm(c_files, desc="Phase 1a │ perfect_matches", unit="fn", leave=True)
    for c_path in pbar:
        func_name = c_path.stem
        cache_key = f"pm_{func_name}"

        rel_path = c_path.relative_to(PERFECT_MATCHES_DIR)
        # Targets liegen unter data/target_asm/nonmatchings/<rel>; aeltere
        # Layouts ohne das Praefix als Fallback zulassen.
        s_path = TARGET_ASM_DIR / "nonmatchings" / rel_path.with_suffix(".s")
        if not s_path.exists():
            s_path = TARGET_ASM_DIR / rel_path.with_suffix(".s")
        if not s_path.exists():
            continue

        words = extract_words_from_original_s(str(s_path))
        if len(words) < MIN_INSTRUCTIONS:
            continue

        coarse, fine = _skeletons_from_words(words)
        cache[cache_key] = {
            "func_name": func_key_name(func_name),
            "cache_key": cache_key,
            "source": "perfect_matches",
            "c_path": str(c_path.relative_to(_SCRIPT_DIR)),
            "sk_coarse": coarse,
            "sk_fine": fine,
            "hex_count": len(fine),
        }
        new_count += 1
        pbar.set_postfix(neu=new_count, refresh=False)

    pbar.close()
    log.info(f"[Phase 1a] Fertig: {new_count} pm-Referenzen")
    return cache


def build_hex_cache_refdata(cache: dict, workers: int = 1) -> dict:
    """Phase 1b: refdata kompilieren (teuer -> gecacht + parallel)."""
    if not REFDATA_DIR.exists():
        log.warning(f"Refdata Ordner nicht gefunden: {REFDATA_DIR}")
        return cache

    c_files = sorted(REFDATA_DIR.glob("*.c"))

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

    desc = "Phase 1b │ compile refdata"
    if workers > 1:
        desc += f" ({workers}w)"

    pbar = tqdm(total=len(todo), desc=desc, unit="fn", leave=True)

    if workers <= 1:
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
        pool = Pool(workers)
        try:
            for entry in pool.imap_unordered(_worker_compile_refdata, todo, chunksize=8):
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
# PHASE 2: TARGET-ASM
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_targets(min_instr: int = 0) -> list:
    """Extrahiert reloc-bewusste Skelette aus allen Target-ASMs."""
    if not TARGET_ASM_DIR.exists():
        log.error(f"Target ASM Ordner nicht gefunden: {TARGET_ASM_DIR}")
        return []

    s_files = sorted(TARGET_ASM_DIR.rglob("*.s"))
    targets = []

    log.info(f"[Phase 2] Target ASMs: {len(s_files)} Dateien")

    pbar = tqdm(s_files, desc="Phase 2  │ extract targets ", unit="fn", leave=True)
    for s_path in pbar:
        func_name = func_key_name(s_path.stem)
        words = extract_words_from_original_s(str(s_path))
        if len(words) < max(MIN_INSTRUCTIONS, min_instr):
            continue
        coarse, fine = _skeletons_from_words(words)
        targets.append({
            "func_name": func_name,
            "s_path": str(s_path.relative_to(_SCRIPT_DIR)),
            "sk_coarse": coarse,
            "sk_fine": fine,
            "hex_count": len(fine),
        })
        pbar.set_postfix(valid=len(targets), refresh=False)

    pbar.close()
    log.info(f"[Phase 2] Fertig: {len(targets)} Targets")
    return targets


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FRAGMENT-MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def find_fragments(targets: list, ref_cache: dict, workers: int = 1) -> dict:
    # Schlanke Ref-Liste fuer die Worker (nur was gebraucht wird).
    refs = [
        {
            "func_name": e["func_name"],
            "source": e["source"],
            "c_path": e["c_path"],
            "sk_coarse": e["sk_coarse"],
            "sk_fine": e["sk_fine"],
            "hex_count": e["hex_count"],
        }
        for e in ref_cache.values()
    ]
    results = {}

    log.info(f"[Phase 3] {len(targets)} Targets vs {len(refs)} Referenzen")

    if not targets:
        return results

    stats = {"done": 0, "with_donors": 0, "cov_sum": 0.0}
    save_interval = 500

    desc = "Phase 3  │ fragments      "
    if workers > 1:
        desc = f"Phase 3  │ fragments ({workers}w)  "

    pbar = tqdm(total=len(targets), desc=desc, unit="fn", leave=True)

    def _process_result(wr: dict):
        if wr["result"]:
            results[wr["result"]["target_asm"]] = wr["result"]
            stats["with_donors"] += 1
            stats["cov_sum"] += wr["union_coverage"]
        stats["done"] += 1
        pbar.update(1)
        avg_cov = stats["cov_sum"] / stats["with_donors"] if stats["with_donors"] else 0
        pbar.set_postfix(
            hit=stats["with_donors"],
            ucov=f"{avg_cov*100:.0f}%",
            refresh=False,
        )

    if workers <= 1:
        global _worker_refs
        _worker_refs = refs
        for target in targets:
            wr = _worker_compare_target(target)
            _process_result(wr)
            if stats["done"] % save_interval == 0:
                _save_output_db(results)
    else:
        chunksize = max(1, min(50, len(targets) // (workers * 4)))
        pool = Pool(workers, initializer=_init_comparison_worker, initargs=(refs,))
        try:
            for wr in pool.imap_unordered(_worker_compare_target, targets, chunksize=chunksize):
                _process_result(wr)
                if stats["done"] % save_interval == 0:
                    _save_output_db(results)
        except KeyboardInterrupt:
            pbar.close()
            tqdm.write("\nCtrl+C — speichere bisherige Ergebnisse...")
            pool.terminate()
            pool.join()
            _save_output_db(results)
            return results
        else:
            pool.close()
        finally:
            pool.join()

    pbar.close()
    log.info(f"[Phase 3] Fertig: {stats['with_donors']} Targets mit Donoren")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _save_output_db(results: dict):
    OUTPUT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DB_PATH, "w", encoding="utf-8") as f:
        for entry in results.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_summary(results: dict):
    if not results:
        print("\nKeine Ergebnisse.")
        return

    n = len(results)
    covs = [r["union_coverage"] for r in results.values()]
    n_sim = [r["n_similars"] for r in results.values()]

    full = [c for c in covs if c >= 0.999]
    high = [c for c in covs if 0.8 <= c < 0.999]
    mid = [c for c in covs if 0.5 <= c < 0.8]
    low = [c for c in covs if c < 0.5]

    print(f"\n{'='*60}")
    print(f"  FRAGMENT-ASM DATABASE — ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print(f"  Targets mit Donoren:   {n:,}")
    print(f"  Donoren/Target (avg):  {sum(n_sim)/n:.1f}  (max {max(n_sim)})")
    print(f"  Union-Coverage (avg):  {sum(covs)/n*100:.1f}%")
    print()
    print(f"  Coverage-Verteilung (Frankenstein-Tauglichkeit):")
    print(f"    100%  (1 Donor reicht oft):  {len(full):>6,}")
    print(f"    80-100% (gut puzzlebar):     {len(high):>6,}")
    print(f"    50-80%  (teils abgedeckt):   {len(mid):>6,}")
    print(f"    <50%    (genuin neu):        {len(low):>6,}")
    print()
    print(f"  Output: {OUTPUT_DB_PATH}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _default_workers() -> int:
    try:
        n = cpu_count()
    except (NotImplementedError, ValueError):
        n = 2
    return max(1, min(n // 2, 6))


def main():
    global SCORE_SLOTS  # wird per --score-slots gesetzt, bevor der Pool forkt
    parser = argparse.ArgumentParser(
        description="Baut Fragment-ASM Datenbank (Multi-Donor / Coverage) fuer Frankenstein"
    )
    parser.add_argument("-j", "--jobs", type=int, default=0, metavar="N",
                        help=f"Parallele Prozesse (default: auto={_default_workers()}).")
    parser.add_argument("--test", type=int, default=0, metavar="N",
                        help="Testmodus: nur N Targets")
    parser.add_argument("--min-target-instr", type=int, default=0, metavar="N",
                        help="Nur Targets mit >= N Instruktionen (Frankenstein zielt auf grosse Funcs)")
    parser.add_argument("--score-slots", type=int, default=SCORE_SLOTS, metavar="N",
                        help=f"Fixe Plaetze nach globalem Score vor der Greedy-Coverage-Auswahl "
                             f"(default {SCORE_SLOTS}). 1 = enger/coverage-fokussiert, 2 = mehr High-Score-Basen.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Hex-Cache komplett neu aufbauen")
    parser.add_argument("--skip-refdata", action="store_true",
                        help="Refdata-Kompilierung ueberspringen (nur perfect_matches)")
    args = parser.parse_args()

    # Globale Strategie setzen, BEVOR der Pool geforkt wird (Worker erben sie).
    SCORE_SLOTS = args.score_slots

    workers = args.jobs if args.jobs > 0 else _default_workers()

    print(f"{'='*60}")
    print(f"  BUILD FRAGMENT-ASM DATABASE")
    print(f"  -j{workers}" + (f"  --test {args.test}" if args.test else "")
          + (f"  --min-target-instr {args.min_target_instr}" if args.min_target_instr else ""))
    print(f"{'='*60}")
    log.info(f"Start: workers={workers}, test={args.test or 'Full'}, "
             f"min_instr={args.min_target_instr}")

    t_total = time.time()

    # ── Phase 1: Cache ──
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

    # ── Phase 2: Targets ──
    targets = extract_all_targets(min_instr=args.min_target_instr)
    if args.test > 0:
        targets = targets[:args.test]
        tqdm.write(f"TESTMODUS: {args.test} Targets\n")

    # ── Phase 3: Fragmente ──
    results = find_fragments(targets, cache, workers=workers)
    _save_output_db(results)

    elapsed = time.time() - t_total
    print_summary(results)
    print(f"  Gesamtdauer: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"Gesamtdauer: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
