# modules/ido_comparison.py
# Vergleicht kompilierte .o Dateien mit originalen .s Target-Dateien.
# Extrahiert Hex-Opcodes, berechnet Match-Scores, verwaltet Leaderboard.

import os
import re
import json
import atexit
import shutil
import logging
import tempfile
import threading
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

OBJDUMP = "mips-linux-gnu-objdump"

# --- Wasserdichter .o-vs-.o-Vergleich: Recipe zum Assemblieren der Target-.s ---
# Das einzige korrekte Byte-Match-Kriterium vergleicht zwei UNRELOCIERTE .o
# (Draft vs. assemblierte Target-.s): rohe Immediates sind beidseitig 0, und
# Relocations werden ueber Symbol+Typ verglichen — nicht ueber den bereits
# aufgeloesten .s-ROM-Hex (der sonst jede %hi/%lo/jal-Stelle faelschlich als
# Mismatch zaehlt).
_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HEADER_DIR = os.path.join(_PIPELINE_ROOT, "data", "header")
_PRELUDE = next(
    (p for p in [
        os.path.join(_HEADER_DIR, "include", "prelude.inc"),
        os.path.join(_PIPELINE_ROOT, "modules", "decomp-permuter", "prelude.inc"),
    ] if os.path.exists(p)),
    "",
)
# Identisch zu batch_permuter.py / ASFLAGS aus banjo-tooie/Makefile.
_ASM_CMD = [
    "mips-linux-gnu-gcc", "-march=vr4300", "-mabi=32", "-mgp32", "-mfp32",
    "-mips3", "-mno-abicalls", "-G0", "-fno-pic", "-gdwarf", "-c",
    "-x", "assembler-with-cpp", "-D_LANGUAGE_ASSEMBLY", "-I", _HEADER_DIR,
    "-DBUILD_VERSION=VERSION_J", "-D_FINALROM", "-DF3DEX_GBI_2",
]

# Target-.s -> .o wird pro Pfad genau einmal assembliert (Target ist statisch).
_target_o_cache = {}
_target_o_lock = threading.Lock()
_target_o_dir = None

_INSN_LINE_RE = re.compile(r"\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+\S+")
_RELOC_LINE_RE = re.compile(r"\s*[0-9a-fA-F]+:\s+(R_MIPS_\S+)\s+(\S+)")


def _words_with_reloc(o_path: str) -> list:
    """list[(hex, reloc_key)] aus objdump -dr. reloc_key='TYPE SYMBOL(+addend)'
    oder None. Die in-section-Offsets werden bewusst weggelassen (sie sind bei
    byte-identischen Funktionen ohnehin gleich, aber so ist der Vergleich robust
    gegen Verschiebungen)."""
    if not o_path or not os.path.exists(o_path):
        return []
    try:
        res = subprocess.run([OBJDUMP, "-dr", "-z", o_path],
                             capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.warning(f"objdump fehlgeschlagen ({o_path}): {e}")
        return []
    words = []  # [[hex, reloc_key]]
    for line in res.stdout.splitlines():
        m = _INSN_LINE_RE.match(line)
        if m:
            words.append([m.group(1).lower(), None])
            continue
        rm = _RELOC_LINE_RE.match(line)
        if rm and words:
            words[-1][1] = f"{rm.group(1)} {rm.group(2)}"
    return [(h, r) for h, r in words]


def _assemble_target_o(target_s_path: str):
    """Assembliert die Target-.s (mit prelude.inc) zu einer unrelocierten .o.
    Ergebnis wird pro Pfad gecacht. Returns .o-Pfad oder None bei Fehler
    (dann faellt evaluate_match auf den alten .s-Hex-Vergleich zurueck)."""
    global _target_o_dir
    if not target_s_path or not os.path.exists(target_s_path) or not _PRELUDE:
        return None
    key = os.path.abspath(target_s_path)

    # 1) Cache-Lookup unter Lock (kurz), Temp-Dir lazy anlegen.
    with _target_o_lock:
        if key in _target_o_cache:
            return _target_o_cache[key]
        if _target_o_dir is None:
            _target_o_dir = tempfile.mkdtemp(prefix="ido_target_o_")
            atexit.register(shutil.rmtree, _target_o_dir, ignore_errors=True)

    # 2) Assemblierung OHNE globalen Lock -> verschiedene Targets blockieren
    #    sich nicht. Eindeutige Temp-Namen (mkstemp) verhindern Kollisionen,
    #    falls zwei Threads zufaellig dasselbe Target gleichzeitig bauen.
    stem = os.path.basename(target_s_path)[:-2] or "tgt"
    fd_s, tmp_s = tempfile.mkstemp(prefix=stem + "_", suffix=".s", dir=_target_o_dir)
    os.close(fd_s)
    tmp_o = tmp_s[:-2] + ".o"
    result = None
    try:
        with open(tmp_s, "w", encoding="utf-8") as out:
            out.write(open(_PRELUDE, encoding="utf-8").read())
            out.write("\n")
            out.write(open(target_s_path, encoding="utf-8").read())
        r = subprocess.run(_ASM_CMD + [tmp_s, "-o", tmp_o],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and os.path.exists(tmp_o):
            result = tmp_o
        else:
            log.warning(f"Target-Assemblierung fehlgeschlagen ({stem}): "
                        f"{r.stderr.strip()[:200]}")
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning(f"Target-Assemblierung Exception ({stem}): {e}")

    # 3) Ergebnis cachen (Recheck: anderer Thread war evtl. schneller).
    with _target_o_lock:
        if key in _target_o_cache:
            return _target_o_cache[key]
        _target_o_cache[key] = result
        return result

LEADERBOARD_PATH = os.path.join("data", "best_matches.jsonl")
HISTORY_DIR = os.path.join("data", "function_history")

# Thread-safe Leaderboard
_leaderboard = {}
_leaderboard_loaded = False
_lb_lock = threading.Lock()


def _load_leaderboard():
    """Laedt das globale Leaderboard einmalig (thread-safe)."""
    global _leaderboard, _leaderboard_loaded
    with _lb_lock:
        if _leaderboard_loaded:
            return
        _leaderboard = {}
        if os.path.exists(LEADERBOARD_PATH):
            with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        _leaderboard[entry["func_name"]] = entry
                    except (json.JSONDecodeError, KeyError):
                        pass
        _leaderboard_loaded = True


def _save_leaderboard():
    """Schreibt das Leaderboard zurueck (thread-safe, wird unter Lock aufgerufen)."""
    os.makedirs(os.path.dirname(LEADERBOARD_PATH) or ".", exist_ok=True)
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        for entry in _leaderboard.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_history(func_name: str, score: float, mismatch_count: int, c_code: str):
    """Haengt einen Eintrag an die Funktions-History an."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_path = os.path.join(HISTORY_DIR, f"{func_name}.jsonl")
    # Ersten 500 Zeichen des Codes speichern (kompakt aber debugbar)
    entry = {
        "score": round(score, 2),
        "mismatches": mismatch_count,
        "code_length": len(c_code),
        "code_preview": c_code[:500],
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Jede R_MIPS_* Relocation veraendert das Immediate-/Adressfeld der Instruktion
# (HI16/LO16/26/GPREL16/GOT16/CALL16/LITERAL ...). Treffer => Feld maskieren.
_RELOC_RE = re.compile(r"R_MIPS_\w+")
# jal/j tragen ein link-/section-relatives Sprungziel -> immer maskieren, auch
# wenn objdump die Relocation (lokales Label) nicht als Folgezeile listet.
_JUMP_MNEMS = ("jal", "j")


def extract_words_from_objdump(o_path: str) -> list:
    """Extrahiert (hex, is_relocated)-Paare aus einer .o Datei via objdump -dr.

    is_relocated=True genau dann, wenn das Wort eine R_MIPS_* Relocation traegt
    (Folgezeile von -dr) ODER ein jal/j ist. Nur solche Felder werden im
    Skelett-Vergleich maskiert; echte Offsets/Konstanten/Branch-Distanzen
    bleiben erhalten.
    """
    if not o_path or not os.path.exists(o_path):
        return []

    try:
        result = subprocess.run(
            [OBJDUMP, "-dr", "-z", o_path],
            capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.warning(f"objdump fehlgeschlagen: {e}")
        return []

    insn_re = re.compile(r"\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+(\S+)")
    words = []  # list[[hex, is_reloc]]
    for line in result.stdout.splitlines():
        m = insn_re.match(line)
        if m:
            mnem = m.group(2).lower()
            words.append([m.group(1).lower(), mnem in _JUMP_MNEMS])
            continue
        # Reloc-Folgezeile (z.B. "  4: R_MIPS_LO16  D_80079AC0") markiert das
        # zuletzt gesehene Wort als relocatable.
        if words and _RELOC_RE.search(line):
            words[-1][1] = True

    return [(h, r) for h, r in words]


def extract_hex_from_objdump(o_path: str) -> list:
    """Rueckwaertskompatibel: nur die Hex-Opcodes (ohne Reloc-Flags)."""
    return [h for h, _ in extract_words_from_objdump(o_path)]


def _s_operand_is_reloc(rest: str) -> bool:
    """True wenn die Operanden-Textzeile einer .s-Instruktion ein
    relocatables Feld traegt: %hi(...)/%lo(...) oder ein jal/j Sprungziel."""
    if "%hi(" in rest or "%lo(" in rest:
        return True
    head = rest.split(None, 1)
    return bool(head) and head[0].lower() in _JUMP_MNEMS


def extract_words_from_original_s(s_path: str) -> list:
    """
    Extrahiert (hex, is_relocated)-Paare aus einer originalen .s Datei
    (spimdisasm-Format).

    spimdisasm Format:
        /* ROM_OFFSET VRAM_ADDR HEX_OPCODE */  opcode  operands
        /* 1F70078 80800248 248EFFC0 */  addiu      $t6, $a0, -0x40

    Wir extrahieren nur aus dem .text-Bereich (ignorieren .rodata, .data, .bss).
    is_relocated wird aus dem symbolischen Operanden abgeleitet (%hi/%lo, jal/j).
    """
    if not s_path or not os.path.exists(s_path):
        return []

    words = []  # list[(hex, is_reloc)]
    in_text_section = False

    with open(s_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # Sektions-Tracking
            if stripped.startswith(".section"):
                in_text_section = ".text" in stripped
                continue

            # Auch ohne explizite .section: glabel markiert Code-Beginn
            if stripped.startswith("glabel"):
                in_text_section = True
                continue
            if stripped.startswith("endlabel"):
                in_text_section = False
                continue

            if not in_text_section:
                continue

            # Skip: Labels, Direktiven, leere Zeilen
            if not stripped or stripped.startswith((".L", ".", "#", "dlabel", "enddlabel")):
                continue

            # spimdisasm: /* ROM VRAM HEX */ opcode operands
            # Das HEX ist immer das LETZTE 8-Zeichen-Hex im Kommentar
            m = re.search(
                r"/\*.*?([0-9a-fA-F]{8})\s*\*/\s+([a-z].*)$",
                stripped,
            )
            if m:
                words.append((m.group(1).lower(), _s_operand_is_reloc(m.group(2))))
                continue

            # Fallback: raw format "HEX8 opcode operands"
            m = re.match(r"([0-9a-fA-F]{8})\s+([a-z].*)$", stripped)
            if m:
                words.append((m.group(1).lower(), _s_operand_is_reloc(m.group(2))))

    return words


def extract_hex_from_original_s(s_path: str) -> list:
    """Rueckwaertskompatibel: nur die Hex-Opcodes (ohne Reloc-Flags)."""
    return [h for h, _ in extract_words_from_original_s(s_path)]


def mask_mips_hex_advanced(hex_str: str, is_relocated: bool = True) -> str:
    """
    Maskiert das Immediate-/Adressfeld eines MIPS-Hex-Opcodes, ABER nur wenn
    das Feld tatsaechlich relocatable ist (is_relocated=True). Echte Offsets,
    Konstanten und Branch-Distanzen (is_relocated=False) bleiben vollstaendig
    erhalten — sie kodieren Semantik (Array-Index, Struct-Offset, Loop-Distanz),
    die ein Permuter nicht umschreiben kann.

    is_relocated steuert NUR, OB maskiert wird; die Opcode-Klasse steuert WIE
    (J-Type: 26 Bit, I-Type: untere 16 Bit, R-Type: nichts).

    Default True == altes Verhalten (rueckwaertskompatibel fuer Aufrufer ohne
    Reloc-Info, z.B. die Similar-DB).

    Beispiel: 27bdffe8 (addiu sp,sp,-24), is_relocated=False -> 27bdffe8 (ganz).
              3c028008 (lui v0,%hi(sym)), is_relocated=True  -> 3c02____.
    """
    if len(hex_str) != 8:
        return hex_str

    if not is_relocated:
        # Echtes Immediate / Branch-Distanz / Konstante -> Wort komplett behalten.
        return hex_str

    # Obere 6 Bits = Opcode
    top_byte = int(hex_str[0:2], 16)
    opcode = top_byte >> 2

    # I-Type Instruktionen: obere 16 Bit (Opcode+RS+RT) behalten, untere 16 maskieren
    i_type_opcodes = {
        0x04, 0x05, 0x06, 0x07,  # beq, bne, blez, bgtz
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,  # addi, addiu, slti, sltiu, andi, ori, xori, lui
        0x20, 0x21, 0x23, 0x24, 0x25,  # lb, lh, lw, lbu, lhu
        0x28, 0x29, 0x2B,  # sb, sh, sw
        0x31, 0x35, 0x39, 0x3D,  # lwc1, ldc1, swc1, sdc1
    }

    # J-Type: Opcode behalten, 26-Bit Adresse maskieren
    j_type_opcodes = {0x02, 0x03}  # j, jal

    if opcode in j_type_opcodes:
        return hex_str[0:2] + "______"
    elif opcode in i_type_opcodes:
        return hex_str[0:4] + "____"
    else:
        # R-Type oder Special: gesamtes Wort behalten (keine Relocation)
        return hex_str


def build_skeleton(words: list) -> list:
    """Baut das maskierte Vergleichs-Skelett aus (hex, is_relocated)-Paaren,
    wie sie extract_words_from_objdump / extract_words_from_original_s liefern.
    Nur relocatable Felder werden maskiert, echte Immediates bleiben erhalten."""
    return [mask_mips_hex_advanced(h, r) for h, r in words]


# --- Frame-Stripping fuer Mid-Function-Split-Bloecke --------------------------
# Ein eigenstaendig kompiliertes C bekommt IMMER einen Prolog/Epilog (Frame-
# Setup), den ein Mid-Function-Split-Block NICHT hat. Das verfaelscht den
# struct_score. Maskierte Frame-Tokens (vgl. mask_mips_hex_advanced):
#   27bd____  addiu $sp, $sp, imm     (Stack alloc/dealloc)
#   afb*      sw  $s0-$ra, off($sp)   (callee-saved / ra Store)
#   8fb*      lw  $s0-$ra, off($sp)   (callee-saved / ra Load)
#   e7b* f7b* swc1/sdc1 $fXX,off($sp) (fp callee-saved Store)
#   c7b* d7b* lwc1/ldc1 $fXX,off($sp) (fp callee-saved Load)
#   03e00008  jr  $ra                 (R-Type, vollstaendig behalten)
#   00000000  nop
_FRAME_STORE = ("27bd", "afb", "e7b", "f7b")
_FRAME_LOAD = ("27bd", "8fb", "c7b", "d7b")
_JR_RA = "03e00008"
_NOP = "00000000"


def _starts_any(tok: str, prefixes) -> bool:
    return any(tok.startswith(p) for p in prefixes)


def _is_framed(skel: list) -> bool:
    """True wenn das Skelett wie eine vollstaendige Funktion aussieht: beginnt
    mit Stack-Allokation ODER endet (in den letzten 3 Tokens) mit `jr $ra`."""
    if not skel:
        return False
    return skel[0].startswith("27bd") or (_JR_RA in skel[-3:])


def strip_frame_skeleton(skel: list):
    """Entfernt den fuehrenden Prolog-Lauf und den abschliessenden Epilog-Lauf
    aus einem maskierten Hex-Skelett, damit ein standalone-kompilierter Draft
    Body-gegen-Body mit einem frame-losen Split-Block verglichen werden kann.

    Konservativ + verankert: nur ein maximaler PREFIX, der mit einer Stack-
    Allokation (27bd____) beginnt, und ein maximaler SUFFIX, der das `jr $ra`-
    Terminal enthaelt, werden entfernt. Innere Epiloge (early returns) bleiben
    unangetastet. Rueckgabe: (gestripptes_list, n_prolog, n_epilog).
    """
    if not skel:
        return list(skel), 0, 0
    toks = list(skel)
    n = len(toks)
    # --- Prolog: muss mit einer Stack-Allokation beginnen ---
    i = 0
    if toks[0].startswith("27bd"):
        while i < n and _starts_any(toks[i], _FRAME_STORE):
            i += 1
    # --- Epilog: maximaler Suffix aus Frame-Load/Dealloc/jr/nop ---
    k = n
    while k > i and (_starts_any(toks[k - 1], _FRAME_LOAD)
                     or toks[k - 1] in (_JR_RA, _NOP)):
        k -= 1
    # nur strippen, wenn der Suffix die Funktion wirklich terminiert (jr $ra)
    j = k if _JR_RA in toks[k:n] else n
    return toks[i:j], i, n - j


def evaluate_match(
    func_name: str,
    c_code: str,
    draft_o_path: str,
    target_s_path: str,
    update_leaderboard: bool = True,
    strip_frame=None,
) -> dict:
    """
    Vergleicht die kompilierte .o mit der Target-.s und berechnet den Match-Score.

    Returns: dict mit struct_score, match_rate, mismatch_count, is_permuter_ready.
    """
    _load_leaderboard()

    draft_words = extract_words_from_objdump(draft_o_path)
    target_words = extract_words_from_original_s(target_s_path)

    # Strip alignment padding: trailing NOPs (00000000) in draft beyond target length
    while (len(draft_words) > len(target_words)
           and draft_words and draft_words[-1][0] == "00000000"):
        draft_words.pop()

    draft_hex = [h for h, _ in draft_words]
    target_hex = [h for h, _ in target_words]

    if not draft_hex or not target_hex:
        return {
            "match_rate": 0.0,
            "mismatch_count": max(len(draft_hex), len(target_hex)),
            "is_permuter_ready": False,
            "struct_score": 0.0,
            "struct_score_raw": 0.0,
            "frame_stripped": False,
        }

    # Wasserdichter Byte-Vergleich: Target-.s zu .o assemblieren und .o-vs-.o
    # vergleichen (beide unrelociert) -> Hex-Gleichheit UND Reloc-Symbol/Typ.
    # Nur so zaehlen %hi/%lo/jal-Stellen korrekt: das rohe Draft-Immediate (0)
    # wird gegen das rohe Target-Immediate (0) verglichen, die Relocation ueber
    # Symbol+Typ. Faellt auf den alten .s-ROM-Hex-Vergleich zurueck, falls die
    # Assemblierung nicht moeglich ist (dann Naeherung: Relocs zaehlen als Diff).
    target_o = _assemble_target_o(target_s_path)
    if target_o:
        d_words = _words_with_reloc(draft_o_path)
        t_words = _words_with_reloc(target_o)
        while (len(d_words) > len(t_words)
               and d_words and d_words[-1][0] == "00000000"):
            d_words.pop()
        max_len = max(len(d_words), len(t_words))
        min_len = min(len(d_words), len(t_words))
        exact_matches = 0
        for i in range(min_len):
            if (d_words[i][0] == t_words[i][0]
                    and (d_words[i][1] or "") == (t_words[i][1] or "")):
                exact_matches += 1
        lengths_equal = (len(d_words) == len(t_words))
    else:
        # Fallback: alter exakter Vergleich gegen den aufgeloesten .s-ROM-Hex.
        exact_matches = 0
        max_len = max(len(draft_hex), len(target_hex))
        min_len = min(len(draft_hex), len(target_hex))
        for i in range(min_len):
            if draft_hex[i] == target_hex[i]:
                exact_matches += 1
        lengths_equal = (len(draft_hex) == len(target_hex))

    exact_rate = (exact_matches / max_len * 100.0) if max_len > 0 else 0.0

    # Struktureller Vergleich: maskiert NUR tatsaechlich relocatable Felder
    # (%hi/%lo/jal). Echte Offsets/Konstanten/Branch-Distanzen bleiben erhalten,
    # damit z.B. ein falscher Array-Index (lw 0x0 statt 0x18) nicht faelschlich
    # als 100%-Struct-Match durchgeht.
    #
    # WICHTIG: Wenn das Target assembliert wurde, basiert struct_score auf
    # DENSELBEN .o-Worten wie match_rate (gleiche Laenge/Quelle). Sonst lief
    # struct_score auf dem .s-Textparse, der manche Instruktionen verfehlt
    # (z.B. 41 statt 44 Worte) -> kuerzeres Target -> SequenceMatcher-Strafe ->
    # struct_score konnte UNTER match_rate fallen (logisch unmoeglich, da
    # Struktur die laxere Metrik ist). Konsistente Quelle behebt das.
    if target_o:
        draft_skeleton = build_skeleton([(h, rk is not None) for h, rk in d_words])
        target_skeleton = build_skeleton([(h, rk is not None) for h, rk in t_words])
    else:
        draft_skeleton = build_skeleton(draft_words)
        target_skeleton = build_skeleton(target_words)
    struct_score = SequenceMatcher(None, target_skeleton, draft_skeleton).ratio() * 100.0

    # Frame-Stripping fuer Mid-Function-Split-Bloecke: ein standalone-kompilierter
    # Draft traegt einen Prolog/Epilog, den ein Block aus der Funktionsmitte nicht
    # hat. strip_frame=None -> auto: nur wenn Draft einen Frame hat, das Target
    # aber nicht (genau die Split-Block-Situation). True/False erzwingt/deaktiviert.
    do_strip = strip_frame
    if do_strip is None:
        # Ordner-Erkennung: alles unter splits/ ist ein Block aus einer
        # Funktionsmitte -> der standalone-Draft traegt einen Frame, das Target
        # nicht. (Sicher auch fuer den ersten, gerahmten Chunk: dann ist der
        # Strip auf beiden Seiten symmetrisch und auf einem frame-losen Target
        # ohnehin ein No-op.)
        tp = (target_s_path or "").replace("\\", "/")
        do_strip = "/splits/" in tp
    struct_score_raw = struct_score
    if do_strip:
        stripped_draft, n_pro, n_epi = strip_frame_skeleton(draft_skeleton)
        stripped_target, _, _ = strip_frame_skeleton(target_skeleton)
        struct_score = SequenceMatcher(
            None, stripped_target, stripped_draft).ratio() * 100.0

    # Mismatch-Zaehlung
    mismatch_count = max_len - exact_matches

    # Permuter-Readiness: 
    # - "fast_lane": wenige Mismatches, gleiche Laenge -> hohe Chance auf 100%
    # - "available": Code kompiliert und es gibt Mismatches -> Permuter kann helfen
    is_permuter_fast_lane = (
        mismatch_count <= 10
        and struct_score >= 70.0
    )
    is_permuter_available = mismatch_count > 0 and mismatch_count <= 60

    # 100% Check: ALLE Worte identisch (Hex + Reloc) UND gleiche Laenge
    if exact_matches == max_len and lengths_equal:
        match_rate = 100.0
    else:
        match_rate = exact_rate

    # Leaderboard updaten (thread-safe)
    if update_leaderboard:
        with _lb_lock:
            prev = _leaderboard.get(func_name, {}).get("match_rate", 0.0)
            if match_rate > prev:
                _leaderboard[func_name] = {
                    "func_name": func_name,
                    "match_rate": round(match_rate, 2),
                    "struct_score": round(struct_score, 2),
                    "mismatches": mismatch_count,
                }
                _save_leaderboard()

        _log_history(func_name, match_rate, mismatch_count, c_code)

    return {
        "match_rate": round(match_rate, 2),
        "struct_score": round(struct_score, 2),
        "struct_score_raw": round(struct_score_raw, 2),
        "frame_stripped": bool(do_strip),
        "mismatch_count": mismatch_count,
        "is_permuter_ready": is_permuter_fast_lane,
        "is_permuter_available": is_permuter_available,
    }