# modules/ido_compiler.py
# Fuehrt GCC als Praeprozessor und IDO 5.3 als Compiler aus.
# Inkludiert dynamisch Header aus der JSONL-Datenbank.

import os
import re
import json
import shutil
import signal
import logging
import tempfile
import threading
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# --- KONFIGURATION & PFADE ---
# Pipeline-Root: ein Verzeichnis ueber modules/
_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# IDO Compiler — liegt direkt im IDO_compiler Ordner
PROJECT_ROOT = os.path.join(_PIPELINE_ROOT, "IDO_compiler")
IDO_DIR = os.path.abspath(PROJECT_ROOT)
IDO_CC = os.path.join(IDO_DIR, "cc")

# Header-Zuordnung: data/asm_headers_mapped.jsonl
HEADER_DB_PATH = os.path.join(_PIPELINE_ROOT, "data", "asm_headers_mapped.jsonl")

# Include-Verzeichnisse fuer den GCC Praeprozessor
# Wir scannen ALLE Unterverzeichnisse von data/header/ automatisch,
# damit GCC jede Header-Datei findet, egal wie tief die Hierarchie ist.
_HEADER_ROOT = os.path.join(_PIPELINE_ROOT, "data", "header")

# Header-Dateiname-Map: "playerstate.h" -> set of dirs that contain it
_header_file_map = {}

def _build_include_dirs():
    """Sammelt alle Verzeichnisse unter data/header/ als Include-Pfade.
    Baut gleichzeitig eine Map aller Header-Dateinamen."""
    global _header_file_map
    dirs = []
    _header_file_map = {}
    
    if os.path.isdir(_HEADER_ROOT):
        dirs.append(_HEADER_ROOT)
        for root, subdirs, files in os.walk(_HEADER_ROOT):
            has_headers = False
            for f in files:
                if f.endswith(".h"):
                    has_headers = True
                    _header_file_map.setdefault(f, []).append(root)
            if has_headers and root not in dirs:
                dirs.append(root)
    
    # Zusaetzliche Projekt-Includes
    for extra in [
        os.path.join(PROJECT_ROOT, "include"),
        os.path.join(PROJECT_ROOT, "src"),
        os.path.join(PROJECT_ROOT, "include", "PR"),
        os.path.join(PROJECT_ROOT, "lib", "ultralib", "include"),
    ]:
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    
    return dirs

INCLUDE_DIRS = _build_include_dirs()


def _normalize_includes(c_code: str) -> str:
    """
    Normalisiert alle #include Pfade auf reine Dateinamen.
    '#include "../../src/overlays/ba/playerstate.h"' -> '#include "playerstate.h"'
    '#include "overlays/ba/playerstate.h"' -> '#include "playerstate.h"'
    
    Da ALLE Verzeichnisse mit .h Dateien als -I Pfade registriert sind,
    findet GCC den Header allein ueber den Dateinamen.
    """
    def _replace_include(m):
        full_path = m.group(1)
        basename = os.path.basename(full_path)
        # Nur normalisieren wenn der Dateiname in unserer Map ist
        if basename in _header_file_map:
            return f'#include "{basename}"'
        # Unbekannter Header — Pfad beibehalten
        return m.group(0)
    
    return re.sub(r'#include\s+"([^"]+)"', _replace_include, c_code)


def normalize_all_headers():
    """
    Einmalige Normalisierung: Durchsucht alle .h Dateien in data/header/
    und ersetzt relative Pfade (../../src/overlays/...) durch reine Dateinamen.
    Wird nur einmal ausgefuehrt — Marker-Datei verhindert Wiederholung.
    """
    marker = os.path.join(_HEADER_ROOT, ".normalized")
    if os.path.exists(marker):
        return
    
    if not os.path.isdir(_HEADER_ROOT):
        return
    
    count = 0
    for root, dirs, files in os.walk(_HEADER_ROOT):
        for fname in files:
            if not fname.endswith(".h"):
                continue
            fpath = os.path.join(root, fname)
            try:
                content = open(fpath, "r", encoding="utf-8", errors="replace").read()
                normalized = _normalize_includes(content)
                if normalized != content:
                    open(fpath, "w", encoding="utf-8").write(normalized)
                    count += 1
            except (IOError, PermissionError):
                pass
    
    try:
        open(marker, "w").write(f"normalized {count} files\n")
        log.info(f"Header-Normalisierung: {count} Dateien angepasst.")
    except (IOError, PermissionError):
        pass


# --- GLOBALE DATENBANK ---
_header_db = None
_header_db_lock = threading.Lock()


def _load_header_db():
    """Laedt die Header-Zuordnungen einmalig in den Arbeitsspeicher.

    Thread-sicher: Das dict wird in einer LOKALEN Variable vollstaendig
    aufgebaut und erst danach atomar `_header_db` zugewiesen. So sieht ein
    paralleler Thread niemals ein halb gefuelltes dict (frueherer Race:
    `_header_db = {}` -> anderer Thread sah "nicht-None" und bekam ein leeres
    dict -> Funktion wurde ohne Header kompiliert -> Compile-Fehler/0%).
    """
    global _header_db
    if _header_db is not None:
        return

    with _header_db_lock:
        # Double-checked: ein anderer Thread koennte waehrend des Wartens fertig
        # geworden sein.
        if _header_db is not None:
            return

        db = {}
        if not os.path.exists(HEADER_DB_PATH):
            log.error(f"Header-Datenbank NICHT GEFUNDEN: {HEADER_DB_PATH}")
            log.error("Headers werden nicht injiziert! Pruefe den Pfad.")
            _header_db = db
            return

        with open(HEADER_DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    func_name = entry.get("asm_function_name", "")
                    func_name = func_name.replace(".s", "")
                    db[func_name] = entry.get("headers", [])
                except json.JSONDecodeError:
                    pass

        log.info(f"Header-DB geladen: {len(db)} Eintraege aus {HEADER_DB_PATH}")
        _header_db = db  # erst jetzt atomar sichtbar machen


def _run_cmd_safely(cmd, cwd=None, env=None, timeout=10):
    """
    Startet einen Subprocess in einer eigenen Process-Group.
    Bei Timeout wird die gesamte Gruppe gekillt (verhindert IDO-Zombies).
    """
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
        # Gesamte Process-Group killen
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return -1, b"", b"Timeout"
    except Exception as e:
        return -1, b"", str(e).encode()


def compile_code(c_code: str, func_name: str) -> dict:
    """
    1. Injiziert die korrekten Header fuer func_name.
    2. Nutzt GCC, um den Code zu praeprozessieren (Makros aufloesen).
    3. Nutzt IDO, um die .i Datei in eine .o Datei zu kompilieren.

    Gibt dict zurueck mit: success, temp_dir, temp_o_path, error_log.
    Der Aufrufer ist verantwortlich fuer cleanup_temp(temp_dir) NACHDEM
    die .o Datei nicht mehr benoetigt wird.
    """
    _load_header_db()

    # 1. Header injizieren
    headers = _header_db.get(func_name, [])
    if headers:
        header_block = "\n".join(headers)
        # Redeclaration-Fix: Wenn der Header die Funktion bereits deklariert,
        # kann es zu "redeclaration" / "Incompatible function return type" kommen.
        # Wir pruefen die .i Datei nach dem Preprocessing auf den Header-Prototyp
        # und passen den C-Code entsprechend an.
        full_c_code = f"{header_block}\n\n{c_code}"
        #log.debug(f"[{func_name}] {len(headers)} Header injiziert: {headers}")
    else:
        full_c_code = c_code
        log.warning(f"[{func_name}] Kein Header-Eintrag in DB gefunden. Code wird ohne Header kompiliert.")

    # Temporaeres Verzeichnis erstellen
    temp_dir = tempfile.mkdtemp(prefix=f"ido_{func_name}_")
    c_file_path = os.path.join(temp_dir, f"{func_name}.c")
    i_file_path = os.path.join(temp_dir, f"{func_name}.i")
    o_file_path = os.path.join(temp_dir, f"{func_name}.o")

    with open(c_file_path, "w", encoding="utf-8") as f:
        f.write(full_c_code)

    # 2. GCC Praeprozessor (-E)
    gcc_cmd = ["gcc", "-E", "-P", "-xc", "-D_LANGUAGE_C", "-D_MIPS_SZLONG=32",
               "-D__attribute__(x)=", "-D__extension__="]
    for inc in INCLUDE_DIRS:
        gcc_cmd.extend(["-I", inc])
    gcc_cmd.extend([c_file_path, "-o", i_file_path])

    rc, _, stderr = _run_cmd_safely(gcc_cmd, timeout=5)
    if rc != 0:
        error_log = _clean_error_log(stderr.decode(errors="replace"), temp_dir, "GCC Preprocessor Error")
        # Temp aufraumen bei Fehler
        cleanup_temp(temp_dir)
        return {"success": False, "temp_dir": "", "temp_o_path": "", "error_log": error_log}

    # 3. IDO Compiler (auf die saubere .i Datei)
    env = os.environ.copy()
    env["COMPILER_PATH"] = IDO_DIR
    env["LD_LIBRARY_PATH"] = f"{IDO_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

    # Flags identisch zum echten Build (banjo-tooie/Makefile CFLAGS):
    #   -c -Wab,-r4300_mul -non_shared -G 0 -Xcpluscomm -O2 -mips2 -woff 807
    # -Wab,-r4300_mul beeinflusst Multiply-Codegen (R4300), -Xcpluscomm erlaubt //-Kommentare.
    # (-w statt -woff 807 ist warnungs-only und .o-neutral; fuer sauberes stderr-Parsing beibehalten.)
    ido_cmd = [IDO_CC, "-c", "-Wab,-r4300_mul", "-O2", "-mips2", "-G", "0", "-non_shared", "-Xcpluscomm", "-w", i_file_path, "-o", o_file_path]
    rc, stdout, stderr = _run_cmd_safely(ido_cmd, cwd=temp_dir, env=env, timeout=10)

    if rc == 0 and os.path.exists(o_file_path):
        return {
            "success": True,
            "temp_dir": temp_dir,
            "temp_o_path": str(o_file_path),
            "error_log": "",
        }
    else:
        raw = stderr.decode(errors="replace") if stderr else stdout.decode(errors="replace")
        error_log = _clean_error_log(raw, temp_dir, "IDO Compiler Error")
        cleanup_temp(temp_dir)
        return {"success": False, "temp_dir": "", "temp_o_path": "", "error_log": error_log}


def cleanup_temp(temp_dir: str):
    """Raeumt ein temporaeres Verzeichnis sicher auf."""
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTISCHER AUTO-FIX (kein KI-Call)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Behebt die haeufigsten, mechanischen Compiler-Fehler ohne KI, indem es die
# autoritativen Typen aus 'env.ext' nutzt. Schlaegt es fehl oder gibt es keine
# Typ-Info -> Code bleibt unveraendert und der Caller faellt auf die KI zurueck.
#
# Behandelte Muster (die sicheren, hochfrequenten):
#   1. Incompatible function return type / redeclaration der eigenen Funktion
#      -> Return-Typ exakt auf env.ext-Signatur setzen
#   2. 'X' undefined fuer eine Funktion in env.ext
#      -> Prototyp aus der Signatur einfuegen
#   3. redeclaration eines Callee-Symbols (KI hat extern/Prototyp doppelt gesetzt)
#      -> die doppelte Deklaration entfernen
#
# Alles andere (Syntax, Struct, Arg-Count, untypisierte Variablen) -> KI.

# Register die als 'ret' vorkommen koennen — alles andere in 'ret' ist ein C-Typ.
_RET_REGISTERS = {"v0", "v1", "f0", "f0/f1", "f12"}

# Konservativer C-Typ-Ausdruck: optionale Qualifier + Bezeichner + optionale Pointer.
_TYPE_RE = r"(?:unsigned\s+|signed\s+|struct\s+|const\s+)*[A-Za-z_]\w*(?:\s*\*+)?"


def parse_tech_env(tech_env_str):
    """Extrahiert das TechEnv-JSON aus dem (ggf. um Cheatsheet erweiterten) String.

    Der Wrapper haengt den Cheatsheet-Kontext mit '\\n\\n' an das JSON an.
    raw_decode() parst das erste JSON-Objekt und ignoriert den Rest robust.
    Gibt None zurueck wenn kein gueltiges JSON gefunden wird.
    """
    if not tech_env_str:
        return None
    s = tech_env_str.lstrip()
    idx = s.find("{")
    if idx < 0:
        return None
    s = s[idx:]
    try:
        obj, _end = json.JSONDecoder().raw_decode(s)
        return obj
    except Exception:
        return None


def _parse_sig(sig_str):
    """'(u8, u32) -> u8' -> (['u8','u32'], 'u8'). Bei Fehler (None, None)."""
    if not sig_str:
        return None, None
    m = re.match(r"\s*\((.*)\)\s*->\s*(.+?)\s*$", sig_str)
    if not m:
        return None, None
    params_raw = m.group(1).strip()
    ret = m.group(2).strip()
    if params_raw and params_raw != "void":
        params = [p.strip() for p in params_raw.split(",")]
    else:
        params = []
    return params, ret


def _is_register(t):
    """True wenn 't' ein MIPS-Returnregister ist (kein C-Typ)."""
    return t is not None and t.strip() in _RET_REGISTERS


def _extract_env(te, func_name):
    """Holt (ext_map, own_ret, own_globals) aus dem geparsten TechEnv-Dict."""
    ext_map, own_ret, own_globals = {}, None, {}
    if not te:
        return ext_map, own_ret, own_globals
    ext_map = (te.get("env", {}) or {}).get("ext", {}) or {}
    for fn in te.get("funcs", []) or []:
        if fn.get("n") == func_name:
            own_ret = fn.get("ret")
            own_globals = fn.get("globals", {}) or {}
            break
    return ext_map, own_ret, own_globals


def _authoritative_ret(func_name, ext_map, own_ret):
    """Liefert den autoritativen C-Return-Typ oder None.

    Prioritaet: 1. env.ext-Signatur  2. 'ret'-Feld falls C-Typ (kein Register).
    """
    if func_name in ext_map:
        _params, ret = _parse_sig(ext_map[func_name])
        if ret:
            return ret
    if own_ret and not _is_register(own_ret):
        return own_ret
    return None


def _set_return_type(code, func_name, new_type):
    """Setzt den Return-Typ der DEFINITION (nicht des Prototyps) von func_name.

    Erkennt die Definition an '... func_name(...) {' (kein ';' vor der Klammer).
    Returns: (new_code, old_type) oder (code, None) wenn keine Definition gefunden.
    """
    pat = re.compile(
        r"(?P<type>" + _TYPE_RE + r")\s+"
        + re.escape(func_name) + r"\s*\([^;{]*\)\s*\{",
    )
    m = pat.search(code)
    if not m:
        return code, None
    old_type = m.group("type").strip()
    if old_type == new_type:
        return code, old_type  # bereits korrekt — kein Change (loop-safe)
    s, e = m.span("type")
    return code[:s] + new_type + code[e:], old_type


def _make_prototype(name, sig_str):
    """Baut einen Prototyp 'ret name(params);' aus einer env.ext-Signatur."""
    params, ret = _parse_sig(sig_str)
    if ret is None:
        return None
    pstr = ", ".join(params) if params else "void"
    return f"{ret} {name}({pstr});"


def autofix_errors(c_code, error_log, func_name, te=None, state=None):
    """Deterministischer Compiler-Fehler-Fixer (kein KI-Call).

    Args:
        c_code:    aktueller C-Code
        error_log: Compiler-Fehlerausgabe
        func_name: Name der Zielfunktion
        te:        geparstes TechEnv-Dict (aus parse_tech_env) oder None
        state:     mutable dict, persistiert ueber Loop-Iterationen einer Funktion.
                   Verhindert dass derselbe Fix mehrfach greift (Loop-Schutz).

    Returns:
        (fixed_code, changed: bool, applied: list[str])
        Wenn nichts gegriffen hat -> (c_code, False, []). Caller faellt auf KI zurueck.
    """
    if state is None:
        state = {}
    if "redeclaration" not in error_log and "Incompatible" not in error_log \
            and "undefined" not in error_log:
        return c_code, False, []

    ext_map, own_ret, own_globals = _extract_env(te, func_name)
    fixed = c_code
    applied = []

    set_ret_done = state.setdefault("set_ret", set())
    removed_decl = state.setdefault("removed_decl", set())
    added_proto = state.setdefault("added_proto", set())

    # ── Muster 3: redeclarierte Callee-Symbole (doppelte extern/Prototypen) ──
    # Zuerst, damit ein entferntes Symbol nicht direkt wieder als Prototyp kommt.
    redecl_symbols = re.findall(r"redeclaration of '(\w+)'", error_log)
    for sym in redecl_symbols:
        if sym == func_name or sym in removed_decl:
            continue
        before = fixed
        # extern-Variable:  "extern s32 core2_VRAM_END;"
        fixed = re.sub(rf"^\s*extern\s+\w+[\s*]*{re.escape(sym)}\s*;.*\n?",
                       "", fixed, flags=re.MULTILINE)
        # Funktions-Prototyp:  "void func_80016068(s32, s32 *);"
        fixed = re.sub(rf"^\s*\w+[\s*]*{re.escape(sym)}\s*\([^)]*\)\s*;.*\n?",
                       "", fixed, flags=re.MULTILINE)
        if fixed != before:
            removed_decl.add(sym)
            applied.append(f"removed dup decl '{sym}'")

    # ── Muster 1: eigener Return-Typ aus env.ext ─────────────────────────────
    own_redecl = func_name in redecl_symbols
    ret_mismatch = ("Incompatible function return type" in error_log) or own_redecl
    if ret_mismatch:
        auth_ret = _authoritative_ret(func_name, ext_map, own_ret)
        if auth_ret and auth_ret not in set_ret_done:
            new_code, old_type = _set_return_type(fixed, func_name, auth_ret)
            if old_type is not None and new_code != fixed:
                fixed = new_code
                set_ret_done.add(auth_ret)
                applied.append(f"return type {old_type} -> {auth_ret}")
                # void-Funktion darf keinen Wert zurueckgeben
                if auth_ret == "void":
                    fixed = re.sub(r"\breturn\s+([^;]+);",
                                   r"/* return \1; */", fixed)

    # ── Muster 2: undefined Funktion aus env.ext -> Prototyp einfuegen ───────
    undefined_syms = re.findall(r"'(\w+)' undefined", error_log)
    protos = []
    for sym in undefined_syms:
        if sym == func_name or sym in added_proto:
            continue
        if sym in ext_map:  # nur Funktionen mit bekannter Signatur
            proto = _make_prototype(sym, ext_map[sym])
            if proto and proto not in fixed:
                protos.append(proto)
                added_proto.add(sym)
                applied.append(f"added proto '{sym}'")
    if protos:
        fixed = "\n".join(protos) + "\n" + fixed

    changed = fixed != c_code
    return fixed, changed, applied


def _clean_error_log(raw_log: str, temp_dir: str, prefix: str) -> str:
    """Entfernt temp-Pfade und kuerzt das Log fuer den LLM-Prompt."""
    if not raw_log:
        return f"{prefix}: Unbekannter Fehler (Kein Output)"

    lines = raw_log.splitlines()
    cleaned_lines = [f"--- {prefix} ---"]

    for line in lines:
        clean_line = line.replace(temp_dir + "/", "")
        if clean_line.strip():
            cleaned_lines.append(clean_line)

    if len(cleaned_lines) > 20:
        cleaned_lines = cleaned_lines[:20]
        cleaned_lines.append("... [Log gekuerzt. Behebe zuerst diese Fehler.]")

    return "\n".join(cleaned_lines)