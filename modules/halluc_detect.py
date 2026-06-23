"""Halluzinations-Detektor fuer den Compiler-Error-Experten.

ZWECK (Lukas): Nur die Faelle an die KI reichen, in denen wir SICHER sind, dass ein
Typname HALLUZINIERT ist. Ground Truth = die TATSAECHLICHEN Eingaben des generierenden
Modells: techenv (data/tech_env/.../<fn>.json) + asm (data/target_asm/.../<fn>.s),
plus die im Compile-Environment verfuegbaren Header-Typen + Basistypen.

Ein Typname, der einen Compile-Fehler ausloest UND in KEINER dieser Quellen vorkommt,
ist unsourced -> mit Sicherheit halluziniert (das Modell hat ihn aus dem Training
erfunden, nicht aus Kontext/asm abgeleitet). Konservativ: im Zweifel NICHT flaggen.

WICHTIG (Lukas): Ein falscher NAME heisst NICHT, dass das generierte Struct/der Zugriff
falsch ist. Der Detektor identifiziert nur den unsourced NAMEN; die Reparatur (Name ->
verfuegbarer Typ ODER expliziter Offset, ZUGRIFF beibehalten) ist Sache der KI-Stufe.
"""
import os
import re
import glob
import json
import functools

_PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TECHENV_ROOT = os.path.join(_PIPELINE_ROOT, "data", "tech_env")
_ASM_ROOT = os.path.join(_PIPELINE_ROOT, "data", "target_asm")
_HEADER_ROOT = os.path.join(_PIPELINE_ROOT, "data", "header")

_BASE = set("void char short int long float double signed unsigned u8 u16 u32 u64 s8 s16 s32 "
            "s64 f32 f64 vu8 vu16 vu32 vs8 vs16 vs32 vf32 const volatile static register "
            "struct union enum typedef return if else for while switch case default break "
            "continue do goto sizeof".split())

_IDENT = re.compile(r"[A-Za-z_]\w*")
_ERRLINE = re.compile(r"cfe:\s*(Error|Warning):\s*[^,]*,\s*line\s*\d+:\s*(.+)")


@functools.lru_cache(maxsize=1)
def _header_type_vocab():
    """Alle Typnamen, die im Compile-Environment (data/header) verfuegbar sind."""
    vocab = set()
    for root, _, files in os.walk(_HEADER_ROOT):
        for f in files:
            if not f.endswith(".h"):
                continue
            try:
                txt = open(os.path.join(root, f), errors="replace").read()
            except Exception:
                continue
            for m in re.finditer(r"\b([A-Za-z_]\w+)\s*;", txt):       # typedef ... Name;
                vocab.add(m.group(1))
            for m in re.finditer(r"\b(?:struct|union|enum)\s+([A-Za-z_]\w*)", txt):
                vocab.add(m.group(1))
    return vocab


def _find_one(root, fname):
    hits = glob.glob(os.path.join(root, "**", fname), recursive=True)
    return hits[0] if hits else None


@functools.lru_cache(maxsize=4096)
def sourced_tokens(func_name):
    """Alle Identifier-Tokens, die dem generierenden Modell als Eingabe vorlagen:
    techenv-JSON (env.s-Keys, ext/sym-Signaturen, Feldtypen, funcs) + asm. Als Set."""
    toks = set()
    te = _find_one(_TECHENV_ROOT, func_name + ".json")
    if te:
        try:
            raw = open(te, errors="replace").read()
            toks.update(_IDENT.findall(raw))   # robust: jeder Identifier im techenv-Text
        except Exception:
            pass
    asm = _find_one(_ASM_ROOT, func_name + ".s")
    if asm:
        try:
            toks.update(_IDENT.findall(open(asm, errors="replace").read()))
        except Exception:
            pass
    return toks


@functools.lru_cache(maxsize=4096)
def _techenv_json(func_name):
    te = _find_one(_TECHENV_ROOT, func_name + ".json")
    if not te:
        return None
    try:
        return json.load(open(te, errors="replace"))
    except Exception:
        return None


def techenv_types_text(func_name):
    """Kompakte Beschreibung der dem Modell VERFUEGBAREN Structs/Signaturen aus env.s/ext
    (fuer den KI-Prompt -- Hebel 4 draft-erhaltend: echte Felder+Offsets statt blindem Raten).
    Leer wenn env.s leer (Starvation)."""
    d = _techenv_json(func_name)
    if not d:
        return ""
    env = d.get("env", {}) or {}
    out = []
    for sname, fields in (env.get("s") or {}).items():
        items = "; ".join(f"{off} {desc}" for off, desc in fields.items())
        out.append(f"struct {sname} {{ {items} }}")
    for fn2, sig in (env.get("ext") or {}).items():
        out.append(f"{fn2}: {sig}")
    return "\n".join(out)


def target_asm_text(func_name, max_lines=400):
    """Target-ASM der Funktion (fuer die KI, um echte Offsets erfundener Namen aufzuloesen)."""
    asm = _find_one(_ASM_ROOT, func_name + ".s")
    if not asm:
        return ""
    try:
        lines = open(asm, errors="replace").read().splitlines()
    except Exception:
        return ""
    return "\n".join(lines[:max_lines])


def is_sourced(type_name, func_name):
    """True, wenn der Typname aus einer legitimen Quelle stammt (techenv/asm/header/base)."""
    if type_name in _BASE or type_name in _header_type_vocab():
        return True
    return type_name in sourced_tokens(func_name)


# ---- offendierende Typ-Tokens aus dem cfe-Log extrahieren ------------------------
def _type_candidates(src):
    """Typ-Position-Identifier einer Quellzeile (Decl `T *v`/`T v`, Param `(T *p`, `struct T`)."""
    out = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s+\*?\s*[A-Za-z_]\w*", src):
        out.append(m.group(1))
    for m in re.finditer(r"[(,]\s*([A-Za-z_]\w*)\s*\*?\s*[A-Za-z_]\w*", src):
        out.append(m.group(1))
    for m in re.finditer(r"\b(?:struct|union|enum)\s+([A-Za-z_]\w*)", src):
        out.append(m.group(1))
    return out


# Worte aus der Log-Trunkierungs-Notiz "[Log gekuerzt. Behebe zuerst diese Fehler.]"
_NOTE_WORDS = set("Log Behebe Fehler".split())


def _type_shaped(t):
    """Konservativ: nur was wie ein erfundener Typname aussieht (CamelCase/Underscore-Typ),
    keine lokalen Variablen/Parameter/Datensymbole/Felder."""
    if len(t) < 4 or t in _BASE or t in _NOTE_WORDS:
        return False
    if t.startswith(("func_", "local_", "param_", "D_", "B_", "temp_", "var_")):
        return False
    return any(c.isupper() for c in t)


def unsourced_type_names(error_log, func_name):
    """-> sortierte Liste der Typnamen, die (a) einen cfe-Error ausloesen, (b) typ-foermig
    sind und (c) in KEINER Quelle (techenv/asm/header/base) vorkommen = sicher halluziniert.
    KONSERVATIV: nur Tokens aus den Fehlerzeilen, nur typ-foermige, nur garantiert unsourced."""
    lines = error_log.splitlines()
    found = {}
    for i, ln in enumerate(lines):
        m = _ERRLINE.search(ln)
        if not m or m.group(1) != "Error":
            continue
        msg = m.group(2)
        # nur Fehlerklassen, die von einem Typnamen kommen koennen
        if not re.search(r"Syntax Error|undefined|Selector requires|member of structure|"
                         r"declaration specifiers|incomplete type", msg):
            continue
        src = lines[i + 1] if i + 1 < len(lines) else ""
        # Trunkierte Quellzeilen (Log-Cap-Notiz) verfaelschen Tokens -> ueberspringen
        if "gekuerzt" in src or "[Log" in src or "Behebe" in src:
            continue
        for cand in _type_candidates(src):
            if cand in found:
                continue
            if _type_shaped(cand) and not is_sourced(cand, func_name):
                found[cand] = True
    # Trunkierungs-Artefakte: kurzes Token, das echtes Praefix eines laengeren ist, verwerfen
    names = set(found)
    out = [t for t in names if not any(o != t and o.startswith(t) for o in names)]
    return sorted(out)


# ---- FELD-Halluzinationen (Hebel 1): erfundene Struct-FELDNAMEN ------------------
_ARROW_FIELD = re.compile(r"->\s*([A-Za-z_]\w*)")
_DOT_FIELD = re.compile(r"(?<![\w.])\.\s*([A-Za-z_]\w*)")
_FIELD_ERR = re.compile(r"undefined|Selector requires|member of structure|has no member")


def _field_shaped(f):
    """Konservativ: wie ein erfundener FELDname (klein/camelCase/snake), kein Typ/Var/Symbol/unkNN."""
    if len(f) < 3 or f in _BASE:
        return False
    if f.startswith(("func_", "local_", "param_", "D_", "B_", "temp_", "var_")):
        return False
    if re.fullmatch(r"unk[0-9A-Fa-f]+(?:_\d+)?", f):   # unkNN -> deterministisch behandelt
        return False
    return f[0].islower() and any(c.islower() for c in f)   # Feld beginnt klein


def unsourced_field_names(error_log, func_name):
    """-> sortierte Liste der FELDnamen (`ptr->feld` / `.feld`), die einen feldbezogenen cfe-Error
    ausloesen UND in keiner Quelle (techenv/asm/header) vorkommen = erfundenes Feld. Gleiche
    Ground-Truth wie der Typ-Detektor, nur eine Ebene tiefer. KONSERVATIV (Feld beginnt klein,
    kein unkNN)."""
    lines = error_log.splitlines()
    found = {}
    for i, ln in enumerate(lines):
        m = _ERRLINE.search(ln)
        if not m or m.group(1) != "Error" or not _FIELD_ERR.search(m.group(2)):
            continue
        src = lines[i + 1] if i + 1 < len(lines) else ""
        if "gekuerzt" in src or "[Log" in src or "Behebe" in src:
            continue
        cands = _ARROW_FIELD.findall(src) + _DOT_FIELD.findall(src)
        # auch der in 'X' genannte Identifier (cfe meldet das Feld oft als 'X' undefined)
        q = re.search(r"'([A-Za-z_]\w*)'", m.group(2))
        if q:
            cands.append(q.group(1))
        for f in cands:
            if f not in found and _field_shaped(f) and not is_sourced(f, func_name):
                found[f] = True
    return sorted(found)
