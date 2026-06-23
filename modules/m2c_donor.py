"""m2c als deterministischer Synthese-Donor (vom Target). Phase-0c, parallel zum Similar-Transplant.

run_m2c_ido(): laeuft m2c mit `--valid-syntax` (unbekannte Typen/Casts/Felder als Makros) und
NORMALISIERT den Output zu IDO-cfe-(C89)-kompatiblem C:
  - Makros voranstellen (M2C_UNK=s32, M2C_BITWISE=(T)(e), M2C_FIELD(e,t*,o)=`(*(t*)((s8*)e+o))`
    = unsere explizite-Offset-Form, KEIN erfundener Struct).
  - C89-Fix: `case/default/label:` direkt vor `}` -> `;` einfuegen (Label braucht ein Statement).
Liefert IDO-kompilierbaren C, der die korrekte Struktur AUS DEM TARGET enthaelt -> Synthese-Donor.
"""
import os, re, subprocess

_M2C_DIR = os.path.join(os.path.dirname(__file__), "m2c")
_M2C_PY = os.path.join(_M2C_DIR, "m2c.py")
_MACROS_PATH = os.path.join(_M2C_DIR, "m2c_macros.h")


def _macros():
    return open(_MACROS_PATH, encoding="utf-8").read()


def normalize_m2c(c):
    """m2c-Output -> IDO-cfe(C89)-kompatibel."""
    # (1) m2cs INFERIERTE Callee-Prototypen entfernen (`... func(...); /* extern */`) -- ihre
    #     Arg-Zahlen widersprechen oft den ECHTEN Signaturen aus den Build-Headern. Daten-externs
    #     (`extern ... D_X;` ohne /* extern */-Marker bzw. ohne '(') bleiben.
    c = "\n".join(l for l in c.splitlines()
                  if not (l.rstrip().endswith("/* extern */") and "(" in l))
    # (2) C89: Label (`case X:`, `default:`, `name:`) MUSS ein Statement haben -> `;` vor `}`/Label.
    #     Kommentar-tolerant ( m2c haengt `/* switch N */` an Labels).
    _cm = r"(?:\s|/\*.*?\*/)*"          # whitespace ODER /* ... */
    c = re.sub(r"((?:case\s+[^:;{}]+|default)\s*:)(" + _cm + r"\})", r"\1 ;\2", c)
    c = re.sub(r"(^[ \t]*[A-Za-z_]\w*\s*:)(" + _cm + r"\})", r"\1 ;\2", c, flags=re.M)
    c = re.sub(r"((?:case\s+[^:;{}]+|default)\s*:)(" + _cm + r"(?:case\s|default\s*:))", r"\1 ;\2", c)
    # (3) void*-Arithmetik (IDO lehnt ab): `void *` -> `u8 *` (arithmetik-faehig, gleiche Adresse).
    #     NOETIG auch in Deklarationen, da m2c `param + n` auf void*-Params macht (sonst 7 Fails).
    c = c.replace("(void *)", "(u8 *)").replace("void *", "u8 *")
    return c


def run_m2c_ido(target_s_path, func_name=None, context_c=None, timeout=90):
    """Decompile target .s -> IDO-kompilierbaren C (valid-syntax + Makros + Normalisierung).
    Raises RuntimeError bei m2c-Fehler (z.B. echter Kontrollfluss-Fail)."""
    cmd = ["python3", _M2C_PY, "-t", "mips-ido-c", "--valid-syntax", "--globals", "used"]
    if context_c and os.path.exists(context_c):
        cmd += ["--context", context_c]
    cmd += [os.path.abspath(target_s_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=_M2C_DIR, timeout=timeout)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError("m2c failed:\n" + (p.stderr or p.stdout)[:800])
    return _macros() + "\n" + normalize_m2c(p.stdout)


def _expand_call(code, name, tmpl):
    """Ersetzt alle name(a,b,...)-Aufrufe paren-genau (rekursiv via Wiederholung; aeussere zuerst, innere
    bleiben als Text -> naechste Runde). tmpl(args)->str."""
    guard = 0
    while True:
        i = code.find(name + "(")
        if i < 0 or guard > 200000:
            return code
        guard += 1
        depth = 0; args = []; buf = []; end = -1; k = i + len(name)
        while k < len(code):
            c = code[k]
            if c == "(":
                depth += 1
                if depth == 1:
                    k += 1; continue
            elif c == ")":
                depth -= 1
                if depth == 0:
                    args.append("".join(buf).strip()); end = k + 1; break
            if depth == 1 and c == ",":
                args.append("".join(buf).strip()); buf = []
            else:
                buf.append(c)
            k += 1
        if end < 0:
            return code                       # unbalanciert -> unveraendert (Cleaner schlaegt fehl -> None)
        try:
            code = code[:i] + tmpl(args) + code[end:]
        except Exception:
            return code


def clean_m2c(code):
    """m2c-Transplantat -> SAUBERES C ohne Makros/Kommentare/#define (Pflicht fuer Similar!) + Namens-
    Konvention. Inlined M2C_FIELD/M2C_BITWISE/M2C_UNK* (= explizite *(T*)((char*)e+off)-Form, wie der Rest),
    streift den Makro-Header + ALLE Kommentare, benennt arg{N}->param_{N}. Gibt None zurueck, wenn danach noch
    M2C_/#define/#ifndef/Kommentare uebrig sind (dann ist das Transplantat NICHT verwertbar -> Aufrufer verwirft
    es, statt schmutzig zu speichern). SEMANTISCH identisch (Inlining = was der Praeprozessor eh tut) -> .o
    gleich; Aufrufer gated zusaetzlich ueber das Orakel."""
    mh = _macros()
    if code.startswith(mh):
        code = code[len(mh):]
    # Inlining der gaengigen m2c-Makros
    code = _expand_call(code, "M2C_FIELD", lambda a: f"(*({a[1]})((s8 *)({a[0]}) + ({a[2]})))" if len(a) == 3 else a)
    code = _expand_call(code, "M2C_BITWISE", lambda a: f"(({a[0]})({a[1]}))" if len(a) == 2 else a)
    # M2C_UNK*-Typedef-Zeilen EXPLIZIT entfernen (falls der startswith-Header-Strip nicht griff, z.B. wenn
    # etwas vor dem Header steht). Sonst macht die folgende M2C_UNK->Typ-Substitution aus `typedef s32 M2C_UNK;`
    # ein ungueltiges `typedef s32 s32;` (Selbst-Typedef -> compile-fail). MUSS vor der Substitution stehen.
    code = re.sub(r"(?m)^[ \t]*typedef[ \t]+\w+[ \t]+M2C_UNK\w*[ \t]*;[ \t]*$", "", code)
    for mac, ty in (("M2C_UNK64", "s64"), ("M2C_UNK32", "s32"), ("M2C_UNK16", "s16"),
                    ("M2C_UNK8", "s8"), ("M2C_UNK", "s32")):
        code = re.sub(r"\b" + mac + r"\b", ty, code)
    # WEITERE m2c-Stub-Makros inlinen -- EXAKT wie m2c_macros.h (= semantik-erhaltend, .o identisch). Der
    # Header definiert diese; ohne Inlining bleiben sie nach dem Header-Strip undefiniert (kompiliert nicht).
    # Viele sind 0-Stubs (m2c konnte die Instruktion nicht uebersetzen) -- Inlining behaelt exakt diese Semantik.
    code = _expand_call(code, "GLUE_F64", lambda a: "(0.0)")
    for mac in ("M2C_LWL", "M2C_FIRST3BYTES", "M2C_UNALIGNED32"):   # #define X(expr) (expr) -> Durchreiche
        code = _expand_call(code, mac, lambda a: f"({a[0]})" if a and a[0] else "(0)")
    for mac in ("M2C_ERROR", "M2C_TRAP_IF", "M2C_BREAK", "M2C_SYNC", "M2C_OVERFLOW", "MULTU_HI", "MULT_HI",
                "DMULTU_HI", "DMULT_HI", "CLZ", "REVERSE_BITS", "ROTATE_RIGHT", "ARM_RRX",
                "BSWAP32", "BSWAP16X2", "BSWAP16"):                 # #define X(...) (0)
        code = _expand_call(code, mac, lambda a: "(0)")
    code = re.sub(r"\bM2C_CARRY\b", "0", code)
    code = re.sub(r"\b(?:M2C_MEMCPY_ALIGNED|M2C_MEMCPY_UNALIGNED|M2C_STRUCT_COPY)\b", "memcpy", code)
    # ALLE Kommentare weg (Header-Block + m2c-Inline wie /* switch N */ /* irregular */)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    # Praeprozessor-Zeilen weg (#ifndef/#define/#endif Reste)
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))
    # arg{N} -> param_{N} (Konvention; m2c nutzt arg, kein param -> kollisionsfrei)
    code = re.sub(r"\barg(\d+)\b", r"param_\1", code)
    # Leerzeilen-Salat einkuerzen
    code = re.sub(r"\n{3,}", "\n\n", code).strip() + "\n"
    if ("M2C_" in code or "#define" in code or "#ifndef" in code or "/*" in code
            or re.search(r"\b(?:GLUE_F64|MULTU?_HI|DMULTU?_HI|CLZ|REVERSE_BITS|ROTATE_RIGHT|ARM_RRX|BSWAP[0-9X]+)\b", code)):
        return None                           # nicht vollstaendig saeuberbar (Rest-Makro) -> Transplantat verwerfen
    return code
