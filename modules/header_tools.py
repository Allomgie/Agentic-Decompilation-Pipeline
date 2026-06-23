"""READ-ONLY Header-Werkzeuge fuer den agentischen Domaenen-Schliesser (TYP-Agent zuerst).

Sucht/liest die Spiel-Header unter data/header/ (= der Header-Root, gegen den ido_compiler kompiliert).
Zweck: dem Agenten ECHTE Struct-/Feldtypen geben, statt Element-Groessen raten zu lassen. WICHTIG: Drafts
enthalten manchmal von der KI HALLUZINIERTE Struct-Namen -> findet die Suche den Namen NICHT, ist er
wahrscheinlich halluziniert (das signalisieren wir explizit). NUR LESEN, nie schreiben.

API (fuer den Loop):
  search_symbol(name)   -> {found, kind, file, line, definition} ODER {found:False, hint:"...halluziniert"}
  read_header(name)     -> {found, file, includes[], text}  (bounded)
  grep_headers(pattern) -> Liste (file, line, text)  (z.B. nach einem Feldnamen suchen)
"""
import os, re, functools

_PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HEADER_ROOT = os.path.join(_PIPELINE_ROOT, "data", "header")


@functools.lru_cache(maxsize=1)
def _header_files():
    out = []
    for root, _dirs, files in os.walk(_HEADER_ROOT):
        for f in files:
            if f.endswith((".h", ".inc")):
                out.append(os.path.join(root, f))
    return out


def _rel(p):
    return os.path.relpath(p, _HEADER_ROOT)


def _extract_block(text, start):
    """Ab dem '{' bei oder nach `start` den balancierten {..}-Block + folgendes ';' zurueck (inkl. Namen)."""
    b = text.find("{", start)
    if b < 0:
        return None
    depth = 0
    for j in range(b, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                k = text.find(";", j)
                return text[start:(k + 1 if 0 <= k < j + 80 else j + 1)]
    return None


def search_symbol(name):
    """Findet die Definition eines Typ-/Struct-/Enum-/Union-/typedef-/#define-Namens in den Headern."""
    if not re.fullmatch(r"[A-Za-z_]\w*", name or ""):
        return {"found": False, "hint": f"'{name}' ist kein gueltiger C-Bezeichner."}
    n = re.escape(name)
    # Reihenfolge: aggregierte Typen (mit Body) zuerst, dann Aliase/Defines
    pats = [
        ("struct/union/enum (named)", re.compile(r"\b(struct|union|enum)\s+" + n + r"\s*\{")),
        ("typedef aggregate", re.compile(r"\btypedef\s+(struct|union|enum)\b[^;{]*\{")),  # Body, Name am Ende geprueft
        ("typedef alias", re.compile(r"\btypedef\b[^;{]*\b" + n + r"\s*(\[[^\]]*\])?\s*;")),
        ("#define", re.compile(r"^[ \t]*#[ \t]*define[ \t]+" + n + r"\b", re.M)),
    ]
    for f in _header_files():
        try:
            txt = open(f, errors="replace").read()
        except Exception:
            continue
        # named struct/union/enum
        m = pats[0][1].search(txt)
        if m:
            blk = _extract_block(txt, m.start())
            if blk:
                return {"found": True, "kind": "aggregate", "file": _rel(f),
                        "line": txt[:m.start()].count("\n") + 1, "definition": blk[:2500]}
        # typedef struct {...} NAME;  -> Body + Name-am-Ende muss matchen
        for tm in pats[1][1].finditer(txt):
            blk = _extract_block(txt, tm.start())
            if blk and re.search(r"\}\s*" + n + r"\s*;", blk):
                return {"found": True, "kind": "typedef-aggregate", "file": _rel(f),
                        "line": txt[:tm.start()].count("\n") + 1, "definition": blk[:2500]}
        # typedef alias
        m = pats[2][1].search(txt)
        if m:
            return {"found": True, "kind": "typedef", "file": _rel(f),
                    "line": txt[:m.start()].count("\n") + 1, "definition": txt[m.start():m.end()].strip()}
        # #define
        m = pats[3][1].search(txt)
        if m:
            ln = txt[m.start():].split("\n", 1)[0]
            return {"found": True, "kind": "define", "file": _rel(f),
                    "line": txt[:m.start()].count("\n") + 1, "definition": ln.strip()}
    return {"found": False,
            "hint": f"'{name}' nicht in den Headern gefunden -> wahrscheinlich ein halluzinierter/ungueltiger Name. "
                    f"Verwende einen echten Typ oder leite den Typ aus dem Diff (Load/Store-Breite, Stride) ab."}


def read_header(name, max_lines=400):
    """Liest einen Header per Dateiname (read-only, bounded) + listet seine #include-Namen (fuer rekursives Folgen)."""
    base = os.path.basename(name)
    hit = next((f for f in _header_files() if os.path.basename(f) == base), None)
    if not hit:
        return {"found": False, "hint": f"Header '{name}' existiert nicht unter data/header/."}
    lines = open(hit, errors="replace").read().splitlines()
    includes = re.findall(r'#\s*include\s*[<"]([^">]+)[">]', "\n".join(lines))
    return {"found": True, "file": _rel(hit), "includes": includes,
            "text": "\n".join(lines[:max_lines]) + ("" if len(lines) <= max_lines else f"\n... (+{len(lines)-max_lines} Zeilen)")}


def grep_headers(pattern, cap=40):
    """Regex-Suche ueber alle Header (z.B. nach einem Feldnamen). -> Liste (file, line, text)."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return [("<error>", 0, f"ungueltiges Regex: {e}")]
    out = []
    for f in _header_files():
        try:
            for i, ln in enumerate(open(f, errors="replace"), 1):
                if rx.search(ln):
                    out.append((_rel(f), i, ln.rstrip()[:160]))
                    if len(out) >= cap:
                        return out
        except Exception:
            continue
    return out


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: header_tools.py search <Name> | read <header.h> | grep <regex>"); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "search":
        print(json.dumps(search_symbol(sys.argv[2]), indent=2, ensure_ascii=False))
    elif cmd == "read":
        r = read_header(sys.argv[2]); print(r.get("file"), "includes:", r.get("includes"));
        print(r.get("text", r.get("hint", ""))[:1500])
    elif cmd == "grep":
        for x in grep_headers(sys.argv[2]):
            print(x)
